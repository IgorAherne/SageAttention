/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Spellcast3D: a twin of qk_int_sv_i8_cuda_sm80.cu's int8 P.V kernel (op ..._sc3d) with the same arithmetic in another
// instruction schedule (op qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d2). Every value is computed by the same operations
// in the same order: the online softmax block by block, the same rescale points, P_u8 = round(P * 255) by the same
// FFMA, the int32 P.V sum per 64-key block; only where the instructions sit relative to each other and which
// instructions carry a value from one form to another (where the result is the same bit for bit) differ. The two ops
// take the same arguments (V^T, scales and mean from quant_per_block_int8_v_cuda) and give the same bits.
//
// Not part of the package build (no op registration, not in setup.py): on an RTX 3060 the chosen schedule (the
// defaults below) is 3.2-5.1 % faster than the package kernel, below the 5 % an install race would need to pay for
// itself; the other knobs are the measured alternatives and timing studies.

#include "../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include "../cp_async.cuh"
#include "../mma.cuh"
#include "../permuted_smem.cuh"
#include "../math.cuh"
#include "../dispatch_utils.h"

#include "attn_utils.cuh"

// The chosen schedule (bits equal to the package kernel's):
// who loads what: 0 the package kernel's (group A V^T(w) inside trip w, then a group wait before its P.V; group B
// K(w + 1)); 1 group B both K(w + 1) and V^T(w + 1); 2 group A K(w + 1) at the start of its trip, group B V^T(w + 1)
// right after its P.V of block w - 1 (a group barrier first: the last reader of that buffer), so that no group waits
// inside a trip; 3 group A both, V^T(w + 1) at the end of its trip
#ifndef SC3D2_LOADS
#define SC3D2_LOADS 2
#endif
// 1: K tiles by load_k_tile (no bound checks but in the last block; the 12-head NHD row stride a constant)
#ifndef SC3D2_KLOAD
#define SC3D2_KLOAD 1
#endif

// Measured and left out (bits equal, not faster):
// the int32 sums of S (1) and of the P.V blocks (2) started from the bits of 1.5 * 2^23 held in four registers (the
// MMAs' C operand) and read by one FADD instead of an I2F: the compiler re-creates the constants (MOVs) and spills
#ifndef SC3D2_MAGIC
#define SC3D2_MAGIC 0
#endif
// where group A waits for its V^T with SC3D2_LOADS 0: 0 where the compiler puts it, 1 after the softmax, past a branch
#ifndef SC3D2_VWAIT
#define SC3D2_VWAIT 0
#endif
// 1: the accumulators rescaled in every block (by exactly 1 in rows without a new maximum), no branch: 64 more FMULs
#ifndef SC3D2_RESC
#define SC3D2_RESC 0
#endif
// with SC3D2_LOADS 1 / 2: group B issues V^T(w + 1) after its Q.K^T of block w instead of before (spills)
#ifndef SC3D2_VLATE
#define SC3D2_VLATE 0
#endif
// with SC3D2_LOADS 2: where group A issues K(w + 1): 0 at the start of its trip, 1 after its Q.K^T, 2 after its softmax
#ifndef SC3D2_KLATE
#define SC3D2_KLATE 0
#endif
// 1: V^T tiles one channel row a thread (load_v_rows: four copies from one pointer)
#ifndef SC3D2_VLOAD
#define SC3D2_VLOAD 0
#endif
// 1: V^T in the blocked layout [B, H, Npad / 64, 128, 64] (one 8 KB piece a block; needs its own quantiser)
#ifndef SC3D2_VBLK
#define SC3D2_VBLK 0
#endif
// 1: V^T tiles by load_v_tile (load_fp8_V_global_to_share's addresses, offsets as immediates)
#ifndef SC3D2_VLOCAL
#define SC3D2_VLOCAL 0
#endif
// global -> shared copies through L1 (cp.async.ca): 1 K, 2 V^T (with SC3D2_VLOCAL); else past L1 as the package
#ifndef SC3D2_CA
#define SC3D2_CA 0
#endif
// a start delay in ns for every other wave of blocks (0 = none): the two blocks of an SM half a block apart
#ifndef SC3D2_DEPHASE
#define SC3D2_DEPHASE 0
#endif
// 1: blocks of 16 warps (256 query rows, one block an SM: each K / V^T tile serves twice the rows), else 8 (two an SM)
#ifndef SC3D2_W16
#define SC3D2_W16 0
#endif
// minimum blocks an SM for __launch_bounds__
#ifndef SC3D2_MINB
#define SC3D2_MINB 2
#endif

// Timing studies only (0 in the package; wrong answers otherwise). SC3D2_ABL, a bit mask: 1 V^T loaded for the first
// two blocks only, 2 K likewise, 4 no P.V MMAs, 8 no conversion of their products, 16 no ex2 in the softmax, 32 no
// barriers a block (with 1 and 2 only), 64 no softmax (P from S's bits), 128 K and V^T streamed from the first two
// blocks over and over (the same copies, always in L2)
#ifndef SC3D2_ABL
#define SC3D2_ABL 0
#endif
// SC3D2_PAD dummy instructions of class SC3D2_PADK (0 FFMA, 1 an IADD3 + LOP3 pair, 2 MUFU.EX2, 3 I2F, 4 PRMT,
// 5 FMNMX) a block and warp, a dependent chain on one register, after the softmax (SC3D2_PADAT 0) or after the P.V
// step (1), in both groups (SC3D2_PADG 0), group A only (1) or group B only (2)
#ifndef SC3D2_PAD
#define SC3D2_PAD 0
#endif
#ifndef SC3D2_PADK
#define SC3D2_PADK 0
#endif
#ifndef SC3D2_PADAT
#define SC3D2_PADAT 0
#endif
#ifndef SC3D2_PADG
#define SC3D2_PADG 0
#endif

#if SC3D2_W16 && (SC3D2_VLOAD || SC3D2_VBLK)
#error "SC3D2_VLOAD / SC3D2_VBLK assume groups of four warps"
#endif

namespace sc3d_i8s {

constexpr uint32_t CTA_Q = SC3D2_W16 ? 256 : 128;     // 16 warps of 16 rows (one block an SM) or 8 (two)
constexpr uint32_t CTA_K = 64;
constexpr uint32_t WARP_Q = 16;
constexpr uint32_t HEAD = 128;
constexpr uint32_t NUM_WARPS = CTA_Q / WARP_Q;
constexpr uint32_t THREADS = 32 * NUM_WARPS;
constexpr uint32_t TILES_K = CTA_K / 16;         // 16-key tiles of S per warp
constexpr uint32_t TILES_QK_INNER = HEAD / 32;   // k32 steps of Q.K^T
constexpr uint32_t TILES_V = HEAD / 16;          // 16-channel tiles of O
constexpr uint32_t GROUP_WARPS = NUM_WARPS / 2;  // the warps of one ping-pong group
constexpr uint32_t GROUP_THREADS = 32 * GROUP_WARPS;
constexpr uint32_t K_ITERS = CTA_K / (4 * GROUP_WARPS);   // a group's K tile: 4 rows a warp and step
constexpr uint32_t V_ITERS = HEAD / (8 * GROUP_WARPS);    // a group's V^T tile: 8 channel rows a warp and step
constexpr uint32_t Q_GROUPS_PER_128 = 128 / WARP_Q;       // q scales: per_warp_int8 makes ceil(N / 128) * 8 a head

// Q int8 [128][128], two K int8 [64][128], two V^T int8 [128][64], two blocks' v scales fp32 [128]; O bf16 [128][128]
// at the end
constexpr uint32_t K_BUF_UNITS = CTA_K * HEAD / 16;
constexpr uint32_t V_BUF_UNITS = HEAD * CTA_K / 16;
constexpr uint32_t SMEM_V_OFFSET = (CTA_Q + 2 * CTA_K) * HEAD;
constexpr uint32_t SMEM_VS_OFFSET = SMEM_V_OFFSET + 2 * HEAD * CTA_K;
constexpr size_t SMEM_BYTES = (SMEM_VS_OFFSET + 2 * HEAD * sizeof(float)) > (CTA_Q * HEAD * sizeof(nv_bfloat16))
                              ? (SMEM_VS_OFFSET + 2 * HEAD * sizeof(float)) : (CTA_Q * HEAD * sizeof(nv_bfloat16));

constexpr int32_t MAGIC_I = 0x4B400000;          // the bits of 1.5 * 2^23
constexpr float MAGIC_F = 12582912.0f;

__device__ __forceinline__ void pad(uint32_t &x)
{
#pragma unroll
  for (int i = 0; i < SC3D2_PAD; i++)
  {
    if constexpr (SC3D2_PADK == 0) asm volatile("fma.rn.f32 %0, %0, %0, 0f3F000000;" : "+r"(x));
    if constexpr (SC3D2_PADK == 1) asm volatile("{.reg .u32 t; add.u32 t, %0, %0; xor.b32 %0, t, %0;}" : "+r"(x));
    if constexpr (SC3D2_PADK == 2) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+r"(x));
    if constexpr (SC3D2_PADK == 3) asm volatile("{.reg .f32 t; cvt.rn.f32.s32 t, %0; mov.b32 %0, t;}" : "+r"(x));
    if constexpr (SC3D2_PADK == 4) asm volatile("prmt.b32 %0, %0, %0, 0x1230;" : "+r"(x));
    if constexpr (SC3D2_PADK == 5) asm volatile("max.f32 %0, %0, 0f3F000000;" : "+r"(x));
  }
}

__device__ __forceinline__ float ex2(float x)
{
#if SC3D2_ABL & 16
  return x * 0.5f;
#else
  return math::ptx_exp2(x);
#endif
}

// round(x * 255) of four x in [0, 1] as four bytes, lowest first (the package kernel's pack_u8x4)
__device__ __forceinline__ uint32_t pack_u8x4(float a, float b, float c, float e)
{
  const uint32_t ua = __float_as_uint(fmaf(a, 255.0f, 12582912.0f));
  const uint32_t ub = __float_as_uint(fmaf(b, 255.0f, 12582912.0f));
  const uint32_t uc = __float_as_uint(fmaf(c, 255.0f, 12582912.0f));
  const uint32_t ue = __float_as_uint(fmaf(e, 255.0f, 12582912.0f));
  return __byte_perm(__byte_perm(ua, ub, 0x0040), __byte_perm(uc, ue, 0x0040), 0x5410);
}

// D = A . B + C, m16n8k32, C and D in different registers
template <bool kUnsignedA>
__device__ __forceinline__ void mma_k32_c(int32_t *D, const uint32_t *A, const uint32_t *B, const int32_t *C)
{
  if constexpr (kUnsignedA)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                   "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
  else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
                 : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                   "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
}

// Q.K^T of a warp's 16 rows and the block's 64 keys (compute_int_qk's order: k32 step by step, the 16-key tiles in
// order); with SC3D2_MAGIC & 1 the first step adds the magic quad instead of zero
__device__ __forceinline__ void compute_qk(const smem_t<SwizzleMode::k128B, HEAD / 16> &smem_Q,
                                           const smem_t<SwizzleMode::k128B, HEAD / 16> &smem_K,
                                           int32_t RS[][TILES_K][8], uint32_t offset_Q, uint32_t offset_K,
                                           const int32_t *magic)
{
  uint32_t RQ[4];
  uint32_t RK[4];
#pragma unroll
  for (uint32_t iter = 0; iter < TILES_QK_INNER; iter++)
  {
    smem_Q.ldmatrix_m8n8x4(offset_Q, RQ);
    offset_Q = smem_Q.advance_offset_by_column<2>(offset_Q, iter);
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);
      if (iter == 0)
      {
#if SC3D2_MAGIC & 1
        mma_k32_c<false>(RS[0][fk], RQ, RK, magic);
        mma_k32_c<false>(RS[0][fk] + 4, RQ, RK + 2, magic);
#else
        mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[0][fk], RQ, RK);
#endif
      }
      else
      {
        mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(RS[0][fk], RQ, RK);
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(offset_K - (TILES_K * 16 * (HEAD / 16)), iter);
  }
}

__device__ __forceinline__ float s_float(const int32_t x)
{
#if SC3D2_MAGIC & 1
  return __int_as_float(x) - MAGIC_F;
#else
  return __int2float_rz(x);
#endif
}

__device__ __forceinline__ float pv_float(const int32_t x)
{
#if SC3D2_MAGIC & 2
  return __int_as_float(x) - MAGIC_F;
#else
  return __int2float_rn(x);
#endif
}

// the package kernel's update_mdo_blk: m and d of the online softmax, o_scale = exp2(m_prev - m) per row with whether
// any row of the warp has a new maximum, P relative to the block's maximum, pmax = exp2(m_blk - m), d += pmax * this
// thread's part of the block's row sum
__device__ __forceinline__ void update_mdo_blk(float RS[][TILES_K][8], float m[][2], float d[][2], float pmax[2], float o_scale[2], bool &rescale, const float &sm_scale)
{
  bool new_max = false;
#pragma unroll
  for (uint32_t k = 0; k < 2; k++)
  {
    float m_prev = m[0][k];
    float m_temp = -5000000.0f;
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
      float m_local = max(max(RS[0][fk][k * 2 + 0], RS[0][fk][k * 2 + 1]),
                          max(RS[0][fk][k * 2 + 4], RS[0][fk][k * 2 + 5]));
      m_temp = max(m_temp, m_local);
    }
    m_temp *= sm_scale;
    m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
    m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));

    m[0][k] = max(m[0][k], m_temp);
    new_max |= (m[0][k] != m_prev);
    o_scale[k] = ex2(m_prev - m[0][k]);
    d[0][k] *= o_scale[k];

    pmax[k] = ex2(m_temp - m[0][k]);
    float negative_m = -m_temp;
    float rowsum = 0.0f;
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
      RS[0][fk][k * 2 + 0] = ex2(fmaf(RS[0][fk][k * 2 + 0], sm_scale, negative_m));
      RS[0][fk][k * 2 + 1] = ex2(fmaf(RS[0][fk][k * 2 + 1], sm_scale, negative_m));
      RS[0][fk][k * 2 + 4] = ex2(fmaf(RS[0][fk][k * 2 + 4], sm_scale, negative_m));
      RS[0][fk][k * 2 + 5] = ex2(fmaf(RS[0][fk][k * 2 + 5], sm_scale, negative_m));
      rowsum += RS[0][fk][k * 2 + 0] + RS[0][fk][k * 2 + 1] + RS[0][fk][k * 2 + 4] + RS[0][fk][k * 2 + 5];
    }
    d[0][k] = fmaf(pmax[k], rowsum, d[0][k]);
  }
  rescale = __any_sync(0xffffffff, new_max);
}

// P as the u8 A fragments of the block's two k32 steps (the package kernel's pack_p)
__device__ __forceinline__ void pack_p(float RS[][TILES_K][8], uint32_t RP[2][4])
{
#pragma unroll
  for (uint32_t kk = 0; kk < 2; kk++)
  {
    const float *a = RS[0][2 * kk], *b = RS[0][2 * kk + 1];
    RP[kk][0] = pack_u8x4(a[0], a[1], a[4], a[5]);
    RP[kk][1] = pack_u8x4(a[2], a[3], a[6], a[7]);
    RP[kk][2] = pack_u8x4(b[0], b[1], b[4], b[5]);
    RP[kk][3] = pack_u8x4(b[2], b[3], b[6], b[7]);
  }
}

// P.V of one block (the package kernel's compute_sv_t): per 16-channel tile the int32 product over the 64 keys, then
// RO = product * (pmax / 255) * v_scale + RO (* o_scale with kRescale) per row and channel
template <bool kRescale>
__device__ __forceinline__ void compute_sv_t(const smem_t<SwizzleMode::k64B, 4> &smem_V, const float *smem_VS, uint32_t RP[2][4],
                                             float RO[][TILES_V][8], const float pmax[2], const float o_scale[2],
                                             const uint32_t offset_V0, const uint32_t offset_V1, const int32_t *magic)
{
  const float f0 = pmax[0] * (1.0f / 255.0f);
  const float f1 = pmax[1] * (1.0f / 255.0f);
  const float *vs = smem_VS + 2 * (get_lane_id() % 4);
#pragma unroll
  for (uint32_t fv = 0; fv < TILES_V; fv++)
  {
    int32_t acc[8];
#if SC3D2_ABL & 4
#pragma unroll
    for (uint32_t i = 0; i < 8; i++)
      acc[i] = (int32_t)(RP[i / 4][i % 4] ^ fv);
#else
    uint32_t RV[4];
    smem_V.ldmatrix_m8n8x4(offset_V0 + fv * 16 * 4, RV);
#if SC3D2_MAGIC & 2
    mma_k32_c<true>(acc, RP[0], RV, magic);
    mma_k32_c<true>(acc + 4, RP[0], RV + 2, magic);
#else
    mma::mma_sync_m16n16k32_row_col_u8s8s32<mma::MMAMode::kInit>(acc, RP[0], RV);
#endif
    smem_V.ldmatrix_m8n8x4(offset_V1 + fv * 16 * 4, RV);
    mma::mma_sync_m16n16k32_row_col_u8s8s32<mma::MMAMode::kInplaceUpdate>(acc, RP[1], RV);
#endif
#if SC3D2_ABL & 8
#pragma unroll
    for (uint32_t i = 0; i < 8; i++)
      RO[0][fv][i] += __int_as_float(acc[i]);
#else
    const float2 sa = *(const float2*)(vs + fv * 16);
    const float2 sb = *(const float2*)(vs + fv * 16 + 8);
    RO[0][fv][0] = fmaf(pv_float(acc[0]), f0 * sa.x, kRescale ? RO[0][fv][0] * o_scale[0] : RO[0][fv][0]);
    RO[0][fv][1] = fmaf(pv_float(acc[1]), f0 * sa.y, kRescale ? RO[0][fv][1] * o_scale[0] : RO[0][fv][1]);
    RO[0][fv][2] = fmaf(pv_float(acc[2]), f1 * sa.x, kRescale ? RO[0][fv][2] * o_scale[1] : RO[0][fv][2]);
    RO[0][fv][3] = fmaf(pv_float(acc[3]), f1 * sa.y, kRescale ? RO[0][fv][3] * o_scale[1] : RO[0][fv][3]);
    RO[0][fv][4] = fmaf(pv_float(acc[4]), f0 * sb.x, kRescale ? RO[0][fv][4] * o_scale[0] : RO[0][fv][4]);
    RO[0][fv][5] = fmaf(pv_float(acc[5]), f0 * sb.y, kRescale ? RO[0][fv][5] * o_scale[0] : RO[0][fv][5]);
    RO[0][fv][6] = fmaf(pv_float(acc[6]), f1 * sb.x, kRescale ? RO[0][fv][6] * o_scale[1] : RO[0][fv][6]);
    RO[0][fv][7] = fmaf(pv_float(acc[7]), f1 * sb.y, kRescale ? RO[0][fv][7] * o_scale[1] : RO[0][fv][7]);
#endif
  }
}

__device__ __forceinline__ void compute_sv(const smem_t<SwizzleMode::k64B, 4> &smem_V, const float *smem_VS, uint32_t RP[2][4],
                                           float RO[][TILES_V][8], const float pmax[2], const float o_scale[2], const bool rescale,
                                           const uint32_t offset_V0, const uint32_t offset_V1, const int32_t *magic)
{
#if SC3D2_RESC
  // o_scale is exp2(0) = 1 exactly in every row without a new maximum: rescaling always gives the same bits
  compute_sv_t<true>(smem_V, smem_VS, RP, RO, pmax, o_scale, offset_V0, offset_V1, magic);
#else
  if (rescale)
    compute_sv_t<true>(smem_V, smem_VS, RP, RO, pmax, o_scale, offset_V0, offset_V1, magic);
  else
    compute_sv_t<false>(smem_V, smem_VS, RP, RO, pmax, o_scale, offset_V0, offset_V1, magic);
#endif
}

// V^T(W) of one group's four warps, one channel row a thread (its 64 bytes as four 16-byte copies from one pointer):
// row r's chunk c lands at unit r * 4 + (c ^ ((r / 2) % 4)) of the k64B layout, which is (r * 4 + (r / 2) % 4) ^ c
__device__ __forceinline__ void load_v_rows(const int8_t *row_ptr, const uint32_t smem_addr_unit0)
{
#pragma unroll
  for (uint32_t c = 0; c < 4; c++)
  {
    const uint32_t dst = smem_addr_unit0 ^ (c * 16);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"(row_ptr + c * 16));
  }
}

// V^T(W) of one group's four warps from the blocked layout [B, H, Npad / 64, 128, 64] (a block's tile is 8 KB in one
// piece): thread t copies the 16-byte units t + 128 i (i = 0..3), unit q = channel row q / 4, chunk q % 4, which the
// k64B layout puts at row * 4 + (chunk ^ ((row / 2) % 4)) = t's first place + 128 i
__device__ __forceinline__ void load_v_blocked(const int8_t *tile_ptr, const uint32_t smem_addr)
{
#pragma unroll
  for (uint32_t i = 0; i < 4; i++)
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(smem_addr + i * 2048), "l"(tile_ptr + i * 2048));
}

// one 16-byte global -> shared copy: through L1 (.ca, kL1) or past it with an L2 prefetch hint (the package's .cg)
template <bool kL1>
__device__ __forceinline__ void cp16(const uint32_t smem_addr, const void *gptr)
{
  if constexpr (kL1)
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr));
  else
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gptr));
}

// V^T(W) of one group (load_fp8_V_global_to_share's addresses: 4 lanes a 64-byte row, 8 rows a warp and step)
template <bool kL1>
__device__ __forceinline__ void load_v_tile(int8_t *&lane_ptr, const uint32_t smem_offset, const uint32_t stride,
                                            const smem_t<SwizzleMode::k64B, CTA_K / 16> &smem)
{
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem.base);
#pragma unroll
  for (uint32_t i = 0; i < V_ITERS; i++)
    cp16<kL1>(base + (smem_offset + i * 8 * (CTA_K / 16)) * 16, lane_ptr + i * 8 * stride);
  lane_ptr += CTA_K;
}

// K(W) of one group's four warps (load_global_to_share's order and addresses: 16 rows a warp, 4 a step), with the row
// stride a compile-time constant when kStride != 0 (immediate address offsets) and no bound checks when `full`
template <uint32_t kStride>
__device__ __forceinline__ void load_k_tile(int8_t *&lane_ptr, uint32_t smem_offset, const uint32_t stride,
                                            const smem_t<SwizzleMode::k128B, HEAD / 16> &smem, const bool full,
                                            const uint32_t base_idx, const uint32_t kv_len)
{
  const uint32_t st = kStride ? kStride : stride;
  if (full)
  {
#if SC3D2_CA & 1
    const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem.base);
#pragma unroll
    for (uint32_t i = 0; i < K_ITERS; i++)
    {
      cp16<true>(base + smem_offset * 16, lane_ptr + i * 4 * st);
      smem_offset = smem.advance_offset_by_row<4>(smem_offset);
    }
#else
#pragma unroll
    for (uint32_t i = 0; i < K_ITERS; i++)
    {
      smem.load_128b_async(smem_offset, lane_ptr + i * 4 * st);
      smem_offset = smem.advance_offset_by_row<4>(smem_offset);
    }
#endif
  }
  else
  {
#pragma unroll
    for (uint32_t i = 0; i < K_ITERS; i++)
    {
      smem.load_128b_async<cp_async::SharedMemFillMode::kNoFill>(smem_offset, lane_ptr + i * 4 * st, base_idx + i * 4 < kv_len);
      smem_offset = smem.advance_offset_by_row<4>(smem_offset);
    }
  }
  lane_ptr += CTA_K * st;
}

template <typename DTypeOut, uint32_t kStrideK>
__global__ void __launch_bounds__(THREADS, SC3D2_W16 ? 1 : SC3D2_MINB) qk_int_sv_i8_attn_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V, DTypeOut *__restrict__ O,
                      float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale, float *__restrict__ V_mean,
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
                      const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
                      const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
                      float sm_scale)
{
  static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value, "DTypeOut must be half or nv_bfloat16");

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;
  const uint32_t kv_head = head_id / num_kv_groups;
  const uint32_t num_kv_heads = num_qo_heads / num_kv_groups;
  const uint32_t num_iterations = div_ceil(kv_len, CTA_K);

  sm_scale *= math::log2e;

  int32_t RS[1][TILES_K][8];
  float RO[1][TILES_V][8];
  float m[1][2];
  float d[1][2];
  const int32_t magic[4] = {MAGIC_I, MAGIC_I, MAGIC_I, MAGIC_I};

#if SC3D2_W16
  // the q scales come per 16 rows, div_ceil(qo_len, 128) * 8 a head; a warp past the last (rows past qo_len, never
  // stored) reads none
  const uint32_t q_groups = div_ceil(qo_len, 128u) * Q_GROUPS_PER_128;
  const uint32_t q_group = bx * NUM_WARPS + warp_id;
  const uint32_t q_scale_idx = (batch_id * num_qo_heads + head_id) * q_groups + (q_group < q_groups ? q_group : 0);
#else
  const uint32_t q_scale_idx = batch_id * num_qo_heads * (gridDim.x * NUM_WARPS) + head_id * (gridDim.x * NUM_WARPS) + bx * NUM_WARPS + warp_id;
#endif
  const uint32_t k_scale_idx = batch_id * num_kv_heads * num_iterations + kv_head * num_iterations;

#pragma unroll
  for (uint32_t fv = 0; fv < TILES_V; fv++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 8; k++)
    {
      RO[0][fv][k] = 0.0f;
    }
  }
#pragma unroll
  for (uint32_t k = 0; k < 2; k++)
  {
    m[0][k] = -5000000.0f;
    d[0][k] = 1.0f;
  }

  smem_t<SwizzleMode::k128B, HEAD / 16> smem_Q(smem);
  smem_t<SwizzleMode::k128B, HEAD / 16> smem_K(smem + CTA_Q * HEAD);
  smem_t<SwizzleMode::k64B, CTA_K / 16> smem_V(smem + SMEM_V_OFFSET);
  float *smem_VS = (float*)(smem + SMEM_VS_OFFSET);
  smem_t<SwizzleMode::k128B, HEAD / 8> smem_O(smem);

  // the ping-pong of the package kernel: warps 0-3 (group A) run each block w in order: the V(w) load, Q.K^T,
  // softmax, P.V; warps 4-7 (group B) half a block behind: the K(w + 1) load, P.V of block w - 1, then Q.K^T and
  // softmax of block w
  const bool group_a = warp_id < GROUP_WARPS;
  const uint32_t gw = warp_id % GROUP_WARPS;

  int8_t *Q_lane_base_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8) * stride_seq_q + (lane_id % 8) * 16;
  int8_t *KG_lane_ptr = K + batch_id * stride_bz_k + kv_head * stride_h_k + (CTA_K / GROUP_WARPS * gw + lane_id / 8) * stride_seq_k + (lane_id % 8) * 16;
  int8_t *VG_lane_ptr = V + batch_id * stride_bz_v + kv_head * stride_h_v + (HEAD / GROUP_WARPS * gw + lane_id / 4) * stride_d_v + (lane_id % 4) * 16;
  const float *VS_lane_ptr = V_scale + (batch_id * num_kv_heads + kv_head) * num_iterations * HEAD + lane_id * 4;
#if SC3D2_VBLK
  // the blocked V^T: [B, H_kv, Npad / 64, 128, 64]; stride_h_v is the head stride (Npad * 128)
  const uint32_t v_t = gw * 32 + lane_id;
  const int8_t *VB_lane_ptr = V + batch_id * stride_bz_v + kv_head * stride_h_v + v_t * 16;
  const uint32_t VB_smem_addr = (uint32_t)__cvta_generic_to_shared(smem + SMEM_V_OFFSET) + ((v_t / 4) * 4 + ((v_t % 4) ^ ((v_t / 8) % 4))) * 16;
#endif
#if SC3D2_VLOAD
  const uint32_t v_row = gw * 32 + lane_id;
  const int8_t *VR_lane_ptr = V + batch_id * stride_bz_v + kv_head * stride_h_v + v_row * stride_d_v;
  const uint32_t VR_smem_addr = (uint32_t)__cvta_generic_to_shared(smem + SMEM_V_OFFSET) + (v_row * 4 + (v_row / 2) % 4) * 16;
#endif
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * 4 * (CTA_Q / (NUM_WARPS * 4)) + lane_id / 8, lane_id % 8);
  uint32_t KG_smem_offset_load = smem_K.get_permuted_offset(gw * (CTA_K / GROUP_WARPS) + lane_id / 8, lane_id % 8);
  uint32_t VG_smem_offset_load = smem_V.get_permuted_offset(gw * (HEAD / GROUP_WARPS) + lane_id / 4, lane_id % 4);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(warp_id * WARP_Q + lane_id % 16, lane_id / 16);
  const uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  const uint32_t V_smem_offset_mma0 = smem_V.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  const uint32_t V_smem_offset_mma1 = smem_V.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2 + 2);

  const uint32_t K_idx_lane_base = 2 * (lane_id % 4);
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8;
  uint32_t KG_load_idx = CTA_K / GROUP_WARPS * gw + lane_id / 8;

  load_global_to_share<8, 4, 1, CTA_Q / (NUM_WARPS * 4), SwizzleMode::k128B, HEAD / 16, CTA_Q>(
    &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  // K(0), by the group that loads K
  if (group_a == (SC3D2_LOADS >= 2))
  {
    load_global_to_share<8, 4, 1, K_ITERS, SwizzleMode::k128B, HEAD / 16, CTA_K>(
      &KG_lane_ptr, KG_smem_offset_load, stride_seq_k, smem_K, KG_load_idx, kv_len);
    KG_load_idx += CTA_K;
  }
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();
#if SC3D2_DEPHASE
  // blocks of equal work launched together stay in step with their SM neighbour: every other wave of blocks starts
  // SC3D2_DEPHASE ns late, so that the two blocks of an SM sit half a block apart (their tensor and softmax phases
  // interleave instead of coinciding)
  {
    uint32_t nsm;
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsm));
    const uint32_t linear = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    if ((linear / nsm) & 1)
      __nanosleep(SC3D2_DEPHASE);
  }
#endif
  const float q_scale = Q_scale[q_scale_idx];
  const float original_sm_scale = sm_scale;

  // Q.K^T of block W into RS_f32 and scale_w: S converted, scaled and masked as the fp16 kernel's steps do (the
  // main blocks with sm_scale * dequant, the last two with S * dequant, out-of-bound keys of the last masked); then
  // the online softmax into RP, pmax, o_scale, rescale
  float RS_f32[1][TILES_K][8];
  float scale_w;
#define SC3D2_QK(W)                                                                                                     \
  {                                                                                                                     \
    const float dequant_w = q_scale * K_scale[k_scale_idx + (W)];                                                       \
    uint32_t k_mma = K_smem_offset_mma + ((W) & 1) * K_BUF_UNITS;                                                       \
    compute_qk(smem_Q, smem_K, RS, Q_smem_offset_mma, k_mma, magic);                                                    \
    _Pragma("unroll")                                                                                                   \
    for (uint32_t fk = 0; fk < TILES_K; fk++)                                                                           \
    {                                                                                                                   \
      _Pragma("unroll")                                                                                                 \
      for (uint32_t k = 0; k < 8; k++)                                                                                  \
        RS_f32[0][fk][k] = s_float(RS[0][fk][k]);                                                                       \
    }                                                                                                                   \
    if ((W) + 2 < num_iterations)                                                                                       \
    {                                                                                                                   \
      scale_w = original_sm_scale * dequant_w;                                                                          \
    }                                                                                                                   \
    else                                                                                                                \
    {                                                                                                                   \
      _Pragma("unroll")                                                                                                 \
      for (uint32_t fk = 0; fk < TILES_K; fk++)                                                                         \
      {                                                                                                                 \
        _Pragma("unroll")                                                                                               \
        for (uint32_t k = 0; k < 8; k++)                                                                                \
          RS_f32[0][fk][k] = RS_f32[0][fk][k] * dequant_w;                                                              \
      }                                                                                                                 \
      if ((W) + 1 == num_iterations)                                                                                    \
        apply_out_of_bound_mask<1, TILES_K>(K_idx_lane_base + (W) * CTA_K, RS_f32, kv_len);                             \
      scale_w = original_sm_scale;                                                                                      \
    }                                                                                                                   \
  }
#define SC3D2_SOFTMAX(W)                                                                                                \
  {                                                                                                                     \
    SC3D2_SOFTMAX_BODY                                                                                                  \
  }
#define SC3D2_QK_SOFTMAX(W)                                                                                             \
  SC3D2_QK(W)                                                                                                           \
  SC3D2_SOFTMAX(W)

#if SC3D2_ABL & 64
#define SC3D2_SOFTMAX_BODY                                                                                               \
    _Pragma("unroll")                                                                                                    \
    for (uint32_t kk = 0; kk < 2; kk++)                                                                                  \
      _Pragma("unroll")                                                                                                  \
      for (uint32_t i = 0; i < 4; i++)                                                                                   \
        RP[kk][i] = __float_as_uint(RS_f32[0][2 * kk + i / 2][(i % 2) * 2]) ^ __float_as_uint(scale_w);                 \
    pmax[0] = pmax[1] = 1.0f; o_scale[0] = o_scale[1] = 1.0f; rescale = false;
#else
#define SC3D2_SOFTMAX_BODY                                                                                               \
    update_mdo_blk(RS_f32, m, d, pmax, o_scale, rescale, scale_w);                                                       \
    pack_p(RS_f32, RP);                                                                                                  \
    if (SC3D2_PADAT == 0 && (SC3D2_PADG == 0 || (SC3D2_PADG == 1) == group_a)) pad(padx);
#endif

#if SC3D2_ABL & 1
#define SC3D2_V_LOAD_ON(W) ((W) < 2)
#else
#define SC3D2_V_LOAD_ON(W) true
#endif
#if SC3D2_ABL & 2
#define SC3D2_K_LOAD_ON(W) ((W) < 2)
#else
#define SC3D2_K_LOAD_ON(W) true
#endif

  uint32_t RP[2][4];
  float pmax[2];
  float o_scale[2];
  bool rescale;
  uint32_t padx = lane_id;

  // K(W) into buffer W % 2, by the four warps of one group
#if SC3D2_KLOAD
#define SC3D2_LOAD_K(W)                                                                                                 \
  {                                                                                                                     \
    load_k_tile<kStrideK>(KG_lane_ptr, KG_smem_offset_load + ((W) & 1) * K_BUF_UNITS, stride_seq_k, smem_K,             \
                          (W) + 1 < num_iterations, KG_load_idx, kv_len);                                               \
    KG_load_idx += CTA_K;                                                                                               \
    if ((SC3D2_ABL & 128) && ((W) & 1)) KG_lane_ptr -= 2 * CTA_K * (kStrideK ? kStrideK : stride_seq_k);                \
  }
#else
#define SC3D2_LOAD_K(W)                                                                                                  \
  {                                                                                                                      \
    uint32_t k_off = KG_smem_offset_load + ((W) & 1) * K_BUF_UNITS;                                                      \
    load_global_to_share<8, 4, 1, K_ITERS, SwizzleMode::k128B, HEAD / 16, CTA_K>(                                     \
      &KG_lane_ptr, k_off, stride_seq_k, smem_K, KG_load_idx, kv_len);                                                   \
    KG_load_idx += CTA_K;                                                                                                \
  }
#endif
  // V^T(W) and its 128 scales into buffer W % 2, by the four warps of one group
#if SC3D2_VLOCAL
#define SC3D2_LOAD_V(W)                                                                                                 \
  {                                                                                                                     \
    load_v_tile<(SC3D2_CA & 2) != 0>(VG_lane_ptr, VG_smem_offset_load + ((W) & 1) * V_BUF_UNITS, stride_d_v, smem_V);   \
    if (gw == 0)                                                                                                        \
    {                                                                                                                   \
      cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(((b128_t*)(smem_VS + ((W) & 1) * HEAD)) + lane_id, (const b128_t*)VS_lane_ptr);\
    }                                                                                                                   \
    VS_lane_ptr += HEAD;                                                                                                \
  }
#elif SC3D2_VBLK
#define SC3D2_LOAD_V(W)                                                                                                 \
  {                                                                                                                     \
    load_v_blocked(VB_lane_ptr, VB_smem_addr + ((W) & 1) * (V_BUF_UNITS * 16));                                         \
    VB_lane_ptr += CTA_K * HEAD;                                                                                        \
    if (gw == 0)                                                                                                        \
    {                                                                                                                   \
      cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(((b128_t*)(smem_VS + ((W) & 1) * HEAD)) + lane_id, (const b128_t*)VS_lane_ptr);\
    }                                                                                                                   \
    VS_lane_ptr += HEAD;                                                                                                \
  }
#elif SC3D2_VLOAD
#define SC3D2_LOAD_V(W)                                                                                                 \
  {                                                                                                                     \
    load_v_rows(VR_lane_ptr, VR_smem_addr + ((W) & 1) * (V_BUF_UNITS * 16));                                            \
    VR_lane_ptr += CTA_K;                                                                                               \
    if (gw == 0)                                                                                                        \
    {                                                                                                                   \
      cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(((b128_t*)(smem_VS + ((W) & 1) * HEAD)) + lane_id, (const b128_t*)VS_lane_ptr);\
    }                                                                                                                   \
    VS_lane_ptr += HEAD;                                                                                                \
  }
#else
#define SC3D2_LOAD_V(W)                                                                                                  \
  {                                                                                                                      \
    uint32_t v_off = VG_smem_offset_load + ((W) & 1) * V_BUF_UNITS;                                                      \
    load_fp8_V_global_to_share<4, 8, 1, V_ITERS, SwizzleMode::k64B, CTA_K / 16, CTA_K>(                                \
      &VG_lane_ptr, v_off, stride_d_v, smem_V);                                                                          \
    if (gw == 0)                                                                                                         \
    {                                                                                                                    \
      cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(((b128_t*)(smem_VS + ((W) & 1) * HEAD)) + lane_id, (const b128_t*)VS_lane_ptr); \
    }                                                                                                                    \
    VS_lane_ptr += HEAD;                                                                                                 \
    if ((SC3D2_ABL & 128) && ((W) & 1)) { VG_lane_ptr -= 2 * CTA_K; VS_lane_ptr -= 2 * HEAD; }                          \
  }
#endif

#if SC3D2_LOADS == 0
  for (uint32_t w = 0; w <= num_iterations; w++)
  {
    // here K(w) and V(w - 1) have arrived and are visible (the barrier closing the previous trip)
    if (group_a)
    {
      if (w < num_iterations)
      {
        // V(w) into buffer w % 2 (it held V(w - 2), whose readers finished before the last barrier)
        if (SC3D2_V_LOAD_ON(w))
          SC3D2_LOAD_V(w)
        cp_async::commit_group();
        SC3D2_QK_SOFTMAX(w)
#if SC3D2_VWAIT == 1
        // a branch the compiler cannot remove keeps the wait below the softmax
        if (__any_sync(0xffffffff, RP[1][3] == 0x5a5a5a5au && pmax[1] < 0.0f))
          asm volatile("trap;");
#endif
        cp_async::wait_group<0>();
#if !(SC3D2_ABL & 32)
        asm volatile("bar.sync 1, %0;\n" :: "n"(GROUP_THREADS) : "memory");     // group A's V(w) arrived and visible to group A
#endif
        compute_sv(smem_V, smem_VS + (w & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + (w & 1) * V_BUF_UNITS, V_smem_offset_mma1 + (w & 1) * V_BUF_UNITS, magic);
        if (SC3D2_PADAT == 1 && SC3D2_PADG != 2) pad(padx);
      }
    }
    else
    {
      if (w + 1 < num_iterations && SC3D2_K_LOAD_ON(w + 1))
      {
        // K(w + 1) into buffer (w + 1) % 2 (it held K(w - 1), whose readers finished before the last barrier)
        SC3D2_LOAD_K(w + 1)
        cp_async::commit_group();
      }
      if (w >= 1)
      {
        compute_sv(smem_V, smem_VS + ((w - 1) & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + ((w - 1) & 1) * V_BUF_UNITS, V_smem_offset_mma1 + ((w - 1) & 1) * V_BUF_UNITS, magic);
        if (SC3D2_PADAT == 1 && SC3D2_PADG != 1) pad(padx);
      }
      if (w < num_iterations)
      {
        SC3D2_QK_SOFTMAX(w)
      }
      cp_async::wait_group<0>();
    }
#if !(SC3D2_ABL & 32)
    __syncthreads();
#endif
  }
#elif SC3D2_LOADS == 3
  // group A loads both: K(w + 1) at the start of trip w, V^T(w + 1) at its end, once group B has arrived from its P.V
  // of block w - 1 (the last reader of that buffer); group A waits for V^T(w) before its P.V of block w (issued a
  // trip earlier). Group B only computes. V(0) comes with K(0) before the loop.
  if (group_a)
  {
    SC3D2_LOAD_V(0)
  }
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();
  for (uint32_t w = 0; w <= num_iterations; w++)
  {
    // here K(w) has arrived and is visible (the barrier closing the previous trip); V(w) is in flight or arrived
    if (group_a)
    {
      if (w < num_iterations)
      {
        if (w + 1 < num_iterations && SC3D2_K_LOAD_ON(w + 1))
          SC3D2_LOAD_K(w + 1)
        cp_async::commit_group();
        SC3D2_QK_SOFTMAX(w)
        cp_async::wait_group<1>();                            // V(w) arrived (K(w + 1) may still be in flight)
#if !(SC3D2_ABL & 32)
        asm volatile("bar.sync 1, %0;\n" :: "n"(GROUP_THREADS) : "memory");     // ... and visible to group A
#endif
        compute_sv(smem_V, smem_VS + (w & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + (w & 1) * V_BUF_UNITS, V_smem_offset_mma1 + (w & 1) * V_BUF_UNITS, magic);
        if (SC3D2_PADAT == 1 && SC3D2_PADG != 2) pad(padx);
#if !(SC3D2_ABL & 32)
        asm volatile("bar.sync 3, %0;\n" :: "n"(THREADS) : "memory");     // group B is past its P.V of block w - 1
#endif
        if (w + 1 < num_iterations && SC3D2_V_LOAD_ON(w + 1))
          SC3D2_LOAD_V(w + 1)
        cp_async::commit_group();
        cp_async::wait_group<1>();                            // K(w + 1) arrived (V(w + 1) may still be in flight)
      }
    }
    else
    {
      if (w >= 1)
      {
        compute_sv(smem_V, smem_VS + ((w - 1) & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + ((w - 1) & 1) * V_BUF_UNITS, V_smem_offset_mma1 + ((w - 1) & 1) * V_BUF_UNITS, magic);
        if (SC3D2_PADAT == 1 && SC3D2_PADG != 1) pad(padx);
      }
      if (w < num_iterations)
      {
#if !(SC3D2_ABL & 32)
        asm volatile("bar.arrive 3, %0;\n" :: "n"(THREADS) : "memory");
#endif
        SC3D2_QK_SOFTMAX(w)
      }
    }
#if !(SC3D2_ABL & 32)
    __syncthreads();
#endif
  }
#else
  // V^T is loaded a block ahead by group B, right after its P.V of block w - 1 (the last reader of that buffer in
  // group B: a group barrier first), so group A reads V(w) without waiting inside the trip; K(w + 1) by group B
  // (SC3D2_LOADS 1) or by group A at the start of its trip (2). V(0) comes with K(0) before the loop.
  if (!group_a)
  {
    SC3D2_LOAD_V(0)
  }
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();
  for (uint32_t w = 0; w <= num_iterations; w++)
  {
    // here K(w) and V(w) have arrived and are visible (the barrier closing the previous trip)
    if (group_a)
    {
      if (w < num_iterations)
      {
#if SC3D2_LOADS == 2 && SC3D2_KLATE == 0
        if (w + 1 < num_iterations && SC3D2_K_LOAD_ON(w + 1))
          SC3D2_LOAD_K(w + 1)
        cp_async::commit_group();
#endif
#if SC3D2_LOADS == 2 && SC3D2_KLATE == 1
        // K(w + 1) issued between this block's Q.K^T and its softmax (no shared-memory reads there)
        SC3D2_QK(w)
        if (w + 1 < num_iterations && SC3D2_K_LOAD_ON(w + 1))
          SC3D2_LOAD_K(w + 1)
        cp_async::commit_group();
        SC3D2_SOFTMAX(w)
#elif SC3D2_LOADS == 2 && SC3D2_KLATE == 2
        // K(w + 1) issued after this block's softmax, before its P.V
        SC3D2_QK_SOFTMAX(w)
        if (w + 1 < num_iterations && SC3D2_K_LOAD_ON(w + 1))
          SC3D2_LOAD_K(w + 1)
        cp_async::commit_group();
#else
        SC3D2_QK_SOFTMAX(w)
#endif
        compute_sv(smem_V, smem_VS + (w & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + (w & 1) * V_BUF_UNITS, V_smem_offset_mma1 + (w & 1) * V_BUF_UNITS, magic);
        if (SC3D2_PADAT == 1 && SC3D2_PADG != 2) pad(padx);
#if SC3D2_LOADS == 2
        cp_async::wait_group<0>();
#endif
      }
    }
    else
    {
#if SC3D2_LOADS == 1
      if (w + 1 < num_iterations && SC3D2_K_LOAD_ON(w + 1))
        SC3D2_LOAD_K(w + 1)
#endif
      if (w >= 1)
      {
        compute_sv(smem_V, smem_VS + ((w - 1) & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + ((w - 1) & 1) * V_BUF_UNITS, V_smem_offset_mma1 + ((w - 1) & 1) * V_BUF_UNITS, magic);
        if (SC3D2_PADAT == 1 && SC3D2_PADG != 1) pad(padx);
      }
#if SC3D2_VLATE
      if (w < num_iterations)
        SC3D2_QK(w)
#endif
      if (w + 1 < num_iterations)
      {
        // V(w + 1) into buffer (w + 1) % 2: it held V(w - 1), read by group A before the last barrier and by group B
        // just now
#if !(SC3D2_ABL & 32)
        if (w >= 1)
          asm volatile("bar.sync 2, %0;\n" :: "n"(GROUP_THREADS) : "memory");
#endif
        if (SC3D2_V_LOAD_ON(w + 1))
          SC3D2_LOAD_V(w + 1)
      }
      cp_async::commit_group();
      if (w < num_iterations)
      {
#if SC3D2_VLATE
        SC3D2_SOFTMAX(w)
#else
        SC3D2_QK_SOFTMAX(w)
#endif
      }
      cp_async::wait_group<0>();
    }
#if !(SC3D2_ABL & 32)
    __syncthreads();
#endif
  }
#endif
#undef SC3D2_LOAD_K
#undef SC3D2_LOAD_V
#undef SC3D2_QK_SOFTMAX
#undef SC3D2_QK
#undef SC3D2_SOFTMAX
#undef SC3D2_SOFTMAX_BODY
#undef SC3D2_V_LOAD_ON
#undef SC3D2_K_LOAD_ON

  normalize_d<1, TILES_V, ComputeUnit::kCudaCore>(RO, m, d);
#if SC3D2_PAD
  RO[0][0][0] += (padx == 0x9u) ? 1.0f : 0.0f;
#endif

  // the mean of v passes through the softmax weights
  const float *V_mean_lane_ptr = V_mean + (batch_id * num_kv_heads + kv_head) * HEAD + 2 * (lane_id % 4);
#pragma unroll
  for (uint32_t fv = 0; fv < TILES_V; fv++)
  {
    const float2 ma = *(const float2*)(V_mean_lane_ptr + fv * 16);
    const float2 mb = *(const float2*)(V_mean_lane_ptr + fv * 16 + 8);
    RO[0][fv][0] += ma.x;
    RO[0][fv][1] += ma.y;
    RO[0][fv][2] += ma.x;
    RO[0][fv][3] += ma.y;
    RO[0][fv][4] += mb.x;
    RO[0][fv][5] += mb.y;
    RO[0][fv][6] += mb.x;
    RO[0][fv][7] += mb.y;
  }

#if SC3D2_ABL & 32
  __syncthreads();
#endif

  // save the result to shared memory, then to global memory
  uint32_t smem_O_row_base = warp_id * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fv = 0; fv < TILES_V; fv++)
  {
    uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base, fv * 2);
    uint32_t RO_16[4];
#pragma unroll
    for (uint32_t k = 0; k < 4; k++)
    {
      if constexpr (std::is_same<DTypeOut, half>::value)
        ((half2*)RO_16)[k] = __float22half2_rn(((float2*)RO[0][fv])[k]);
      else
        ((nv_bfloat162*)RO_16)[k] = __float22bfloat162_rn(((float2*)RO[0][fv])[k]);
    }
    ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_16[0];
    ((uint32_t*)(smem_O.base + offset_O + 8 * (HEAD / 8)))[lane_id % 4] = RO_16[1];
    ((uint32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = RO_16[2];
    ((uint32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * (HEAD / 8)))[lane_id % 4] = RO_16[3];
  }

  __syncwarp();

  DTypeOut *O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o + (bx * CTA_Q + WARP_Q * warp_id + lane_id / 8) * stride_seq_o + lane_id % 8 * 8;
  uint32_t offset_O = smem_O.get_permuted_offset(warp_id * WARP_Q + lane_id / 8, lane_id % 8);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8;

#pragma unroll
  for (uint32_t i = 0; i < CTA_Q / (NUM_WARPS * 4); i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < HEAD / 64; j++)
    {
      if (O_load_idx_lane_base < qo_len)
      {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += 64;
      offset_O = smem_O.advance_offset_by_column<8>(offset_O);
    }
    offset_O = smem_O.advance_offset_by_row<4>(offset_O - (HEAD / 64) * 8);
    O_lane_ptr += 4 * stride_seq_o - (HEAD / 64) * 64;
    O_load_idx_lane_base += 4;
  }
}

} // namespace sc3d_i8s

// the arguments and checks of qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d (qk_int_sv_i8_cuda_sm80.cu)
torch::Tensor qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d2(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    torch::Tensor value_scale,
                    torch::Tensor value_mean,
                    int64_t tensor_layout,
                    int64_t is_causal,
                    int64_t qk_quant_gran,
                    double sm_scale,
                    int64_t return_lse)
{
  using namespace sc3d_i8s;

  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);
  CHECK_CUDA(value_scale);
  CHECK_CUDA(value_mean);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_CONTIGUOUS(value_scale);
  CHECK_CONTIGUOUS(value_mean);

  CHECK_DTYPE(query, torch::kInt8);
  CHECK_DTYPE(key, torch::kInt8);
  CHECK_DTYPE(value, torch::kInt8);
  CHECK_DTYPE(query_scale, torch::kFloat32);
  CHECK_DTYPE(key_scale, torch::kFloat32);
  CHECK_DTYPE(value_scale, torch::kFloat32);
  CHECK_DTYPE(value_mean, torch::kFloat32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, SC3D2_VBLK ? 5 : 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_scale, 4);
  CHECK_DIMS(value_mean, 3);

  const int head_dim = query.size(3);
  const int batch_size = query.size(0);
  TORCH_CHECK(head_dim == (int)HEAD && is_causal == 0 && qk_quant_gran == static_cast<int64_t>(QuantGranularity::kPerWarp) && return_lse == 0,
              "qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d2: implemented for head 128, not causal, per_warp, no LSE only");

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_o, stride_h_q, stride_h_k, stride_h_o;
  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_o = output.stride(1);
    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_o = output.stride(2);
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_o = output.stride(2);
    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_o = output.stride(1);
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }
  TORCH_CHECK(qo_len > 0 && kv_len > 0, "empty sequence");
  if (num_qo_heads % num_kv_heads != 0)
  {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());
  }
  const int num_kv_groups = num_qo_heads / num_kv_heads;
  const int num_blocks_k = div_ceil(kv_len, (int)CTA_K);

#if SC3D2_VBLK
  CHECK_SHAPE(value, batch_size, num_kv_heads, num_blocks_k, head_dim, (int)CTA_K);
#else
  CHECK_SHAPE(value, batch_size, num_kv_heads, head_dim, num_blocks_k * (int)CTA_K);
#endif
  CHECK_SHAPE(value_scale, batch_size, num_kv_heads, num_blocks_k, head_dim);
  CHECK_SHAPE(value_mean, batch_size, num_kv_heads, head_dim);
  CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, 128) * (int)Q_GROUPS_PER_128);
  CHECK_SHAPE(key_scale, batch_size, num_kv_heads, num_blocks_k);

  dim3 grid(div_ceil(qo_len, (int)CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, NUM_WARPS);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto *out_ptr) {
    using DTypeOut = std::remove_pointer_t<decltype(out_ptr)>;
    auto kernel_func = (SC3D2_KLOAD && stride_seq_k == 1536) ? qk_int_sv_i8_attn_kernel<DTypeOut, 1536>
                                                             : qk_int_sv_i8_attn_kernel<DTypeOut, 0>;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    kernel_func<<<grid, block, SMEM_BYTES, stream>>>(
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), value.data_ptr<int8_t>(), out_ptr,
      reinterpret_cast<float*>(query_scale.data_ptr()), reinterpret_cast<float*>(key_scale.data_ptr()),
      reinterpret_cast<float*>(value_scale.data_ptr()), reinterpret_cast<float*>(value_mean.data_ptr()),
      qo_len, kv_len, num_kv_groups,
      query.stride(0), stride_seq_q, stride_h_q,
      key.stride(0), stride_seq_k, stride_h_k,
      value.stride(0), value.stride(1), value.stride(2),
      output.stride(0), stride_seq_o, stride_h_o,
      (float)sm_scale);
  };
  if (output.scalar_type() == at::ScalarType::BFloat16)
    launch(reinterpret_cast<nv_bfloat16*>(output.data_ptr()));
  else if (output.scalar_type() == at::ScalarType::Half)
    launch(reinterpret_cast<half*>(output.data_ptr()));
  else
    TORCH_CHECK(false, "output must be bfloat16 or float16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return torch::empty({0});
}
