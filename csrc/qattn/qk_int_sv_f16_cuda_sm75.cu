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

// Spellcast3D round L: the flows' SageAttention kernel for Turing (sm_75: RTX 20xx, GTX 16xx). The op
// qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d of qk_int_sv_f16_cuda_sm80.cu for its one configuration - head 128,
// CTA_Q 128, WARP_Q 16, CTA_K 64, q per 16 tokens and k per 64 tokens as int8 (per_warp), not causal, no LSE, P.V in
// fp16 summed over each 64-key block on the tensor cores and added to fp32 accumulators ("fp16+fp32") - with the
// same arguments, built from Turing's instructions:
//  - Q.K^T: every mma m16n8k32 s8 of sm_80 as four mma m8n8k16 on the same fragments - exact integer sums, the same S;
//  - P.V and the row sums of P: every m16n8k16 f16 as two m16n8k8 - the fp16 block sums are rounded after every 8 keys
//    instead of every 16, the only difference in the arithmetic;
//  - no cp.async: the next K and V tiles are loaded into registers while the current ones are in use and stored to
//    shared memory after it (register-staged double buffering); rows past the sequence's end are zeros;
//  - everything else is sm_80's code in sm_80's order (attn_utils.cuh): the S conversions and scalings (the main
//    blocks with sm_scale * dequant, the last two with S * dequant), the out-of-bound mask of the last block, the
//    online softmax (update_mdo), P rounded to fp16, the normalisation, the output through shared memory.

#include "../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include "../mma.cuh"
#include "../permuted_smem.cuh"
#include "../math.cuh"

#include "attn_utils.cuh"

// output column tiles of the P.V step in flight at a time (1, 2, 4 or 8: the same values, a register / latency trade)
#ifndef SM75_SV_GROUP
#define SM75_SV_GROUP 2
#endif
// the blocks' schedule: 0 one block after the other (two block barriers a block), 1 the ping-pong of the two warp
// groups (two K and two V buffers: 64 KB of shared memory); the same operations in the same order per warp
#ifndef SM75_PIPE
#define SM75_PIPE 1
#endif
// S to float by the magic add (an integer and a float add, exact for |S| < 2^22) instead of I2F (Turing's
// quarter-rate conversion unit, shared with exp2 and the fp16 rounding of P): the same values
#ifndef SM75_I2F_MAGIC
#define SM75_I2F_MAGIC 0
#endif
// timing study only (0 in the package; wrong answers), bits: 1 no P.V MMAs, 2 no Q.K^T MMAs, 4 no softmax, 8 no V
// ldmatrix, 16 no K ldmatrix (what is left out costs nothing: its inputs are kept alive by empty asm statements)
#ifndef SM75_STUDY
#define SM75_STUDY 0
#endif

namespace sc3d_sm75 {

constexpr uint32_t CTA_Q = 128;
constexpr uint32_t CTA_K = 64;
constexpr uint32_t WARP_Q = 16;
constexpr uint32_t HEAD = 128;
constexpr uint32_t NUM_WARPS = CTA_Q / WARP_Q;
constexpr uint32_t THREADS = 32 * NUM_WARPS;
constexpr uint32_t TILES_K = CTA_K / 16;         // 16-key tiles of S per warp
constexpr uint32_t TILES_QK_INNER = HEAD / 32;   // k32 steps of Q.K^T
constexpr uint32_t TILES_V = HEAD / 16;          // 16-channel tiles of O
constexpr uint32_t QK_STRIDE = HEAD / 16;        // 16-byte units in a row of Q or K (int8)
constexpr uint32_t V_STRIDE = HEAD / 8;          // ... of V (fp16) and of O

// Q int8 [128][128], K int8 [64][128] and V fp16 [64][128] once (SM75_PIPE 0: 40 KB) or twice (1: 64 KB, Turing's
// most): one block an SM; O [128][128] 16-bit at the end, over Q and K
constexpr uint32_t BUFFERS = (SM75_PIPE == 1) ? 2 : 1;
constexpr uint32_t K_BUF_UNITS = CTA_K * HEAD / 16;                  // one tile in 16-byte units
constexpr uint32_t V_BUF_UNITS = CTA_K * HEAD * sizeof(half) / 16;
constexpr uint32_t SMEM_K_OFFSET = CTA_Q * HEAD;
constexpr uint32_t SMEM_V_OFFSET = SMEM_K_OFFSET + BUFFERS * CTA_K * HEAD;
constexpr size_t SMEM_BYTES = (SMEM_V_OFFSET + BUFFERS * CTA_K * HEAD * sizeof(half)) > (CTA_Q * HEAD * sizeof(half))
                              ? (SMEM_V_OFFSET + BUFFERS * CTA_K * HEAD * sizeof(half)) : (CTA_Q * HEAD * sizeof(half));

// A tile of ROWS rows of UNITS 16-byte units, spread over WARPS warps as sm_80's load_global_to_share spreads it over
// the block: a warp takes ROWS / WARPS rows, a lane the rows ROWS / WARPS * warp + lane / 8 + 4 i and the units
// lane % 8 + 8 j
template <uint32_t ROWS, uint32_t UNITS, uint32_t WARPS>
struct tile_t
{
  static constexpr uint32_t ITERS_COL = ROWS / (WARPS * 4);
  static constexpr uint32_t ITERS_ROW = UNITS / 8;
  static constexpr uint32_t REGS = ITERS_COL * ITERS_ROW;
  static_assert(ROWS % (WARPS * 4) == 0 && UNITS % 8 == 0, "4 rows of 8 lanes a warp and step");
};

// the lane's part of a tile from global memory into registers (lane_ptr: the lane's first unit, row_idx: its first
// row's index); with kPred rows at or past max_len are zeros
template <uint32_t ROWS, uint32_t UNITS, uint32_t WARPS, bool kPred>
__device__ __forceinline__ void load_tile(uint4 *R, const int8_t *lane_ptr, const uint32_t stride_bytes, const uint32_t row_idx, const uint32_t max_len)
{
  using T = tile_t<ROWS, UNITS, WARPS>;
#pragma unroll
  for (uint32_t i = 0; i < T::ITERS_COL; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < T::ITERS_ROW; j++)
    {
      const uint4 *p = reinterpret_cast<const uint4*>(lane_ptr + i * 4 * stride_bytes + j * 8 * 16);
      if (!kPred || row_idx + 4 * i < max_len)
        R[i * T::ITERS_ROW + j] = __ldg(p);
      else
        R[i * T::ITERS_ROW + j] = make_uint4(0, 0, 0, 0);
    }
  }
}

// ... and from registers into the (swizzled) shared memory tile, at the offsets of sm_80's walk (offset: the lane's
// first unit)
template <uint32_t ROWS, uint32_t UNITS, uint32_t WARPS, SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void store_tile(const uint4 *R, const smem_t<swizzle_mode, stride> &smem, uint32_t offset)
{
  using T = tile_t<ROWS, UNITS, WARPS>;
#pragma unroll
  for (uint32_t i = 0; i < T::ITERS_COL; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < T::ITERS_ROW; j++)
    {
      smem.base[offset + j * 8] = R[i * T::ITERS_ROW + j];
    }
    offset = smem.advance_offset_by_row<4>(offset);
  }
}

// compute_int_qk (attn_utils.cuh, Q and K from shared memory) with the m8n8k16 product
__device__ __forceinline__ void compute_int_qk(const smem_t<SwizzleMode::k128B, QK_STRIDE> &smem_Q, const smem_t<SwizzleMode::k128B, QK_STRIDE> &smem_K,
                                               int32_t RS[][TILES_K][8], uint32_t offset_Q, uint32_t offset_K)
{
  uint32_t RQ[4];
  uint32_t RK[4];
#pragma unroll
  for (uint32_t iter = 0; iter < TILES_QK_INNER; iter++)
  {
    smem_Q.ldmatrix_m8n8x4(offset_Q, RQ);
    // ! using permutation invariance (as sm_80)
    offset_Q = smem_Q.advance_offset_by_column<2>(offset_Q, iter);
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
#if SM75_STUDY & 16
#pragma unroll
      for (uint32_t i = 0; i < 4; i++)
        RK[i] = RQ[(i + fk) % 4];
#else
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
#endif
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);
#if SM75_STUDY & 2
      asm volatile("" :: "r"(RQ[0]), "r"(RQ[1]), "r"(RQ[2]), "r"(RQ[3]), "r"(RK[0]), "r"(RK[1]), "r"(RK[2]), "r"(RK[3]));
      if (iter == 0)
      {
#pragma unroll
        for (uint32_t i = 0; i < 8; i++)
          RS[0][fk][i] = (int32_t)RK[i % 4];
      }
#else
      if (iter == 0)
        mma::mma_sync_m16n16k32_row_col_x8s8s32_sm75<mma::MMAMode::kInit>(RS[0][fk], RQ, RK);
      else
        mma::mma_sync_m16n16k32_row_col_x8s8s32_sm75<mma::MMAMode::kInplaceUpdate>(RS[0][fk], RQ, RK);
#endif
    }
    offset_K = smem_K.advance_offset_by_column<2>(offset_K - (TILES_K * 16 * QK_STRIDE), iter);
  }
}

// S as a float: exact either way (|S| <= 128 * 127^2 < 2^22)
__device__ __forceinline__ float s_to_float(const int32_t x)
{
#if SM75_I2F_MAGIC
  return __int_as_float(x + 0x4B400000) - 12582912.0f;
#else
  return __int2float_rz(x);
#endif
}

// accumulate_d<kTensorCore> (attn_utils.cuh) with the m16n8k8 row sum
__device__ __forceinline__ void accumulate_d(uint32_t RS_f16[][TILES_K][4], float d[][2])
{
#pragma unroll
  for (uint32_t fk = 0; fk < TILES_K; fk++)
  {
    mma::rowsum_f16f16f32_sm75(d[0], RS_f16[0][fk]);
  }
}

// update_mdo (attn_utils.cuh, fp32 accumulators) with the accumulators' rescale left to the P.V step: m, d and P as
// there, o_scale = exp2(m_prev - m) per row returned. sm_80's compiler fuses that rescale with the P.V step's add into
// one FMA per accumulator, RO = fma(RO, o_scale, the block's fp16 sum) (its SASS: no FADD, 64 FFMA a block beside the
// softmax's 32), so the P.V step does it in that form - in every block and for either schedule
__device__ __forceinline__ void update_md(float RS[][TILES_K][8], float m[][2], float d[][2], float o_scale[2], const float &sm_scale)
{
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
    // exchange element with the 4 threads in the row
    m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
    m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));

    m[0][k] = max(m[0][k], m_temp);
    o_scale[k] = math::ptx_exp2(m_prev - m[0][k]);
    d[0][k] *= o_scale[k];

    float negative_m = -m[0][k];
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
      RS[0][fk][k * 2 + 0] = math::ptx_exp2(fmaf(RS[0][fk][k * 2 + 0], sm_scale, negative_m));
      RS[0][fk][k * 2 + 1] = math::ptx_exp2(fmaf(RS[0][fk][k * 2 + 1], sm_scale, negative_m));
      RS[0][fk][k * 2 + 4] = math::ptx_exp2(fmaf(RS[0][fk][k * 2 + 4], sm_scale, negative_m));
      RS[0][fk][k * 2 + 5] = math::ptx_exp2(fmaf(RS[0][fk][k * 2 + 5], sm_scale, negative_m));
    }
  }
}

// The softmax of key block `iter` from its S as sm_80's steps do it: the main blocks with S unscaled and sm_scale *
// dequant, the last two with S * dequant and sm_scale, the out-of-bound keys of the last masked; then the online
// softmax (m, d, o_scale), P rounded to fp16 (RS_f16) and d += P's row sums on the tensor cores
__device__ __forceinline__ void softmax(int32_t RS[][TILES_K][8], uint32_t RS_f16[][TILES_K][4], float o_scale[2], float m[][2], float d[][2],
                                        const uint32_t iter, const uint32_t num_iterations, const float dequant_scale,
                                        const float original_sm_scale, const uint32_t K_idx_lane_base, const uint32_t kv_len)
{
  float RS_f32[1][TILES_K][8];
#if SM75_STUDY & 4
#pragma unroll
  for (uint32_t fk = 0; fk < TILES_K; fk++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 4; k++)
      RS_f16[0][fk][k] = (uint32_t)RS[0][fk][2 * k] ^ (uint32_t)RS[0][fk][2 * k + 1];
  }
  o_scale[0] = o_scale[1] = 1.0f;
  (void)RS_f32; (void)m; (void)d; (void)iter; (void)num_iterations; (void)dequant_scale;
  (void)original_sm_scale; (void)K_idx_lane_base; (void)kv_len;
  return;
#endif
  if (iter + 2 < num_iterations)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        RS_f32[0][fk][k] = s_to_float(RS[0][fk][k]);
      }
    }
    update_md(RS_f32, m, d, o_scale, original_sm_scale * dequant_scale);
  }
  else
  {
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        RS_f32[0][fk][k] = s_to_float(RS[0][fk][k]) * dequant_scale;
      }
    }
    if (iter + 1 == num_iterations)
    {
      apply_out_of_bound_mask<1, TILES_K>(K_idx_lane_base + iter * CTA_K, RS_f32, kv_len);
    }
    update_md(RS_f32, m, d, o_scale, original_sm_scale);
  }
  RS_32_to_16<1, TILES_K>(RS_f32, RS_f16);
  accumulate_d(RS_f16, d);
}

// The P.V step of the tuned sm_80 kernel (sc3d_sv_fv: SM75_SV_GROUP output column tiles at a time, each summing its
// four 16-key steps in fp16) with the m16n8k8 product; then RO = fma(RO, o_scale, the sum) (update_md). The
// shared-memory offsets are sc3d_sv_fv's closed form of the k128B walk
__device__ __forceinline__ void compute_sv(const smem_t<SwizzleMode::k128B, V_STRIDE> &smem_V, uint32_t RS_f16[][TILES_K][4], float RO[][TILES_V][8],
                                           const float o_scale[2], const uint32_t offset_V)
{
  constexpr uint32_t group = SM75_SV_GROUP;
  static_assert(TILES_V % group == 0, "SM75_SV_GROUP divides the 8 output column tiles");
#pragma unroll
  for (uint32_t fv0 = 0; fv0 < TILES_V; fv0 += group)
  {
    uint32_t buf[group][4];
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
#pragma unroll
      for (uint32_t j = 0; j < group; j++)
      {
        const uint32_t fv = fv0 + j;
        uint32_t RV[4];
#if SM75_STUDY & 8
#pragma unroll
        for (uint32_t i = 0; i < 4; i++)
          RV[i] = RS_f16[0][(fk + j) % TILES_K][i];
#else
        smem_V.ldmatrix_m8n8x4_trans(((offset_V + (fv / 4) * 8) ^ (2 * (fv % 4))) + fk * 16 * V_STRIDE, RV);
#endif
#if SM75_STUDY & 1
        asm volatile("" :: "r"(RV[0]), "r"(RV[1]), "r"(RV[2]), "r"(RV[3]));
        if (fk == 0)
        {
#pragma unroll
          for (uint32_t i = 0; i < 4; i++)
            buf[j][i] = RS_f16[0][fk][i];
        }
#else
        if (fk == 0)
          mma::mma_sync_m16n16k16_row_col_f16f16f16_sm75<mma::MMAMode::kInit>(buf[j], RS_f16[0][fk], RV);
        else
          mma::mma_sync_m16n16k16_row_col_f16f16f16_sm75<mma::MMAMode::kInplaceUpdate>(buf[j], RS_f16[0][fk], RV);
#endif
      }
    }
    // accumulator elements 0, 1, 4, 5: row g; 2, 3, 6, 7: row g + 8
#pragma unroll
    for (uint32_t j = 0; j < group; j++)
    {
      const uint32_t fv = fv0 + j;
      RO[0][fv][0] = fmaf(RO[0][fv][0], o_scale[0], __half2float(((half2*)buf[j])[0].x));
      RO[0][fv][1] = fmaf(RO[0][fv][1], o_scale[0], __half2float(((half2*)buf[j])[0].y));
      RO[0][fv][2] = fmaf(RO[0][fv][2], o_scale[1], __half2float(((half2*)buf[j])[1].x));
      RO[0][fv][3] = fmaf(RO[0][fv][3], o_scale[1], __half2float(((half2*)buf[j])[1].y));
      RO[0][fv][4] = fmaf(RO[0][fv][4], o_scale[0], __half2float(((half2*)buf[j])[2].x));
      RO[0][fv][5] = fmaf(RO[0][fv][5], o_scale[0], __half2float(((half2*)buf[j])[2].y));
      RO[0][fv][6] = fmaf(RO[0][fv][6], o_scale[1], __half2float(((half2*)buf[j])[3].x));
      RO[0][fv][7] = fmaf(RO[0][fv][7], o_scale[1], __half2float(((half2*)buf[j])[3].y));
    }
  }
}

template <typename DTypeOut>
__global__ void __launch_bounds__(THREADS, 1) qk_int_sv_f16_attn_kernel(const int8_t *__restrict__ Q, const int8_t *__restrict__ K, const half *__restrict__ V, DTypeOut *__restrict__ O,
                      const float *__restrict__ Q_scale, const float *__restrict__ K_scale,
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q,
                      const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_bz_v, const uint32_t stride_seq_v, const uint32_t stride_h_v,
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

  // base 2 instead of base e, as sm_80
  sm_scale *= math::log2e;

  float RO[1][TILES_V][8];
  float m[1][2];
  float d[1][2];

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

  // per warp q scales (16 rows), per 64-key block k scales
  const uint32_t q_scale_idx = batch_id * num_qo_heads * (gridDim.x * NUM_WARPS) + head_id * (gridDim.x * NUM_WARPS) + bx * NUM_WARPS + warp_id;
  const float *K_scale_ptr = K_scale + batch_id * num_kv_heads * num_iterations + kv_head * num_iterations;

  smem_t<SwizzleMode::k128B, QK_STRIDE> smem_Q(smem);
  smem_t<SwizzleMode::k128B, QK_STRIDE> smem_K(smem + SMEM_K_OFFSET);
  smem_t<SwizzleMode::k128B, V_STRIDE> smem_V(smem + SMEM_V_OFFSET);
  smem_t<SwizzleMode::k128B, V_STRIDE> smem_O(smem);

  // Q: rows of 128 bytes, 8 lanes a row, 4 rows a warp and step, all warps
  const int8_t *Q_lane_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8) * stride_seq_q + (lane_id % 8) * 16;
  const uint32_t Q_load_idx = bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8;
  const uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(CTA_Q / NUM_WARPS * warp_id + lane_id / 8, lane_id % 8);
  const uint32_t K_stride_bytes = stride_seq_k;
  const uint32_t V_stride_bytes = stride_seq_v * sizeof(half);

  const uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(warp_id * WARP_Q + lane_id % 16, lane_id / 16);
  const uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  const uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(lane_id % 16, lane_id / 16);

  const uint32_t K_idx_lane_base = 2 * (lane_id % 4);

  {
    uint4 RQ_load[tile_t<CTA_Q, QK_STRIDE, NUM_WARPS>::REGS];
    load_tile<CTA_Q, QK_STRIDE, NUM_WARPS, true>(RQ_load, Q_lane_ptr, stride_seq_q, Q_load_idx, qo_len);
    store_tile<CTA_Q, QK_STRIDE, NUM_WARPS>(RQ_load, smem_Q, Q_smem_offset_load);
  }
  const float q_scale = Q_scale[q_scale_idx];
  const float original_sm_scale = sm_scale;

#if SM75_PIPE == 1
  // The ping-pong (round K's int8 P.V kernel, round J's sv_kernel_pp.cuh): warps 0-3 (group A) run each block w in
  // order - the V(w) load, Q.K^T, softmax, P.V; warps 4-7 (group B, one on each scheduler beside an A warp) run half a
  // block behind - the K(w + 1) load, P.V of block w - 1 (its P carried in registers), then Q.K^T and softmax of block
  // w - so while one warp of a scheduler is in its softmax the other has tensor work. Per warp the operations and
  // their order are those of one block after the other. Two K and two V buffers, one block barrier a trip and a group
  // barrier for A's V.
  constexpr uint32_t GROUP_WARPS = NUM_WARPS / 2;
  const bool group_a = warp_id < GROUP_WARPS;
  const uint32_t gw = warp_id % GROUP_WARPS;
  const int8_t *K_lane_ptr = K + batch_id * stride_bz_k + kv_head * stride_h_k + (CTA_K / GROUP_WARPS * gw + lane_id / 8) * stride_seq_k + (lane_id % 8) * 16;
  const int8_t *V_lane_ptr = reinterpret_cast<const int8_t*>(V + batch_id * stride_bz_v + kv_head * stride_h_v + (CTA_K / GROUP_WARPS * gw + lane_id / 8) * stride_seq_v + (lane_id % 8) * 8);
  uint32_t KV_load_idx = CTA_K / GROUP_WARPS * gw + lane_id / 8;
  const uint32_t K_smem_offset_load = smem_K.get_permuted_offset(CTA_K / GROUP_WARPS * gw + lane_id / 8, lane_id % 8);
  const uint32_t V_smem_offset_load = smem_V.get_permuted_offset(CTA_K / GROUP_WARPS * gw + lane_id / 8, lane_id % 8);
  using KTile = tile_t<CTA_K, QK_STRIDE, GROUP_WARPS>;
  using VTile = tile_t<CTA_K, V_STRIDE, GROUP_WARPS>;

  if (!group_a)
  {
    uint4 RK_load[KTile::REGS];
    load_tile<CTA_K, QK_STRIDE, GROUP_WARPS, true>(RK_load, K_lane_ptr, K_stride_bytes, KV_load_idx, kv_len);
    store_tile<CTA_K, QK_STRIDE, GROUP_WARPS>(RK_load, smem_K, K_smem_offset_load);
    K_lane_ptr += CTA_K * K_stride_bytes;
  }
  __syncthreads();

  uint32_t RS_f16[1][TILES_K][4];       // group B: P of the block before and its rows' o_scale, from one trip to the next
  float o_scale[2];
  for (uint32_t w = 0; w <= num_iterations; w++)
  {
    // here K(w) is in buffer w % 2 and V(w - 1) in buffer (w - 1) % 2, both visible (the barrier closing the trip)
    if (group_a)
    {
      if (w < num_iterations)
      {
        // V(w) into buffer w % 2 (it held V(w - 2), whose readers finished before the last barrier)
        uint4 RV_load[VTile::REGS];
        if (w + 1 == num_iterations)
          load_tile<CTA_K, V_STRIDE, GROUP_WARPS, true>(RV_load, V_lane_ptr, V_stride_bytes, KV_load_idx + w * CTA_K, kv_len);
        else
          load_tile<CTA_K, V_STRIDE, GROUP_WARPS, false>(RV_load, V_lane_ptr, V_stride_bytes, KV_load_idx + w * CTA_K, kv_len);
        V_lane_ptr += CTA_K * V_stride_bytes;
        int32_t RS[1][TILES_K][8];
        compute_int_qk(smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma + (w & 1) * K_BUF_UNITS);
        softmax(RS, RS_f16, o_scale, m, d, w, num_iterations, q_scale * K_scale_ptr[w], original_sm_scale, K_idx_lane_base, kv_len);
        store_tile<CTA_K, V_STRIDE, GROUP_WARPS>(RV_load, smem_V, V_smem_offset_load + (w & 1) * V_BUF_UNITS);
        asm volatile("bar.sync 1, %0;\n" :: "n"(GROUP_WARPS * 32) : "memory");     // group A's V(w) visible to group A
        compute_sv(smem_V, RS_f16, RO, o_scale, V_smem_offset_mma + (w & 1) * V_BUF_UNITS);
      }
    }
    else
    {
      // K(w + 1) into buffer (w + 1) % 2 (it held K(w - 1), whose readers finished before the last barrier)
      const bool has_next = w + 1 < num_iterations;
      uint4 RK_load[KTile::REGS];
      if (has_next)
      {
        if (w + 2 == num_iterations)
          load_tile<CTA_K, QK_STRIDE, GROUP_WARPS, true>(RK_load, K_lane_ptr, K_stride_bytes, KV_load_idx + (w + 1) * CTA_K, kv_len);
        else
          load_tile<CTA_K, QK_STRIDE, GROUP_WARPS, false>(RK_load, K_lane_ptr, K_stride_bytes, KV_load_idx + (w + 1) * CTA_K, kv_len);
        K_lane_ptr += CTA_K * K_stride_bytes;
      }
      if (w >= 1)
      {
        compute_sv(smem_V, RS_f16, RO, o_scale, V_smem_offset_mma + ((w - 1) & 1) * V_BUF_UNITS);
      }
      if (has_next)
      {
        store_tile<CTA_K, QK_STRIDE, GROUP_WARPS>(RK_load, smem_K, K_smem_offset_load + ((w + 1) & 1) * K_BUF_UNITS);
      }
      if (w < num_iterations)
      {
        int32_t RS[1][TILES_K][8];
        compute_int_qk(smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma + (w & 1) * K_BUF_UNITS);
        softmax(RS, RS_f16, o_scale, m, d, w, num_iterations, q_scale * K_scale_ptr[w], original_sm_scale, K_idx_lane_base, kv_len);
      }
    }
    __syncthreads();
  }
#else
  // one block after the other: K(iter) in shared memory and V(iter) in registers at the top; V stored after Q.K^T,
  // K(iter + 1) loaded during the softmax and P.V and stored after them, V(iter + 1) loaded during the next Q.K^T
  const int8_t *K_lane_ptr = K + batch_id * stride_bz_k + kv_head * stride_h_k + (CTA_K / NUM_WARPS * warp_id + lane_id / 8) * stride_seq_k + (lane_id % 8) * 16;
  const int8_t *V_lane_ptr = reinterpret_cast<const int8_t*>(V + batch_id * stride_bz_v + kv_head * stride_h_v + (CTA_K / NUM_WARPS * warp_id + lane_id / 8) * stride_seq_v + (lane_id % 8) * 8);
  uint32_t KV_load_idx = CTA_K / NUM_WARPS * warp_id + lane_id / 8;
  const uint32_t K_smem_offset_load = smem_K.get_permuted_offset(CTA_K / NUM_WARPS * warp_id + lane_id / 8, lane_id % 8);
  const uint32_t V_smem_offset_load = smem_V.get_permuted_offset(CTA_K / NUM_WARPS * warp_id + lane_id / 8, lane_id % 8);
  using KTile = tile_t<CTA_K, QK_STRIDE, NUM_WARPS>;
  using VTile = tile_t<CTA_K, V_STRIDE, NUM_WARPS>;
  uint4 RK_next[KTile::REGS];
  uint4 RV_next[VTile::REGS];
  load_tile<CTA_K, QK_STRIDE, NUM_WARPS, true>(RK_next, K_lane_ptr, K_stride_bytes, KV_load_idx, kv_len);
  load_tile<CTA_K, V_STRIDE, NUM_WARPS, true>(RV_next, V_lane_ptr, V_stride_bytes, KV_load_idx, kv_len);
  store_tile<CTA_K, QK_STRIDE, NUM_WARPS>(RK_next, smem_K, K_smem_offset_load);
  K_lane_ptr += CTA_K * K_stride_bytes;
  V_lane_ptr += CTA_K * V_stride_bytes;
  KV_load_idx += CTA_K;
  __syncthreads();

  for (uint32_t iter = 0; iter < num_iterations; iter++)
  {
    int32_t RS[1][TILES_K][8];
    compute_int_qk(smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);

    store_tile<CTA_K, V_STRIDE, NUM_WARPS>(RV_next, smem_V, V_smem_offset_load);
    const bool has_next = iter + 1 < num_iterations;
    const bool next_is_last = iter + 2 == num_iterations;
    if (has_next)
    {
      if (next_is_last)
        load_tile<CTA_K, QK_STRIDE, NUM_WARPS, true>(RK_next, K_lane_ptr, K_stride_bytes, KV_load_idx, kv_len);
      else
        load_tile<CTA_K, QK_STRIDE, NUM_WARPS, false>(RK_next, K_lane_ptr, K_stride_bytes, KV_load_idx, kv_len);
    }

    uint32_t RS_f16[1][TILES_K][4];
    float o_scale[2];
    softmax(RS, RS_f16, o_scale, m, d, iter, num_iterations, q_scale * K_scale_ptr[iter], original_sm_scale, K_idx_lane_base, kv_len);

    // V(iter) visible to every warp, and every warp done with K(iter)
    __syncthreads();

    compute_sv(smem_V, RS_f16, RO, o_scale, V_smem_offset_mma);

    if (has_next)
    {
      store_tile<CTA_K, QK_STRIDE, NUM_WARPS>(RK_next, smem_K, K_smem_offset_load);
      if (next_is_last)
        load_tile<CTA_K, V_STRIDE, NUM_WARPS, true>(RV_next, V_lane_ptr, V_stride_bytes, KV_load_idx, kv_len);
      else
        load_tile<CTA_K, V_STRIDE, NUM_WARPS, false>(RV_next, V_lane_ptr, V_stride_bytes, KV_load_idx, kv_len);
      K_lane_ptr += CTA_K * K_stride_bytes;
      V_lane_ptr += CTA_K * V_stride_bytes;
      KV_load_idx += CTA_K;
    }

    // K(iter + 1) visible to every warp, and every warp done with V(iter)
    __syncthreads();
  }
#endif

  normalize_d<1, TILES_V, ComputeUnit::kTensorCore>(RO, m, d);

  // save the result to shared memory, then to global memory (as sm_80)
  const uint32_t smem_O_row_base = warp_id * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fv = 0; fv < TILES_V; fv++)
  {
    const uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base, fv * 2);
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
    ((uint32_t*)(smem_O.base + offset_O + 8 * V_STRIDE))[lane_id % 4] = RO_16[1];
    // ! permuted, make sure you know what you are doing (as sm_80)
    ((uint32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = RO_16[2];
    ((uint32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * V_STRIDE))[lane_id % 4] = RO_16[3];
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

} // namespace sc3d_sm75

// query, key int8 and value fp16 [B, N, H, 128] (tensor_layout 0) or [B, H, N, 128] (1), output fp16 or bf16 in the
// same layout, query_scale [B, H, ceil(N / 128) * 8] and key_scale [B, H_kv, ceil(N_kv / 64)] fp32 (per_warp_int8 with
// BLKQ 128, WARPQ 16, BLKK 64): the arguments of the sm_80 op qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d, for head
// 128, not causal, per_warp, no LSE (anything else raises). Returns an empty tensor (no LSE).
torch::Tensor qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d_sm75(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    int64_t tensor_layout,
                    int64_t is_causal,
                    int64_t qk_quant_gran,
                    double sm_scale,
                    int64_t return_lse)
{
  using namespace sc3d_sm75;

  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, torch::kInt8);
  CHECK_DTYPE(key, torch::kInt8);
  CHECK_DTYPE(value, torch::kHalf);
  CHECK_DTYPE(query_scale, torch::kFloat32);
  CHECK_DTYPE(key_scale, torch::kFloat32);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.size(3);
  const int batch_size = query.size(0);
  TORCH_CHECK(head_dim == (int)HEAD && is_causal == 0 && qk_quant_gran == static_cast<int64_t>(QuantGranularity::kPerWarp) && return_lse == 0,
              "qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d (sm_75): implemented for head 128, not causal, per_warp, no LSE only");

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o, stride_h_q, stride_h_k, stride_h_v, stride_h_o;
  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);
    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_v = value.stride(1);
    stride_seq_o = output.stride(1);
    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_h_o = output.stride(2);
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);
    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_v = value.stride(2);
    stride_seq_o = output.stride(2);
    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
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
  // 16-byte loads of V: its rows, heads and batches start at multiples of 8 elements
  TORCH_CHECK(reinterpret_cast<uintptr_t>(value.data_ptr()) % 16 == 0 && stride_seq_v % 8 == 0 && stride_h_v % 8 == 0 && value.stride(0) % 8 == 0,
              "value must be 16-byte aligned with strides of multiples of 8 elements");
  const int num_kv_groups = num_qo_heads / num_kv_heads;

  CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, (int)CTA_Q) * (int)NUM_WARPS);
  CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, (int)CTA_K));

  dim3 grid(div_ceil(qo_len, (int)CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, NUM_WARPS);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto *out_ptr) {
    using DTypeOut = std::remove_pointer_t<decltype(out_ptr)>;
    auto kernel_func = qk_int_sv_f16_attn_kernel<DTypeOut>;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    kernel_func<<<grid, block, SMEM_BYTES, stream>>>(
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), reinterpret_cast<half*>(value.data_ptr()), out_ptr,
      reinterpret_cast<float*>(query_scale.data_ptr()), reinterpret_cast<float*>(key_scale.data_ptr()),
      qo_len, kv_len, num_kv_groups,
      query.stride(0), stride_seq_q, stride_h_q,
      key.stride(0), stride_seq_k, stride_h_k,
      value.stride(0), stride_seq_v, stride_h_v,
      output.stride(0), stride_seq_o, stride_h_o,
      (float)sm_scale);
  };
  if (output.scalar_type() == at::ScalarType::Half)
    launch(reinterpret_cast<half*>(output.data_ptr()));
  else if (output.scalar_type() == at::ScalarType::BFloat16)
    launch(reinterpret_cast<nv_bfloat16*>(output.data_ptr()));
  else
    TORCH_CHECK(false, "output must be float16 or bfloat16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return torch::empty({0});
}
