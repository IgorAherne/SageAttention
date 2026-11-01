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

// Spellcast3D round K: the flows' kernel (qk_int_sv_f16_attn_kernel's sc3d = true instantiation: head 128, CTA_Q 128,
// WARP_Q 16, CTA_K 64, per-warp q and k scales, not causal, no LSE, __launch_bounds__(256, 2)) with its P.V step on the
// int8 tensor cores. q, k, their scales, the Q.K^T and the online softmax are that kernel's; then
//  - P (u8, per row and 64-key block): P is raised relative to the BLOCK's maximum m_blk, so that it lies in (0, 1],
//    and P_u8 = round(P * 255); the block's int32 product P_u8 . v_s8 enters the fp32 accumulators times
//    pmax / 255 per row (pmax = exp2(m_blk - m), m the running maximum) and v's scale per channel; the accumulators
//    are rescaled by exp2(m_prev - m) as in the fp16 kernel, in the P.V step (RO = fma(product, factor, RO * o_scale),
//    the same values), and not at all in a block where no row of the warp has a new maximum (o_scale = exp2(0) = 1);
//  - v (s8, per 64-key block and channel, the channel's mean over the tokens subtracted): quant_per_block_int8_v_cuda
//    below, V^T in the fp8 kernels' layout (tokens permuted within each 16 for the k32 fragments), [B, H, 128, Npad];
//    the mean is added back after the normalisation (the softmax weights sum to 1);
//  - the denominator from the fp32 P on the CUDA cores (the kCudaCore way), pmax times the block's row sum.
// The P.V step one 16-channel output tile at a time (two m16n16k32 steps, 8 int32 accumulators). The blocks are run
// as a ping-pong of the two warp groups (round J's sv_kernel_pp.cuh): see the main loop.

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

// timing study only (0 in the package): 1 no P.V MMAs, 2 no conversion of their products, 3 neither
#ifndef I8_STUDY
#define I8_STUDY 0
#endif

namespace sc3d_i8 {

constexpr uint32_t CTA_Q = 128;
constexpr uint32_t CTA_K = 64;
constexpr uint32_t WARP_Q = 16;
constexpr uint32_t HEAD = 128;
constexpr uint32_t NUM_WARPS = CTA_Q / WARP_Q;
constexpr uint32_t THREADS = 32 * NUM_WARPS;
constexpr uint32_t TILES_K = CTA_K / 16;         // 16-key tiles of S per warp
constexpr uint32_t TILES_QK_INNER = HEAD / 32;   // k32 steps of Q.K^T
constexpr uint32_t TILES_V = HEAD / 16;          // 16-channel tiles of O

// Q int8 [128][128], two K int8 [64][128], two V^T int8 [128][64], two blocks' v scales fp32 [128]; O bf16 [128][128]
// at the end
constexpr uint32_t K_BUF_UNITS = CTA_K * HEAD / 16;    // one tile in 16-byte units: 64 rows of K, 128 rows of V^T
constexpr uint32_t V_BUF_UNITS = HEAD * CTA_K / 16;
constexpr uint32_t SMEM_V_OFFSET = (CTA_Q + 2 * CTA_K) * HEAD;
constexpr uint32_t SMEM_VS_OFFSET = SMEM_V_OFFSET + 2 * HEAD * CTA_K;
constexpr size_t SMEM_BYTES = (SMEM_VS_OFFSET + 2 * HEAD * sizeof(float)) > (CTA_Q * HEAD * sizeof(nv_bfloat16))
                              ? (SMEM_VS_OFFSET + 2 * HEAD * sizeof(float)) : (CTA_Q * HEAD * sizeof(nv_bfloat16));

// round(x * 255) of four x in [0, 1] as four bytes, lowest first: x * 255 + 1.5 * 2^23 (one rounding, to nearest even)
// holds the integer in its low byte
__device__ __forceinline__ uint32_t pack_u8x4(float a, float b, float c, float e)
{
  const uint32_t ua = __float_as_uint(fmaf(a, 255.0f, 12582912.0f));
  const uint32_t ub = __float_as_uint(fmaf(b, 255.0f, 12582912.0f));
  const uint32_t uc = __float_as_uint(fmaf(c, 255.0f, 12582912.0f));
  const uint32_t ue = __float_as_uint(fmaf(e, 255.0f, 12582912.0f));
  return __byte_perm(__byte_perm(ua, ub, 0x0040), __byte_perm(uc, ue, 0x0040), 0x5410);
}

// update_mdo (attn_utils.cuh, float accumulators) for the int8 P.V: m and d as there, the accumulators' factor
// o_scale = exp2(m_prev - m) per row left to the P.V step (o_scale) with whether any row of the warp has a new maximum
// (rescale); P raised relative to the block's maximum m_blk instead of m; pmax = exp2(m_blk - m) per row; d gains
// pmax times this thread's part of the block's row sum of the fp32 P (the lanes' parts are summed by
// normalize_d<kCudaCore>)
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
    o_scale[k] = math::ptx_exp2(m_prev - m[0][k]);
    d[0][k] *= o_scale[k];

    pmax[k] = math::ptx_exp2(m_temp - m[0][k]);
    float negative_m = -m_temp;
    float rowsum = 0.0f;
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
      RS[0][fk][k * 2 + 0] = math::ptx_exp2(fmaf(RS[0][fk][k * 2 + 0], sm_scale, negative_m));
      RS[0][fk][k * 2 + 1] = math::ptx_exp2(fmaf(RS[0][fk][k * 2 + 1], sm_scale, negative_m));
      RS[0][fk][k * 2 + 4] = math::ptx_exp2(fmaf(RS[0][fk][k * 2 + 4], sm_scale, negative_m));
      RS[0][fk][k * 2 + 5] = math::ptx_exp2(fmaf(RS[0][fk][k * 2 + 5], sm_scale, negative_m));
      rowsum += RS[0][fk][k * 2 + 0] + RS[0][fk][k * 2 + 1] + RS[0][fk][k * 2 + 4] + RS[0][fk][k * 2 + 5];
    }
    d[0][k] = fmaf(pmax[k], rowsum, d[0][k]);
  }
  rescale = __any_sync(0xffffffff, new_max);
}

// P as the u8 A fragments of the block's two k32 steps (RS_32_to_8's order: with V^T permuted within each 16 tokens,
// S's accumulator elements 0, 1, 4, 5 of a 16-key tile are the k32 fragment's four bytes of row g, 2, 3, 6, 7 of g + 8)
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

// P.V of one block: per 16-channel tile, the int32 product over the 64 keys (V^T fragments by ldmatrix, two
// m16n16k32 u8 x s8), then RO = product * (pmax / 255) * v_scale + RO (* o_scale with kRescale) per row and channel
template <bool kRescale>
__device__ __forceinline__ void compute_sv_t(const smem_t<SwizzleMode::k64B, 4> &smem_V, const float *smem_VS, uint32_t RP[2][4],
                                             float RO[][TILES_V][8], const float pmax[2], const float o_scale[2],
                                             const uint32_t offset_V0, const uint32_t offset_V1)
{
  const float f0 = pmax[0] * (1.0f / 255.0f);
  const float f1 = pmax[1] * (1.0f / 255.0f);
  const float *vs = smem_VS + 2 * (get_lane_id() % 4);
#pragma unroll
  for (uint32_t fv = 0; fv < TILES_V; fv++)
  {
    int32_t acc[8];
#if I8_STUDY & 1
#pragma unroll
    for (uint32_t i = 0; i < 8; i++)
      acc[i] = (int32_t)(RP[i / 4][i % 4] ^ fv);
#else
    uint32_t RV[4];
    // rows: channels fv * 16 + (lane % 8) + 8 * (lane / 16), 16-byte units: keys 0-15 / 16-31 (+ 32 for the second step)
    smem_V.ldmatrix_m8n8x4(offset_V0 + fv * 16 * 4, RV);
    mma::mma_sync_m16n16k32_row_col_u8s8s32<mma::MMAMode::kInit>(acc, RP[0], RV);
    smem_V.ldmatrix_m8n8x4(offset_V1 + fv * 16 * 4, RV);
    mma::mma_sync_m16n16k32_row_col_u8s8s32<mma::MMAMode::kInplaceUpdate>(acc, RP[1], RV);
#endif
#if I8_STUDY & 2
#pragma unroll
    for (uint32_t i = 0; i < 8; i++)
      RO[0][fv][i] += __int_as_float(acc[i]);
#else
    // accumulator elements 0, 1 / 4, 5: row g, channels fv * 16 + 2 (lane % 4) + 0, 1 / + 8; 2, 3 / 6, 7: row g + 8
    const float2 sa = *(const float2*)(vs + fv * 16);
    const float2 sb = *(const float2*)(vs + fv * 16 + 8);
    RO[0][fv][0] = fmaf(__int2float_rn(acc[0]), f0 * sa.x, kRescale ? RO[0][fv][0] * o_scale[0] : RO[0][fv][0]);
    RO[0][fv][1] = fmaf(__int2float_rn(acc[1]), f0 * sa.y, kRescale ? RO[0][fv][1] * o_scale[0] : RO[0][fv][1]);
    RO[0][fv][2] = fmaf(__int2float_rn(acc[2]), f1 * sa.x, kRescale ? RO[0][fv][2] * o_scale[1] : RO[0][fv][2]);
    RO[0][fv][3] = fmaf(__int2float_rn(acc[3]), f1 * sa.y, kRescale ? RO[0][fv][3] * o_scale[1] : RO[0][fv][3]);
    RO[0][fv][4] = fmaf(__int2float_rn(acc[4]), f0 * sb.x, kRescale ? RO[0][fv][4] * o_scale[0] : RO[0][fv][4]);
    RO[0][fv][5] = fmaf(__int2float_rn(acc[5]), f0 * sb.y, kRescale ? RO[0][fv][5] * o_scale[0] : RO[0][fv][5]);
    RO[0][fv][6] = fmaf(__int2float_rn(acc[6]), f1 * sb.x, kRescale ? RO[0][fv][6] * o_scale[1] : RO[0][fv][6]);
    RO[0][fv][7] = fmaf(__int2float_rn(acc[7]), f1 * sb.y, kRescale ? RO[0][fv][7] * o_scale[1] : RO[0][fv][7]);
#endif
  }
}

__device__ __forceinline__ void compute_sv(const smem_t<SwizzleMode::k64B, 4> &smem_V, const float *smem_VS, uint32_t RP[2][4],
                                           float RO[][TILES_V][8], const float pmax[2], const float o_scale[2], const bool rescale,
                                           const uint32_t offset_V0, const uint32_t offset_V1)
{
  if (rescale)
    compute_sv_t<true>(smem_V, smem_VS, RP, RO, pmax, o_scale, offset_V0, offset_V1);
  else
    compute_sv_t<false>(smem_V, smem_VS, RP, RO, pmax, o_scale, offset_V0, offset_V1);
}

template <typename DTypeOut>
__global__ void __launch_bounds__(THREADS, 2) qk_int_sv_i8_attn_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V, DTypeOut *__restrict__ O,
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

  // per warp q scales (16 rows), per 64-key block k scales
  const uint32_t q_scale_idx = batch_id * num_qo_heads * (gridDim.x * NUM_WARPS) + head_id * (gridDim.x * NUM_WARPS) + bx * NUM_WARPS + warp_id;
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

  // the ping-pong: warps 0-3 (group A) run each block w in order: the V(w) load, Q.K^T, softmax, P.V; warps 4-7
  // (group B, one on each scheduler beside an A warp) run half a block behind: the K(w + 1) load, P.V of block w - 1
  // (P, pmax and o_scale carried in registers), then Q.K^T and softmax of block w - so while one warp of a scheduler
  // is in its softmax the other has tensor work. Per warp the operations and their order are those of one block after
  // the other. Two K and two V buffers, one block barrier a trip and a group barrier for A's V.
  const bool group_a = warp_id < 4;
  const uint32_t gw = warp_id % 4;

  // Q (all warps), rows of 128 bytes: 8 lanes a row, 4 rows a warp; K (group B, 16 rows a warp); V^T (group A, rows of
  // 64 bytes: 4 lanes a row, 32 rows a warp; padded: no predicate) and, by warp 0, the block's 128 scales
  int8_t *Q_lane_base_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8) * stride_seq_q + (lane_id % 8) * 16;
  int8_t *KG_lane_ptr = K + batch_id * stride_bz_k + kv_head * stride_h_k + (CTA_K / 4 * gw + lane_id / 8) * stride_seq_k + (lane_id % 8) * 16;
  int8_t *VG_lane_ptr = V + batch_id * stride_bz_v + kv_head * stride_h_v + (HEAD / 4 * gw + lane_id / 4) * stride_d_v + (lane_id % 4) * 16;
  const float *VS_lane_ptr = V_scale + (batch_id * num_kv_heads + kv_head) * num_iterations * HEAD + lane_id * 4;
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * 4 * (CTA_Q / (NUM_WARPS * 4)) + lane_id / 8, lane_id % 8);
  uint32_t KG_smem_offset_load = smem_K.get_permuted_offset(gw * 4 * (CTA_K / 16) + lane_id / 8, lane_id % 8);
  uint32_t VG_smem_offset_load = smem_V.get_permuted_offset(gw * 8 * (HEAD / 32) + lane_id / 4, lane_id % 4);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(warp_id * WARP_Q + lane_id % 16, lane_id / 16);
  const uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  const uint32_t V_smem_offset_mma0 = smem_V.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  const uint32_t V_smem_offset_mma1 = smem_V.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2 + 2);

  const uint32_t K_idx_lane_base = 2 * (lane_id % 4);
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8;
  uint32_t KG_load_idx = CTA_K / 4 * gw + lane_id / 8;

  load_global_to_share<8, 4, 1, CTA_Q / (NUM_WARPS * 4), SwizzleMode::k128B, HEAD / 16, CTA_Q>(
    &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  if (!group_a)
  {
    load_global_to_share<8, 4, 1, CTA_K / 16, SwizzleMode::k128B, HEAD / 16, CTA_K>(
      &KG_lane_ptr, KG_smem_offset_load, stride_seq_k, smem_K, KG_load_idx, kv_len);
    KG_load_idx += CTA_K;
  }
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();
  const float q_scale = Q_scale[q_scale_idx];
  const float original_sm_scale = sm_scale;

  // Q.K^T and the softmax of block W into RP, pmax, o_scale, rescale: S converted, scaled and masked as the fp16
  // kernel's steps do (the main blocks with sm_scale * dequant, the last two with S * dequant, out-of-bound keys of
  // the last masked)
#define SC3D_I8_QK_SOFTMAX(W)                                                                                            \
  {                                                                                                                      \
    const float dequant_w = q_scale * K_scale[k_scale_idx + (W)];                                                        \
    uint32_t k_mma = K_smem_offset_mma + ((W) & 1) * K_BUF_UNITS;                                                        \
    compute_int_qk<NUM_WARPS, 1, 1, TILES_K, TILES_QK_INNER, SwizzleMode::k128B, HEAD / 16, DataType::kInt8>(           \
      smem_Q, smem_K, RS, Q_smem_offset_mma, k_mma);                                                                     \
    float RS_f32[1][TILES_K][8];                                                                                         \
    _Pragma("unroll")                                                                                                    \
    for (uint32_t fk = 0; fk < TILES_K; fk++)                                                                            \
    {                                                                                                                    \
      _Pragma("unroll")                                                                                                  \
      for (uint32_t k = 0; k < 8; k++)                                                                                   \
        RS_f32[0][fk][k] = __int2float_rz(RS[0][fk][k]);                                                                 \
    }                                                                                                                    \
    float scale_w;                                                                                                       \
    if ((W) + 2 < num_iterations)                                                                                        \
    {                                                                                                                    \
      scale_w = original_sm_scale * dequant_w;                                                                           \
    }                                                                                                                    \
    else                                                                                                                 \
    {                                                                                                                    \
      _Pragma("unroll")                                                                                                  \
      for (uint32_t fk = 0; fk < TILES_K; fk++)                                                                          \
      {                                                                                                                  \
        _Pragma("unroll")                                                                                                \
        for (uint32_t k = 0; k < 8; k++)                                                                                 \
          RS_f32[0][fk][k] = RS_f32[0][fk][k] * dequant_w;                                                               \
      }                                                                                                                  \
      if ((W) + 1 == num_iterations)                                                                                     \
        apply_out_of_bound_mask<1, TILES_K>(K_idx_lane_base + (W) * CTA_K, RS_f32, kv_len);                              \
      scale_w = original_sm_scale;                                                                                       \
    }                                                                                                                    \
    update_mdo_blk(RS_f32, m, d, pmax, o_scale, rescale, scale_w);                                                       \
    pack_p(RS_f32, RP);                                                                                                  \
  }

  uint32_t RP[2][4];
  float pmax[2];
  float o_scale[2];
  bool rescale;
  for (uint32_t w = 0; w <= num_iterations; w++)
  {
    // here K(w) and V(w - 1) have arrived and are visible (the barrier closing the previous trip)
    if (group_a)
    {
      if (w < num_iterations)
      {
        // V(w) into buffer w % 2 (it held V(w - 2), whose readers finished before the last barrier)
        uint32_t v_off = VG_smem_offset_load + (w & 1) * V_BUF_UNITS;
        load_fp8_V_global_to_share<4, 8, 1, HEAD / 32, SwizzleMode::k64B, CTA_K / 16, CTA_K>(
          &VG_lane_ptr, v_off, stride_d_v, smem_V);
        if (gw == 0)
        {
          cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(((b128_t*)(smem_VS + (w & 1) * HEAD)) + lane_id, (const b128_t*)VS_lane_ptr);
        }
        VS_lane_ptr += HEAD;
        cp_async::commit_group();
        SC3D_I8_QK_SOFTMAX(w)
        cp_async::wait_group<0>();
        asm volatile("bar.sync 1, 128;\n" ::: "memory");     // group A's V(w) arrived and visible to group A
        compute_sv(smem_V, smem_VS + (w & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + (w & 1) * V_BUF_UNITS, V_smem_offset_mma1 + (w & 1) * V_BUF_UNITS);
      }
    }
    else
    {
      if (w + 1 < num_iterations)
      {
        // K(w + 1) into buffer (w + 1) % 2 (it held K(w - 1), whose readers finished before the last barrier)
        uint32_t k_off = KG_smem_offset_load + ((w + 1) & 1) * K_BUF_UNITS;
        load_global_to_share<8, 4, 1, CTA_K / 16, SwizzleMode::k128B, HEAD / 16, CTA_K>(
          &KG_lane_ptr, k_off, stride_seq_k, smem_K, KG_load_idx, kv_len);
        cp_async::commit_group();
        KG_load_idx += CTA_K;
      }
      if (w >= 1)
      {
        compute_sv(smem_V, smem_VS + ((w - 1) & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + ((w - 1) & 1) * V_BUF_UNITS, V_smem_offset_mma1 + ((w - 1) & 1) * V_BUF_UNITS);
      }
      if (w < num_iterations)
      {
        SC3D_I8_QK_SOFTMAX(w)
      }
      cp_async::wait_group<0>();
    }
    __syncthreads();
  }
#undef SC3D_I8_QK_SOFTMAX

  normalize_d<1, TILES_V, ComputeUnit::kCudaCore>(RO, m, d);

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

  // save the result to shared memory, then to global memory (as qk_int_sv_f16_attn_kernel)
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

// ---- v: mean per channel over the tokens, then per 64-token block and channel: scale = max |v - mean| / 127 (1 for
// an all-zero block), v_s8 = round(v - mean) / scale (to nearest even) in -127..127, written as V^T [B, H, 128, Npad]
// with the tokens of each 16 in the order the k32 fragments read them (the fp8 kernels' permutation)

constexpr uint32_t VQ_THREADS = HEAD;   // one thread a channel

template <typename T>
__device__ __forceinline__ float to_float(const T x)
{
  if constexpr (std::is_same<T, half>::value)
    return __half2float(x);
  else
    return __bfloat162float(x);
}

// one block of 64 tokens x 128 channels into shared memory, zeros past the last token
template <typename T>
__device__ __forceinline__ void load_v_tile(const T *v, T tile[CTA_K][HEAD], const uint32_t blk, const uint32_t num_tokens,
                                            const uint32_t stride_seq)
{
  const uint32_t t = threadIdx.x;
#pragma unroll
  for (uint32_t pass = 0; pass < CTA_K * HEAD / (8 * VQ_THREADS); pass++)
  {
    const uint32_t row = pass * (VQ_THREADS / 16) + t / 16;
    const uint32_t tok = blk * CTA_K + row;
    uint4 x = make_uint4(0, 0, 0, 0);
    if (tok < num_tokens)
      x = *(const uint4*)(v + (size_t)tok * stride_seq + (t % 16) * 8);
    *(uint4*)(&tile[row][(t % 16) * 8]) = x;
  }
}

template <typename T>
__global__ void __launch_bounds__(VQ_THREADS) v_block_sum_kernel(const T *__restrict__ v, float *__restrict__ part, const uint32_t num_tokens,
                                                                 const uint32_t stride_bz, const uint32_t stride_seq, const uint32_t stride_h)
{
  __shared__ __align__(16) T tile[CTA_K][HEAD];
  const uint32_t blk = blockIdx.x, h = blockIdx.y, b = blockIdx.z, c = threadIdx.x;
  load_v_tile(v + (size_t)b * stride_bz + (size_t)h * stride_h, tile, blk, num_tokens, stride_seq);
  __syncthreads();
  float s = 0.0f;
#pragma unroll 16
  for (uint32_t r = 0; r < CTA_K; r++)
    s += to_float(tile[r][c]);
  part[(((size_t)b * gridDim.y + h) * gridDim.x + blk) * HEAD + c] = s;
}

__global__ void __launch_bounds__(VQ_THREADS) v_mean_kernel(const float *__restrict__ part, float *__restrict__ mean, const uint32_t num_blocks,
                                                            const uint32_t num_tokens)
{
  const uint32_t bh = blockIdx.x, c = threadIdx.x;
  const float *p = part + (size_t)bh * num_blocks * HEAD + c;
  float s = 0.0f;
  for (uint32_t j = 0; j < num_blocks; j++)
    s += p[(size_t)j * HEAD];
  mean[bh * HEAD + c] = __fdiv_rn(s, (float)num_tokens);
}

template <typename T>
__global__ void __launch_bounds__(VQ_THREADS) v_quant_kernel(const T *__restrict__ v, const float *__restrict__ mean, int8_t *__restrict__ out,
                                                             float *__restrict__ scale, const uint32_t num_tokens,
                                                             const uint32_t stride_bz, const uint32_t stride_seq, const uint32_t stride_h,
                                                             const uint32_t stride_bz_o, const uint32_t stride_h_o, const uint32_t stride_d_o,
                                                             const bool smooth)
{
  __shared__ __align__(16) T tile[CTA_K][HEAD];
  const uint32_t blk = blockIdx.x, h = blockIdx.y, b = blockIdx.z, c = threadIdx.x;
  load_v_tile(v + (size_t)b * stride_bz + (size_t)h * stride_h, tile, blk, num_tokens, stride_seq);
  __syncthreads();
  const uint32_t bh = b * gridDim.y + h;
  const float mu = smooth ? mean[bh * HEAD + c] : 0.0f;
  const uint32_t valid = min(CTA_K, num_tokens - blk * CTA_K);
  float amax = 0.0f;
  for (uint32_t r = 0; r < valid; r++)
    amax = fmaxf(amax, fabsf(to_float(tile[r][c]) - mu));
  const float sv = (amax > 0.0f) ? __fdiv_rn(amax, 127.0f) : 1.0f;
  scale[((size_t)bh * gridDim.x + blk) * HEAD + c] = sv;
  // stored position p of a 16-token group holds token (p / 4) * 2 + (p % 2) + 8 * ((p / 2) % 2): 0 1 8 9 2 3 10 11 ...
  uint32_t packed[CTA_K / 4];
#pragma unroll
  for (uint32_t w = 0; w < CTA_K / 4; w++)
  {
    uint32_t word = 0;
#pragma unroll
    for (uint32_t i = 0; i < 4; i++)
    {
      const uint32_t p = w * 4 + i;
      const uint32_t r = (p / 16) * 16 + ((p % 16) / 4) * 2 + (p % 2) + 8 * ((p / 2) % 2);
      int32_t q = 0;
      if (r < valid)
      {
        const float x = __fdiv_rn(to_float(tile[r][c]) - mu, sv);
        q = max(-127, min(127, __float2int_rn(x)));
      }
      word |= ((uint32_t)(q & 0xFF)) << (8 * i);
    }
    packed[w] = word;
  }
  int8_t *o = out + (size_t)b * stride_bz_o + (size_t)h * stride_h_o + (size_t)c * stride_d_o + blk * CTA_K;
#pragma unroll
  for (uint32_t w = 0; w < CTA_K / 16; w++)
    *(uint4*)(o + w * 16) = make_uint4(packed[4 * w], packed[4 * w + 1], packed[4 * w + 2], packed[4 * w + 3]);
}

} // namespace sc3d_i8

// tensor_layout 0 for [B, N, H, D], 1 for [B, H, N, D] (query, key, output); value: V^T [B, H_kv, 128, Npad] int8 with
// value_scale [B, H_kv, Npad / 64, 128] and value_mean [B, H_kv, 128] fp32, as quant_per_block_int8_v_cuda makes them.
// Implemented: head 128, not causal, per_warp q / k scales (BLKQ 128, WARPQ 16, BLKK 64), no LSE, bf16 or fp16 out.
torch::Tensor qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d(torch::Tensor query,
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
  using namespace sc3d_i8;

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
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);
  CHECK_DIMS(value_scale, 4);
  CHECK_DIMS(value_mean, 3);

  const int head_dim = query.size(3);
  const int batch_size = query.size(0);
  TORCH_CHECK(head_dim == (int)HEAD && is_causal == 0 && qk_quant_gran == static_cast<int64_t>(QuantGranularity::kPerWarp) && return_lse == 0,
              "qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d: implemented for head 128, not causal, per_warp, no LSE only");

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

  CHECK_SHAPE(value, batch_size, num_kv_heads, head_dim, num_blocks_k * (int)CTA_K);
  CHECK_SHAPE(value_scale, batch_size, num_kv_heads, num_blocks_k, head_dim);
  CHECK_SHAPE(value_mean, batch_size, num_kv_heads, head_dim);
  CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, (int)CTA_Q) * (int)NUM_WARPS);
  CHECK_SHAPE(key_scale, batch_size, num_kv_heads, num_blocks_k);

  dim3 grid(div_ceil(qo_len, (int)CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, NUM_WARPS);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto *out_ptr) {
    using DTypeOut = std::remove_pointer_t<decltype(out_ptr)>;
    auto kernel_func = qk_int_sv_i8_attn_kernel<DTypeOut>;
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

// value [B, N, H, 128] (tensor_layout 0) or [B, H, N, 128] (1), fp16 or bf16, last dimension contiguous ->
// value_t [B, H, 128, Npad] int8, value_scale [B, H, Npad / 64, 128], value_mean [B, H, 128] fp32 (zeros, and nothing
// subtracted, when smooth_v is 0). Deterministic: fixed summation orders, no atomics.
void quant_per_block_int8_v_cuda(torch::Tensor value,
                    torch::Tensor value_t,
                    torch::Tensor value_scale,
                    torch::Tensor value_mean,
                    int64_t tensor_layout,
                    int64_t smooth_v)
{
  using namespace sc3d_i8;

  CHECK_CUDA(value);
  CHECK_CUDA(value_t);
  CHECK_CUDA(value_scale);
  CHECK_CUDA(value_mean);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(value_t);
  CHECK_CONTIGUOUS(value_scale);
  CHECK_CONTIGUOUS(value_mean);
  CHECK_DTYPE(value_t, torch::kInt8);
  CHECK_DTYPE(value_scale, torch::kFloat32);
  CHECK_DTYPE(value_mean, torch::kFloat32);
  CHECK_DIMS(value, 4);

  const int batch_size = value.size(0);
  const int head_dim = value.size(3);
  int num_tokens, num_heads, stride_seq, stride_h;
  if (tensor_layout == 0)
  {
    num_tokens = value.size(1);
    num_heads = value.size(2);
    stride_seq = value.stride(1);
    stride_h = value.stride(2);
  }
  else if (tensor_layout == 1)
  {
    num_tokens = value.size(2);
    num_heads = value.size(1);
    stride_seq = value.stride(2);
    stride_h = value.stride(1);
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }
  TORCH_CHECK(head_dim == (int)HEAD, "quant_per_block_int8_v_cuda: head 128 only");
  TORCH_CHECK(num_tokens > 0, "empty sequence");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(value.data_ptr()) % 16 == 0 && stride_seq % 8 == 0 && stride_h % 8 == 0 && value.stride(0) % 8 == 0,
              "value must be 16-byte aligned with strides of multiples of 8 elements");
  const int num_blocks = div_ceil(num_tokens, (int)CTA_K);
  CHECK_SHAPE(value_t, batch_size, num_heads, head_dim, num_blocks * (int)CTA_K);
  CHECK_SHAPE(value_scale, batch_size, num_heads, num_blocks, head_dim);
  CHECK_SHAPE(value_mean, batch_size, num_heads, head_dim);

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(num_blocks, num_heads, batch_size);
  auto run = [&](auto *v_ptr) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(v_ptr)>>;
    float *mean = reinterpret_cast<float*>(value_mean.data_ptr());
    if (smooth_v)
    {
      torch::Tensor part = torch::empty({batch_size, num_heads, num_blocks, head_dim}, value_scale.options());
      v_block_sum_kernel<T><<<grid, VQ_THREADS, 0, stream>>>(v_ptr, reinterpret_cast<float*>(part.data_ptr()), num_tokens,
                                                              value.stride(0), stride_seq, stride_h);
      v_mean_kernel<<<batch_size * num_heads, VQ_THREADS, 0, stream>>>(reinterpret_cast<float*>(part.data_ptr()), mean, num_blocks, num_tokens);
    }
    else
    {
      value_mean.zero_();
    }
    v_quant_kernel<T><<<grid, VQ_THREADS, 0, stream>>>(v_ptr, mean, value_t.data_ptr<int8_t>(), reinterpret_cast<float*>(value_scale.data_ptr()),
                                                        num_tokens, value.stride(0), stride_seq, stride_h,
                                                        value_t.stride(0), value_t.stride(1), value_t.stride(2), smooth_v != 0);
  };
  if (value.scalar_type() == at::ScalarType::BFloat16)
    run(reinterpret_cast<const nv_bfloat16*>(value.data_ptr()));
  else if (value.scalar_type() == at::ScalarType::Half)
    run(reinterpret_cast<const half*>(value.data_ptr()));
  else
    TORCH_CHECK(false, "value must be bfloat16 or float16");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
