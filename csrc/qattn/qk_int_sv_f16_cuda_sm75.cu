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
//  - the rest in sm_80's order: the S conversions and scalings (the main blocks with sm_scale * dequant, the last two
//    with S * dequant), the out-of-bound mask of the last block, the online softmax in fp32 (update_mdo), P rounded
//    to fp16, the accumulators' rescale fused with the P.V step's add (RO = fma(RO, o_scale, the block's sum): what
//    sm_80's compiler makes of it, in its SASS), the normalisation, the output through shared memory.
// The blocks run as a ping-pong of the two warp groups (round K's int8 P.V kernel), see the main loop.
// RTX 2060 Super (sm_75, its power limit 135 W), 21,504 tokens x 12 heads: 70.0 ms a call (40.6 TFLOPS), torch SDPA's
// "efficient" fp16 kernel 177.5 ms. The study (round L's tools, branch history): one block after the other with two
// block barriers 73.8 ms; S by the magic add instead of I2F 72.9 ms (the card is held at its power limit: fewer joules,
// not fewer stalls, make it faster); 1, 2, 4 or 8 output tiles in flight alike.

#include "../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

#include "../mma.cuh"
#include "../permuted_smem.cuh"
#include "../math.cuh"

#include "attn_utils.cuh"
#include "attn_utils_sm75.cuh"

namespace sc3d_sm75 {

constexpr uint32_t V_STRIDE = HEAD / 8;          // 16-byte units in a row of V (fp16)
constexpr uint32_t SV_GROUP = 2;                 // output column tiles of the P.V step in flight at a time

// Q int8 [128][128], two K int8 [64][128], two V fp16 [64][128]: 64 KB, Turing's most (one block an SM); O [128][128]
// 16-bit at the end, over Q and K
constexpr uint32_t K_BUF_UNITS = CTA_K * HEAD / 16;                  // one tile in 16-byte units
constexpr uint32_t V_BUF_UNITS = CTA_K * HEAD * sizeof(half) / 16;
constexpr uint32_t SMEM_K_OFFSET = CTA_Q * HEAD;
constexpr uint32_t SMEM_V_OFFSET = SMEM_K_OFFSET + 2 * CTA_K * HEAD;
constexpr size_t SMEM_BYTES = SMEM_V_OFFSET + 2 * CTA_K * HEAD * sizeof(half);
static_assert(SMEM_BYTES >= CTA_Q * HEAD * sizeof(half) && SMEM_BYTES <= 65536, "O fits; Turing's 64 KB a block");

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
// there, o_scale = exp2(m_prev - m) per row returned
__device__ __forceinline__ void update_md(float RS[][TILES_K][8], float m[][2], float d[][2], float o_scale[2], const float sm_scale)
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

// Q.K^T and the softmax of key block `iter` (K from shared memory at offset_K): P in fp16 (RS_f16), each row's o_scale,
// d += P's row sums on the tensor cores
__device__ __forceinline__ void qk_softmax(const smem_t<SwizzleMode::k128B, QK_STRIDE> &smem_Q, const smem_t<SwizzleMode::k128B, QK_STRIDE> &smem_K,
                                           const uint32_t offset_Q, const uint32_t offset_K,
                                           uint32_t RS_f16[][TILES_K][4], float o_scale[2], float m[][2], float d[][2],
                                           const uint32_t iter, const uint32_t num_iterations, const float dequant_scale,
                                           const float original_sm_scale, const uint32_t K_idx_lane_base, const uint32_t kv_len)
{
  int32_t RS[1][TILES_K][8];
  compute_int_qk(smem_Q, smem_K, RS, offset_Q, offset_K);
  float RS_f32[1][TILES_K][8];
  const float scale = s_to_float(RS, RS_f32, iter, num_iterations, dequant_scale, original_sm_scale, K_idx_lane_base, kv_len);
  update_md(RS_f32, m, d, o_scale, scale);
  RS_32_to_16<1, TILES_K>(RS_f32, RS_f16);
  accumulate_d(RS_f16, d);
}

// The P.V step of the tuned sm_80 kernel (sc3d_sv_fv: SV_GROUP output column tiles at a time, each summing its four
// 16-key steps in fp16) with the m16n8k8 product; then RO = fma(RO, o_scale, the sum). The shared-memory offsets are
// sc3d_sv_fv's closed form of the k128B walk.
__device__ __forceinline__ void compute_sv(const smem_t<SwizzleMode::k128B, V_STRIDE> &smem_V, uint32_t RS_f16[][TILES_K][4], float RO[][TILES_V][8],
                                           const float o_scale[2], const uint32_t offset_V)
{
#pragma unroll
  for (uint32_t fv0 = 0; fv0 < TILES_V; fv0 += SV_GROUP)
  {
    uint32_t buf[SV_GROUP][4];
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
#pragma unroll
      for (uint32_t j = 0; j < SV_GROUP; j++)
      {
        const uint32_t fv = fv0 + j;
        uint32_t RV[4];
        smem_V.ldmatrix_m8n8x4_trans(((offset_V + (fv / 4) * 8) ^ (2 * (fv % 4))) + fk * 16 * V_STRIDE, RV);
        if (fk == 0)
          mma::mma_sync_m16n16k16_row_col_f16f16f16_sm75<mma::MMAMode::kInit>(buf[j], RS_f16[0][fk], RV);
        else
          mma::mma_sync_m16n16k16_row_col_f16f16f16_sm75<mma::MMAMode::kInplaceUpdate>(buf[j], RS_f16[0][fk], RV);
      }
    }
    // accumulator elements 0, 1, 4, 5: row g; 2, 3, 6, 7: row g + 8
#pragma unroll
    for (uint32_t j = 0; j < SV_GROUP; j++)
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
  smem_t<SwizzleMode::k128B, O_STRIDE> smem_O(smem);

  // the ping-pong (round K's int8 P.V kernel, round J's sv_kernel_pp.cuh): warps 0-3 (group A) run each block w in
  // order - the V(w) load, Q.K^T, softmax, P.V; warps 4-7 (group B, one on each scheduler beside an A warp) run half a
  // block behind - the K(w + 1) load, P.V of block w - 1 (its P and o_scale carried in registers), then Q.K^T and
  // softmax of block w - so while one warp of a scheduler is in its softmax the other has tensor work. Per warp the
  // operations and their order are those of one block after the other. Two K and two V buffers, one block barrier a
  // trip and a group barrier for A's V. Q (rows of 128 bytes, 8 lanes a row) by all warps; K by group B, V (rows of
  // 256 bytes) by group A, 16 rows a warp.
  const bool group_a = warp_id < GROUP_WARPS;
  const uint32_t gw = warp_id % GROUP_WARPS;

  const int8_t *Q_lane_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8) * stride_seq_q + (lane_id % 8) * 16;
  const int8_t *K_lane_ptr = K + batch_id * stride_bz_k + kv_head * stride_h_k + (CTA_K / GROUP_WARPS * gw + lane_id / 8) * stride_seq_k + (lane_id % 8) * 16;
  const int8_t *V_lane_ptr = reinterpret_cast<const int8_t*>(V + batch_id * stride_bz_v + kv_head * stride_h_v + (CTA_K / GROUP_WARPS * gw + lane_id / 8) * stride_seq_v + (lane_id % 8) * 8);
  const uint32_t Q_load_idx = bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8;
  const uint32_t KV_load_idx = CTA_K / GROUP_WARPS * gw + lane_id / 8;
  const uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(CTA_Q / NUM_WARPS * warp_id + lane_id / 8, lane_id % 8);
  const uint32_t K_smem_offset_load = smem_K.get_permuted_offset(CTA_K / GROUP_WARPS * gw + lane_id / 8, lane_id % 8);
  const uint32_t V_smem_offset_load = smem_V.get_permuted_offset(CTA_K / GROUP_WARPS * gw + lane_id / 8, lane_id % 8);
  const uint32_t K_stride_bytes = stride_seq_k;
  const uint32_t V_stride_bytes = stride_seq_v * sizeof(half);

  const uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(warp_id * WARP_Q + lane_id % 16, lane_id / 16);
  const uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  const uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(lane_id % 16, lane_id / 16);

  const uint32_t K_idx_lane_base = 2 * (lane_id % 4);

  using QTile = tile_t<CTA_Q, QK_STRIDE, NUM_WARPS>;
  using KTile = tile_t<CTA_K, QK_STRIDE, GROUP_WARPS>;
  using VTile = tile_t<CTA_K, V_STRIDE, GROUP_WARPS>;
  {
    uint4 RQ_load[QTile::REGS];
    load_tile<CTA_Q, QK_STRIDE, NUM_WARPS, true>(RQ_load, Q_lane_ptr, stride_seq_q, Q_load_idx, qo_len);
    store_tile<CTA_Q, QK_STRIDE, NUM_WARPS>(RQ_load, smem_Q, Q_smem_offset_load);
  }
  if (!group_a)
  {
    uint4 RK_load[KTile::REGS];
    load_tile<CTA_K, QK_STRIDE, GROUP_WARPS, true>(RK_load, K_lane_ptr, K_stride_bytes, KV_load_idx, kv_len);
    store_tile<CTA_K, QK_STRIDE, GROUP_WARPS>(RK_load, smem_K, K_smem_offset_load);
    K_lane_ptr += CTA_K * K_stride_bytes;
  }
  const float q_scale = Q_scale[q_scale_idx];
  const float original_sm_scale = sm_scale;
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
        qk_softmax(smem_Q, smem_K, Q_smem_offset_mma, K_smem_offset_mma + (w & 1) * K_BUF_UNITS, RS_f16, o_scale, m, d,
                   w, num_iterations, q_scale * K_scale_ptr[w], original_sm_scale, K_idx_lane_base, kv_len);
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
        qk_softmax(smem_Q, smem_K, Q_smem_offset_mma, K_smem_offset_mma + (w & 1) * K_BUF_UNITS, RS_f16, o_scale, m, d,
                   w, num_iterations, q_scale * K_scale_ptr[w], original_sm_scale, K_idx_lane_base, kv_len);
      }
    }
    __syncthreads();
  }

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
    ((uint32_t*)(smem_O.base + offset_O + 8 * O_STRIDE))[lane_id % 4] = RO_16[1];
    // ! permuted, make sure you know what you are doing (as sm_80)
    ((uint32_t*)(smem_O.base + (offset_O ^ 0x1)))[lane_id % 4] = RO_16[2];
    ((uint32_t*)(smem_O.base + (offset_O ^ 0x1) + 8 * O_STRIDE))[lane_id % 4] = RO_16[3];
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
