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

// Spellcast3D round L: the int8 P.V twin of the flows' kernel (qk_int_sv_i8_cuda_sm80.cu, round K) for Turing (sm_75),
// the op qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d with the sm_80 op's arguments and contract: q / k as the fp16 P.V
// kernel takes them, P as u8 per row and 64-key block relative to the block's maximum, v as s8 per 64-key block and
// channel with its mean subtracted (V^T from quant_per_block_int8_v_cuda, compiled here from the sm_80 file). The
// softmax and the P / v codes are that file's own functions (update_mdo_blk, pack_p), the products exact integer sums
// on mma m8n8k16 u8 x s8 (each sm_80 m16n8k32 as four, on the same fragments), the accumulators' update sm_80's
// expression - the same arithmetic as the sm_80 kernel. The schedule is qk_int_sv_f16_cuda_sm75.cu's: the ping-pong of
// the two warp groups with register-staged loads.

#define SC3D_I8_QUANT_ONLY
#include "qk_int_sv_i8_cuda_sm80.cu"
#undef SC3D_I8_QUANT_ONLY

#include "attn_utils_sm75.cuh"

namespace sc3d_sm75_i8 {

using namespace sc3d_sm75;

constexpr uint32_t VT_STRIDE = CTA_K / 16;       // 16-byte units in a row of V^T (64 keys, int8)

// Q int8 [128][128], two K int8 [64][128], two V^T int8 [128][64], two blocks' v scales fp32 [128]: 50,176 bytes, one
// block an SM; O [128][128] 16-bit at the end
constexpr uint32_t K_BUF_UNITS = CTA_K * HEAD / 16;
constexpr uint32_t V_BUF_UNITS = HEAD * CTA_K / 16;
constexpr uint32_t SMEM_K_OFFSET = CTA_Q * HEAD;
constexpr uint32_t SMEM_V_OFFSET = SMEM_K_OFFSET + 2 * CTA_K * HEAD;
constexpr uint32_t SMEM_VS_OFFSET = SMEM_V_OFFSET + 2 * HEAD * CTA_K;
constexpr size_t SMEM_BYTES = SMEM_VS_OFFSET + 2 * HEAD * sizeof(float);
static_assert(SMEM_BYTES >= CTA_Q * HEAD * sizeof(half) && SMEM_BYTES <= 65536, "O fits; Turing's 64 KB a block");

// V^T of one block (128 rows of 64 bytes; padded: no predicate) by the GROUP_WARPS warps of group A: 32 rows a warp,
// 4 lanes a row, 8 rows a step (sm_80's load_fp8_V_global_to_share walk, k64B swizzle)
constexpr uint32_t VT_REGS = HEAD / GROUP_WARPS / 8;

__device__ __forceinline__ void load_vt(uint4 R[VT_REGS], const int8_t *lane_ptr, const uint32_t stride_d)
{
#pragma unroll
  for (uint32_t i = 0; i < VT_REGS; i++)
    R[i] = __ldg(reinterpret_cast<const uint4*>(lane_ptr + i * 8 * stride_d));
}

__device__ __forceinline__ void store_vt(const uint4 R[VT_REGS], const smem_t<SwizzleMode::k64B, VT_STRIDE> &smem, const uint32_t offset)
{
#pragma unroll
  for (uint32_t i = 0; i < VT_REGS; i++)
    smem.base[offset + i * 8 * VT_STRIDE] = R[i];
}

// compute_sv_t of the sm_80 file with the m8n8k16 u8 x s8 product: per 16-channel tile the int32 product over the 64
// keys, then RO = product * (pmax / 255) * v_scale + RO (* o_scale with kRescale) per row and channel
template <bool kRescale>
__device__ __forceinline__ void compute_sv_t(const smem_t<SwizzleMode::k64B, VT_STRIDE> &smem_V, const float *smem_VS, uint32_t RP[2][4],
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
    uint32_t RV[4];
    // rows: channels fv * 16 + (lane % 8) + 8 * (lane / 16), 16-byte units: keys 0-15 / 16-31 (+ 32 for the second step)
    smem_V.ldmatrix_m8n8x4(offset_V0 + fv * 16 * VT_STRIDE, RV);
    mma::mma_sync_m16n16k32_row_col_x8s8s32_sm75<mma::MMAMode::kInit, true>(acc, RP[0], RV);
    smem_V.ldmatrix_m8n8x4(offset_V1 + fv * 16 * VT_STRIDE, RV);
    mma::mma_sync_m16n16k32_row_col_x8s8s32_sm75<mma::MMAMode::kInplaceUpdate, true>(acc, RP[1], RV);
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
  }
}

__device__ __forceinline__ void compute_sv(const smem_t<SwizzleMode::k64B, VT_STRIDE> &smem_V, const float *smem_VS, uint32_t RP[2][4],
                                           float RO[][TILES_V][8], const float pmax[2], const float o_scale[2], const bool rescale,
                                           const uint32_t offset_V0, const uint32_t offset_V1)
{
  if (rescale)
    compute_sv_t<true>(smem_V, smem_VS, RP, RO, pmax, o_scale, offset_V0, offset_V1);
  else
    compute_sv_t<false>(smem_V, smem_VS, RP, RO, pmax, o_scale, offset_V0, offset_V1);
}

// Q.K^T and the softmax of key block `iter`: P as the u8 A fragments (RP), the block's pmax, each row's o_scale, and
// whether any row of the warp has a new maximum (the sm_80 file's SC3D_I8_QK_SOFTMAX)
__device__ __forceinline__ void qk_softmax(const smem_t<SwizzleMode::k128B, QK_STRIDE> &smem_Q, const smem_t<SwizzleMode::k128B, QK_STRIDE> &smem_K,
                                           const uint32_t offset_Q, const uint32_t offset_K,
                                           uint32_t RP[2][4], float pmax[2], float o_scale[2], bool &rescale, float m[][2], float d[][2],
                                           const uint32_t iter, const uint32_t num_iterations, const float dequant_scale,
                                           const float original_sm_scale, const uint32_t K_idx_lane_base, const uint32_t kv_len)
{
  int32_t RS[1][TILES_K][8];
  compute_int_qk(smem_Q, smem_K, RS, offset_Q, offset_K);
  float RS_f32[1][TILES_K][8];
  const float scale = s_to_float(RS, RS_f32, iter, num_iterations, dequant_scale, original_sm_scale, K_idx_lane_base, kv_len);
  sc3d_i8::update_mdo_blk(RS_f32, m, d, pmax, o_scale, rescale, scale);
  sc3d_i8::pack_p(RS_f32, RP);
}

template <typename DTypeOut>
__global__ void __launch_bounds__(THREADS, 1) qk_int_sv_i8_attn_kernel(const int8_t *__restrict__ Q, const int8_t *__restrict__ K, const int8_t *__restrict__ V, DTypeOut *__restrict__ O,
                      const float *__restrict__ Q_scale, const float *__restrict__ K_scale, const float *__restrict__ V_scale, const float *__restrict__ V_mean,
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

  const uint32_t q_scale_idx = batch_id * num_qo_heads * (gridDim.x * NUM_WARPS) + head_id * (gridDim.x * NUM_WARPS) + bx * NUM_WARPS + warp_id;
  const float *K_scale_ptr = K_scale + batch_id * num_kv_heads * num_iterations + kv_head * num_iterations;

  smem_t<SwizzleMode::k128B, QK_STRIDE> smem_Q(smem);
  smem_t<SwizzleMode::k128B, QK_STRIDE> smem_K(smem + SMEM_K_OFFSET);
  smem_t<SwizzleMode::k64B, VT_STRIDE> smem_V(smem + SMEM_V_OFFSET);
  float *smem_VS = (float*)(smem + SMEM_VS_OFFSET);
  smem_t<SwizzleMode::k128B, O_STRIDE> smem_O(smem);

  // the ping-pong of qk_int_sv_f16_cuda_sm75.cu: group A (warps 0-3) the V^T(w) load (and, by its first warp, the
  // block's 128 v scales), Q.K^T, softmax, P.V of block w; group B (warps 4-7) the K(w + 1) load, P.V of block w - 1
  // (P, pmax, o_scale and the rescale flag carried), Q.K^T and softmax of block w
  const bool group_a = warp_id < GROUP_WARPS;
  const uint32_t gw = warp_id % GROUP_WARPS;

  const int8_t *Q_lane_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8) * stride_seq_q + (lane_id % 8) * 16;
  const int8_t *K_lane_ptr = K + batch_id * stride_bz_k + kv_head * stride_h_k + (CTA_K / GROUP_WARPS * gw + lane_id / 8) * stride_seq_k + (lane_id % 8) * 16;
  const int8_t *V_lane_ptr = V + batch_id * stride_bz_v + kv_head * stride_h_v + (HEAD / GROUP_WARPS * gw + lane_id / 4) * stride_d_v + (lane_id % 4) * 16;
  const float *VS_lane_ptr = V_scale + (batch_id * num_kv_heads + kv_head) * num_iterations * HEAD + lane_id * 4;
  const uint32_t Q_load_idx = bx * CTA_Q + CTA_Q / NUM_WARPS * warp_id + lane_id / 8;
  const uint32_t KV_load_idx = CTA_K / GROUP_WARPS * gw + lane_id / 8;
  const uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(CTA_Q / NUM_WARPS * warp_id + lane_id / 8, lane_id % 8);
  const uint32_t K_smem_offset_load = smem_K.get_permuted_offset(CTA_K / GROUP_WARPS * gw + lane_id / 8, lane_id % 8);
  const uint32_t V_smem_offset_load = smem_V.get_permuted_offset(HEAD / GROUP_WARPS * gw + lane_id / 4, lane_id % 4);
  const uint32_t K_stride_bytes = stride_seq_k;

  const uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(warp_id * WARP_Q + lane_id % 16, lane_id / 16);
  const uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  const uint32_t V_smem_offset_mma0 = smem_V.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  const uint32_t V_smem_offset_mma1 = smem_V.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2 + 2);

  const uint32_t K_idx_lane_base = 2 * (lane_id % 4);

  using QTile = tile_t<CTA_Q, QK_STRIDE, NUM_WARPS>;
  using KTile = tile_t<CTA_K, QK_STRIDE, GROUP_WARPS>;
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

  uint32_t RP[2][4];
  float pmax[2];
  float o_scale[2];
  bool rescale;
  for (uint32_t w = 0; w <= num_iterations; w++)
  {
    // here K(w) is in buffer w % 2 and V^T(w - 1) in buffer (w - 1) % 2, both visible (the barrier closing the trip)
    if (group_a)
    {
      if (w < num_iterations)
      {
        // V^T(w) and its scales into buffer w % 2 (it held block w - 2's, whose readers finished before the last
        // barrier)
        uint4 RV_load[VT_REGS];
        uint4 RVS_load;
        load_vt(RV_load, V_lane_ptr, stride_d_v);
        V_lane_ptr += CTA_K;
        if (gw == 0)
          RVS_load = __ldg(reinterpret_cast<const uint4*>(VS_lane_ptr));
        VS_lane_ptr += HEAD;
        qk_softmax(smem_Q, smem_K, Q_smem_offset_mma, K_smem_offset_mma + (w & 1) * K_BUF_UNITS, RP, pmax, o_scale, rescale, m, d,
                   w, num_iterations, q_scale * K_scale_ptr[w], original_sm_scale, K_idx_lane_base, kv_len);
        store_vt(RV_load, smem_V, V_smem_offset_load + (w & 1) * V_BUF_UNITS);
        if (gw == 0)
          reinterpret_cast<uint4*>(smem_VS + (w & 1) * HEAD)[lane_id] = RVS_load;
        asm volatile("bar.sync 1, %0;\n" :: "n"(GROUP_WARPS * 32) : "memory");     // group A's V^T(w) visible to group A
        compute_sv(smem_V, smem_VS + (w & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + (w & 1) * V_BUF_UNITS, V_smem_offset_mma1 + (w & 1) * V_BUF_UNITS);
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
        compute_sv(smem_V, smem_VS + ((w - 1) & 1) * HEAD, RP, RO, pmax, o_scale, rescale,
                   V_smem_offset_mma0 + ((w - 1) & 1) * V_BUF_UNITS, V_smem_offset_mma1 + ((w - 1) & 1) * V_BUF_UNITS);
      }
      if (has_next)
      {
        store_tile<CTA_K, QK_STRIDE, GROUP_WARPS>(RK_load, smem_K, K_smem_offset_load + ((w + 1) & 1) * K_BUF_UNITS);
      }
      if (w < num_iterations)
      {
        qk_softmax(smem_Q, smem_K, Q_smem_offset_mma, K_smem_offset_mma + (w & 1) * K_BUF_UNITS, RP, pmax, o_scale, rescale, m, d,
                   w, num_iterations, q_scale * K_scale_ptr[w], original_sm_scale, K_idx_lane_base, kv_len);
      }
    }
    __syncthreads();
  }

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

} // namespace sc3d_sm75_i8

// The sm_80 op's arguments (qk_int_sv_i8_cuda_sm80.cu): query, key int8 [B, N, H, 128] (tensor_layout 0) or
// [B, H, N, 128] (1), value V^T [B, H_kv, 128, Npad] int8 with value_scale [B, H_kv, Npad / 64, 128] and value_mean
// [B, H_kv, 128] fp32 as quant_per_block_int8_v_cuda makes them, output bf16 or fp16; head 128, not causal, per_warp
// q / k scales (BLKQ 128, WARPQ 16, BLKK 64), no LSE (anything else raises).
torch::Tensor qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d_sm75(torch::Tensor query,
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
  using namespace sc3d_sm75_i8;

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
              "qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d (sm_75): implemented for head 128, not causal, per_warp, no LSE only");

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
