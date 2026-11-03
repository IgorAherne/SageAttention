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

// Spellcast3D round L: what the Turing (sm_75) kernels of the flows' attention share - the configuration of sm_80's
// tuned kernel (head 128, CTA_Q 128, WARP_Q 16, CTA_K 64, per-warp q and per-block k scales), register-staged tile
// loads in place of cp.async, and the int8 Q.K^T on mma m8n8k16.

#pragma once
#include "../utils.cuh"
#include <cuda_fp16.h>

#include "../mma.cuh"
#include "../permuted_smem.cuh"

#include "attn_utils.cuh"

namespace sc3d_sm75 {

constexpr uint32_t CTA_Q = 128;
constexpr uint32_t CTA_K = 64;
constexpr uint32_t WARP_Q = 16;
constexpr uint32_t HEAD = 128;
constexpr uint32_t NUM_WARPS = CTA_Q / WARP_Q;
constexpr uint32_t THREADS = 32 * NUM_WARPS;
constexpr uint32_t GROUP_WARPS = NUM_WARPS / 2;  // the ping-pong's two warp groups
constexpr uint32_t TILES_K = CTA_K / 16;         // 16-key tiles of S per warp
constexpr uint32_t TILES_QK_INNER = HEAD / 32;   // k32 steps of Q.K^T
constexpr uint32_t TILES_V = HEAD / 16;          // 16-channel tiles of O
constexpr uint32_t QK_STRIDE = HEAD / 16;        // 16-byte units in a row of Q or K (int8)
constexpr uint32_t O_STRIDE = HEAD / 8;          // ... of O (16-bit)

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

// compute_int_qk (attn_utils.cuh, Q and K from shared memory, one 16-row tile) with the m8n8k16 product: the same S
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
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);
      if (iter == 0)
        mma::mma_sync_m16n16k32_row_col_x8s8s32_sm75<mma::MMAMode::kInit>(RS[0][fk], RQ, RK);
      else
        mma::mma_sync_m16n16k32_row_col_x8s8s32_sm75<mma::MMAMode::kInplaceUpdate>(RS[0][fk], RQ, RK);
    }
    offset_K = smem_K.advance_offset_by_column<2>(offset_K - (TILES_K * 16 * QK_STRIDE), iter);
  }
}

// S of key block `iter` as a float, as sm_80's kernels convert it: the main blocks unscaled (their softmax takes
// sm_scale * dequant), the last two times dequant (sm_scale), the last's out-of-bound keys masked. Returns the scale
// for the softmax.
__device__ __forceinline__ float s_to_float(int32_t RS[][TILES_K][8], float RS_f32[][TILES_K][8], const uint32_t iter, const uint32_t num_iterations,
                                            const float dequant_scale, const float original_sm_scale, const uint32_t K_idx_lane_base, const uint32_t kv_len)
{
  if (iter + 2 < num_iterations)
  {
#pragma unroll
    for (uint32_t fk = 0; fk < TILES_K; fk++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
      {
        RS_f32[0][fk][k] = __int2float_rz(RS[0][fk][k]);
      }
    }
    return original_sm_scale * dequant_scale;
  }
#pragma unroll
  for (uint32_t fk = 0; fk < TILES_K; fk++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 8; k++)
    {
      RS_f32[0][fk][k] = __int2float_rz(RS[0][fk][k]) * dequant_scale;
    }
  }
  if (iter + 1 == num_iterations)
  {
    apply_out_of_bound_mask<1, TILES_K>(K_idx_lane_base + iter * CTA_K, RS_f32, kv_len);
  }
  return original_sm_scale;
}

} // namespace sc3d_sm75
