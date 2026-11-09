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

// Spellcast3D round L: SageAttention for Pascal (sm_61, the GTX 10xx), which has no tensor cores but dp4a: four int8
// products added into an int32 by one instruction, at the fp32 FMA's issue rate (4x its multiply-adds).
//
// The scheme is the per_warp one of the sm80 kernels: q to int8 with one scale per 16 tokens, k with its mean over the
// tokens subtracted (per channel; it adds the same constant to every score of a query, which the softmax takes out)
// and one scale per 64 tokens, scale = absmax / 127 and code = rint(x * 127 / absmax). The scores are exact int32 dot
// products (dp4a) times the two scales; the online softmax runs in fp32 with exp2 (the softmax scale holds log2(e)).
// The P.V product in one of two ways:
//  - fp32: P in fp32 and v converted to fp32 in shared memory (v fp32, fp16 or bf16 in global memory), fp32 FMAs;
//  - int8: P as u8 per query and 32-key block relative to the block's maximum (rint(255 * exp2(s - s_max_block)),
//    scale exp2(s_max_block - m) / 255 with m the running maximum), v as s8 per 32-key block and channel with the
//    channel's mean over the tokens subtracted (scale = absmax / 127; added back after the normalisation: the
//    weights sum to 1), the u8 x s8 products by dp4a; the denominator from the fp32 P (round K's sm80 kernel's way).
// Both run on one kernel shape, taken from the image engine's GTX 10xx attention (Spellcast3D's ggml fork,
// fattn-tile-dp4a.cu): a block of 8 warps works on 64 queries of one head, 32 keys an iteration; the scores of a
// lane are a 2 x 4 register tile (2 queries x 4 keys, 32 dp4a per 16-byte chunk pair); for P.V a lane owns its warp's
// 8 queries x 4 output channels; the next K and V tiles travel through registers one iteration ahead, two barriers an
// iteration. Exact int32 sums are read as floats by one FADD: they start at the bits of 1.5 * 2^23 (|sum| < 2^22),
// instead of an I2F, a quarter-rate instruction on Pascal.
//
// Layouts made by the quantisers here (the attention kernels read them; Npad: the tokens rounded up to 64, the pad
// rows zeros): q / k int8 [B, H, Npad, 128], q scales [B, H, Npad / 16], k scales [B, H, Npad / 64], k mean [B, H,
// 128] fp32; v for the int8 P.V as words of 4 keys of one channel [B, H, Npad / 4, 128] (byte i = key 4 * row + i),
// its scales [B, H, Npad / 32, 128] and mean [B, H, 128] fp32. Deterministic: fixed summation orders, no atomics.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>
#include <type_traits>

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../utils.cuh"
#include "../attn_cuda_sm61.h"

// compile-time knobs of the main loop's unrolling (Pascal's instruction cache holds ~8 KB of a loop)
#ifndef SC3D_P61_QK_UNROLL
#define SC3D_P61_QK_UNROLL 4    // the score loop over 8 chunks of 16 channels
#endif
#ifndef SC3D_P61_PV_UNROLL
#define SC3D_P61_PV_UNROLL 8    // the fp32 P.V loop over 32 keys / the int8 one over 8 words of 4 keys
#endif
// timing study only (0 in the package; wrong results otherwise): 1 no score loop, 2 no P.V loop, 4 no global loads in
// the main loop, 8 no barriers in the main loop, 16 no exp2 / P stores; round P adds 32 no rescale of the
// accumulators, 64 no shuffles of the block maxima, 128 no P.V epilogue (the 32 FMUL + FADD + FFMA a lane), 256 no
// score conversion (the 8 FADD + FMUL), 512 no q loads in the score loop (each lane's two q rows of chunk 0 reused)
#ifndef SC3D_P61_STUDY
#define SC3D_P61_STUDY 0
#endif
// Spellcast3D round P (pascal-attn): exact reschedules of the int8 P.V kernel (the fp32 P.V kernel stays the shipped
// one in every variant of this file). Each output is still one lane's fma chain over the keys in the shipped order
// with the shipped factors, the integer sums the same sums, the softmax the same operations on the same values: the
// shipped kernel's bits.
//  SC3D_P61_I8_WARPS  warps a block of the int8 P.V kernel (8 shipped: 64 queries, 2 blocks an SM; 4: 32 queries, 4
//                     blocks an SM - a barrier waits for 4 warps, the K / v tiles are staged twice as often)
//  SC3D_P61_DB        1: the int8 P.V kernel's K and v tiles double-buffered in shared memory, one barrier an
//                     iteration instead of two (ptxas pairs every BAR.SYNC with a MEMBAR.CTA on sm_61)
#ifndef SC3D_P61_I8_WARPS
#define SC3D_P61_I8_WARPS 8
#endif
#ifndef SC3D_P61_DB
#define SC3D_P61_DB 0
#endif
//  SC3D_P61_PIPE      1: the int8 P.V kernel software-pipelined by one iteration (qk_int8_attn_pipe_kernel: block i's
//                     softmax beside block i - 1's P.V in one basic block, K / v / P double-buffered, one barrier an
//                     iteration, the last iteration peeled); 8 warps a block
#ifndef SC3D_P61_PIPE
#define SC3D_P61_PIPE 0
#endif
//  SC3D_P61_K64       1: the int8 P.V kernel at 64 keys an iteration (qk_int8_attn_k64_kernel: the shipped kernel's two
//                     32-key blocks in order, one score loop reading a lane's q rows once for both), 8 warps a block
#ifndef SC3D_P61_K64
#define SC3D_P61_K64 0
#endif
// developers only: instantiate fp32 inputs / outputs alone (the what-if builds compile faster)
#ifndef SC3D_P61_F32_ONLY
#define SC3D_P61_F32_ONLY 0
#endif

namespace sc3d_p61 {

constexpr int HEAD = 128;
constexpr int BQ = 64;                  // queries a block
constexpr int NWARPS = 8;               // a warp: 8 queries
constexpr int THREADS = 32 * NWARPS;
constexpr int BK = 32;                  // keys an iteration (and the int8 P.V's P / v scale block)
constexpr int WARPQ = 16;               // tokens a q scale
constexpr int BLKK = 64;                // tokens a k scale
constexpr int TPAD = 64;                // tokens of the quantised layouts rounded up to this
constexpr int QKS = HEAD / 4 + 4;       // ints a Q / K row in shared memory (padded: conflict-free 16-byte reads)
constexpr int PS = BQ + 8;              // floats a row of P [key][query] (fp32 P.V)
constexpr int KW = BK / 4;              // words of 4 keys an iteration (int8 P.V)
constexpr int P8S = BQ + 8;             // words a row of the u8 P [key / 4][query]
constexpr int MAGIC = 0x4B400000;       // the bits of 1.5 * 2^23
constexpr float MAGIC_F = 12582912.0f;  // 1.5 * 2^23
constexpr float LOG2E = 1.4426950408889634f;
constexpr int QK_UNROLL = SC3D_P61_QK_UNROLL;   // as constants: GCC's preprocessor does not expand a macro in a pragma
constexpr int PV_UNROLL = SC3D_P61_PV_UNROLL;

static_assert(BQ == 8 * NWARPS, "a warp works on 8 queries");
static_assert(BK == 32 && THREADS == 8 * BK, "a lane takes 2 queries x 4 keys of an iteration's scores");
static_assert(BLKK % BK == 0 && TPAD % BLKK == 0 && TPAD % BQ == 0 && BLKK % WARPQ == 0, "scale blocks");

// shared memory, in bytes: Q and K int8 rows, then the value tile and P (fp32 P.V: v [key][channel] and P
// [key][query] in fp32; int8 P.V: v words [key / 4][channel] and its scales [channel], P words [key / 4][query] and
// their scales [query])
constexpr int SM_Q = 0;
constexpr int SM_K = SM_Q + BQ * QKS * 4;
constexpr int SM_V = SM_K + BK * QKS * 4;
constexpr int SM_P_F32 = SM_V + BK * HEAD * 4;
constexpr int SM_END_F32 = SM_P_F32 + BK * PS * 4;
constexpr int SM_VS_I8 = SM_V + KW * HEAD * 4;
constexpr int SM_P_I8 = SM_VS_I8 + HEAD * 4;
constexpr int SM_PS_I8 = SM_P_I8 + KW * P8S * 4;
constexpr int SM_END_I8 = SM_PS_I8 + BQ * 4;
static_assert(SM_END_F32 <= 48 * 1024 && SM_END_I8 <= 48 * 1024, "static shared memory: 48 KB a block");

// round P: the attention kernel's block shape and shared memory per P.V mode (the quantisers keep NWARPS / THREADS;
// with the defaults this is the shipped layout byte for byte)
template <bool kInt8PV>
struct Cfg
{
  static constexpr int W = kInt8PV ? SC3D_P61_I8_WARPS : NWARPS;   // warps a block
  static constexpr int T = 32 * W;                                  // threads a block
  static constexpr int Q = 8 * W;                                   // queries a block
  static constexpr int MINB = 16 / W;                               // blocks an SM (16 warps)
  static constexpr bool DB = kInt8PV && SC3D_P61_DB != 0;           // K / v tiles double-buffered, one barrier
  static constexpr int NB = DB ? 2 : 1;                             // tile buffers
  static constexpr int PIECES = 8 * BK / T;     // 16-byte pieces a thread of the K tile (256) and of the int8 v tile
  static constexpr int PSTRIDE = Q + 8;         // a row of P: floats (fp32 P.V) / words (int8)
  static constexpr int K_BYTES = BK * QKS * 4;
  static constexpr int V_BYTES = kInt8PV ? KW * HEAD * 4 + HEAD * 4 : BK * HEAD * 4;   // int8: words + scales
  static constexpr int SM_Q = 0;
  static constexpr int SM_K = SM_Q + Q * QKS * 4;
  static constexpr int SM_V = SM_K + NB * K_BYTES;
  static constexpr int SM_P = SM_V + NB * V_BYTES;
  static constexpr int SM_PS = SM_P + (kInt8PV ? KW : BK) * PSTRIDE * 4;
  static constexpr int SM_END = SM_PS + (kInt8PV ? Q * 4 : 0);
  static_assert(W == 4 || W == 8, "4 or 8 warps a block");
  static_assert(PIECES * T == 8 * BK && TPAD % Q == 0 && (Q % WARPQ == 0 || WARPQ % Q == 0), "tiles");
  static_assert(SM_END <= 48 * 1024, "static shared memory: 48 KB a block");
};
static_assert(Cfg<false>::SM_END == SM_END_F32 && (SC3D_P61_I8_WARPS != 8 || SC3D_P61_DB || Cfg<true>::SM_END == SM_END_I8),
              "the defaults are the shipped layout");

// ---- element access: 4 consecutive values of fp32 / fp16 / bf16

template <typename T> struct Raw4;      // 4 values as loaded from global memory
template <> struct Raw4<float> { using type = float4; };
template <> struct Raw4<half> { using type = uint2; };
template <> struct Raw4<nv_bfloat16> { using type = uint2; };

__device__ __forceinline__ float4 to_float4(const float4 r, const float *) { return r; }
__device__ __forceinline__ float4 to_float4(const uint2 r, const half *)
{
  const float2 a = __half22float2(*reinterpret_cast<const half2*>(&r.x));
  const float2 b = __half22float2(*reinterpret_cast<const half2*>(&r.y));
  return make_float4(a.x, a.y, b.x, b.y);
}
__device__ __forceinline__ float4 to_float4(const uint2 r, const nv_bfloat16 *)
{
  const float2 a = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&r.x));
  const float2 b = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&r.y));
  return make_float4(a.x, a.y, b.x, b.y);
}

template <typename T>
__device__ __forceinline__ float4 load4(const T *p)
{
  return to_float4(*reinterpret_cast<const typename Raw4<T>::type*>(p), p);
}

__device__ __forceinline__ void store4(float *p, const float4 v) { *reinterpret_cast<float4*>(p) = v; }
__device__ __forceinline__ void store4(half *p, const float4 v)
{
  const half2 a = __floats2half2_rn(v.x, v.y), b = __floats2half2_rn(v.z, v.w);
  *reinterpret_cast<uint2*>(p) = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}
__device__ __forceinline__ void store4(nv_bfloat16 *p, const float4 v)
{
  const nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
  *reinterpret_cast<uint2*>(p) = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}

template <typename T>
__device__ __forceinline__ float to_float(const T x)
{
  if constexpr (std::is_same<T, float>::value)
    return x;
  else if constexpr (std::is_same<T, half>::value)
    return __half2float(x);
  else
    return __bfloat162float(x);
}

__device__ __forceinline__ float warp_max(float x)
{
#pragma unroll
  for (int o = 16; o > 0; o /= 2)
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o));
  return x;
}

// u8 x s8 dot product of 4 byte pairs added to c
__device__ __forceinline__ int dp4a_us(const int a_u8, const int b_s8, const int c)
{
  int d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a_u8), "r"(b_s8), "r"(c));
  return d;
}

__device__ __forceinline__ int pack_s8x4(const float a, const float b, const float c, const float d)
{
  const int qa = max(-127, min(127, __float2int_rn(a)));
  const int qb = max(-127, min(127, __float2int_rn(b)));
  const int qc = max(-127, min(127, __float2int_rn(c)));
  const int qd = max(-127, min(127, __float2int_rn(d)));
  return (qa & 0xff) | ((qb & 0xff) << 8) | ((qc & 0xff) << 16) | ((qd & 0xff) << 24);
}

// ---- the mean of every channel over the tokens (k's smoothing, v's for the int8 P.V): the sums of 64-token blocks,
// then their sum in order (deterministic)

template <typename T>
__global__ void __launch_bounds__(HEAD) block_sum_kernel(const T *__restrict__ x, float *__restrict__ part, const int n,
                                                         const int64_t stride_bz, const int64_t stride_seq, const int64_t stride_h)
{
  const int blk = blockIdx.x, h = blockIdx.y, b = blockIdx.z, c = threadIdx.x;
  const T *p = x + b * stride_bz + h * stride_h + c;
  const int r0 = blk * BLKK, r1 = min(n, r0 + BLKK);
  float s = 0.0f;
#pragma unroll 8
  for (int r = r0; r < r1; r++)
    s += to_float(p[r * stride_seq]);
  part[(((int64_t)b * gridDim.y + h) * gridDim.x + blk) * HEAD + c] = s;
}

__global__ void __launch_bounds__(HEAD) mean_kernel(const float *__restrict__ part, float *__restrict__ mean, const int nblk, const int n)
{
  const int bh = blockIdx.x, c = threadIdx.x;
  const float *p = part + (int64_t)bh * nblk * HEAD + c;
  float s = 0.0f;
  for (int j = 0; j < nblk; j++)
    s += p[(int64_t)j * HEAD];
  mean[(int64_t)bh * HEAD + c] = __fdiv_rn(s, (float)n);
}

// ---- q and k to int8: 64 tokens of one head a block of 4 warps (a warp 16 tokens, a lane 4 channels of each); the
// scale per warp (q: 16 tokens) or per block (k: 64 tokens); x minus the mean when one is given. The pad tokens past n
// become zeros.

template <typename T, bool kPerBlock>
__global__ void __launch_bounds__(128) quant_int8_kernel(const T *__restrict__ x, const float *__restrict__ mean,
                                                         int8_t *__restrict__ out, float *__restrict__ scale,
                                                         const int n, const int npad,
                                                         const int64_t stride_bz, const int64_t stride_seq, const int64_t stride_h)
{
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  const int tile = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int64_t bh = (int64_t)b * gridDim.y + h;
  const T *xp = x + b * stride_bz + h * stride_h + 4 * lane;
  const float4 mu = mean ? *reinterpret_cast<const float4*>(mean + bh * HEAD + 4 * lane) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int row0 = tile * 64 + 16 * w;

  float4 v[16];
  float amax = 0.0000001f;    // as the sm80 quantisers: no division by zero
#pragma unroll
  for (int r = 0; r < 16; r++)
  {
    v[r] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (row0 + r < n)
    {
      const float4 t = load4(xp + (row0 + r) * stride_seq);
      v[r] = make_float4(t.x - mu.x, t.y - mu.y, t.z - mu.z, t.w - mu.w);
    }
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[r].x), fabsf(v[r].y)), fmaxf(fabsf(v[r].z), fabsf(v[r].w))));
  }
  amax = warp_max(amax);
  if constexpr (kPerBlock)
  {
    __shared__ float wmax[4];
    if (lane == 0)
      wmax[w] = amax;
    __syncthreads();
    amax = fmaxf(fmaxf(wmax[0], wmax[1]), fmaxf(wmax[2], wmax[3]));
  }
  const float inv = __fdiv_rn(127.0f, amax);
  int8_t *op = out + (bh * npad + row0) * HEAD + 4 * lane;
#pragma unroll
  for (int r = 0; r < 16; r++)
    *reinterpret_cast<int*>(op + r * HEAD) = pack_s8x4(v[r].x * inv, v[r].y * inv, v[r].z * inv, v[r].w * inv);
  const float s = __fdiv_rn(amax, 127.0f);
  if constexpr (kPerBlock)
  {
    if (threadIdx.x == 0)
      scale[bh * (npad / 64) + tile] = s;
  }
  else
  {
    if (lane == 0)
      scale[bh * (npad / WARPQ) + tile * 4 + w] = s;
  }
}

// ---- v to int8 for the int8 P.V: one 32-key block of one head a block of 8 warps (warp w the keys 4w .. 4w + 3, a lane
// 4 channels); v minus its channel mean, one scale per channel (absmax / 127; 1 where the absmax is 0), the codes
// rint(x / scale); stored as words of 4 keys of one channel

template <typename T>
__global__ void __launch_bounds__(THREADS) quant_v_kernel(const T *__restrict__ v, const float *__restrict__ mean,
                                                          int *__restrict__ out, float *__restrict__ scale, const int n,
                                                          const int64_t stride_bz, const int64_t stride_seq, const int64_t stride_h)
{
  const int lane = threadIdx.x % 32, w = threadIdx.x / 32;
  const int blk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int nblk = gridDim.x;
  const int64_t bh = (int64_t)b * gridDim.y + h;
  const T *vp = v + b * stride_bz + h * stride_h + 4 * lane;
  const float4 mu = mean ? *reinterpret_cast<const float4*>(mean + bh * HEAD + 4 * lane) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  float4 x[4];
#pragma unroll
  for (int i = 0; i < 4; i++)
  {
    const int t = blk * BK + 4 * w + i;
    x[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (t < n)
    {
      const float4 r = load4(vp + t * stride_seq);
      x[i] = make_float4(r.x - mu.x, r.y - mu.y, r.z - mu.z, r.w - mu.w);
    }
  }
  __shared__ float4 cmax[NWARPS][32];
  float4 m = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int i = 0; i < 4; i++)
    m = make_float4(fmaxf(m.x, fabsf(x[i].x)), fmaxf(m.y, fabsf(x[i].y)), fmaxf(m.z, fabsf(x[i].z)), fmaxf(m.w, fabsf(x[i].w)));
  cmax[w][lane] = m;
  __syncthreads();
#pragma unroll
  for (int ww = 0; ww < NWARPS; ww++)
  {
    const float4 o = cmax[ww][lane];
    m = make_float4(fmaxf(m.x, o.x), fmaxf(m.y, o.y), fmaxf(m.z, o.z), fmaxf(m.w, o.w));
  }
  const float4 sv = make_float4(m.x > 0.0f ? __fdiv_rn(m.x, 127.0f) : 1.0f, m.y > 0.0f ? __fdiv_rn(m.y, 127.0f) : 1.0f,
                                m.z > 0.0f ? __fdiv_rn(m.z, 127.0f) : 1.0f, m.w > 0.0f ? __fdiv_rn(m.w, 127.0f) : 1.0f);
  // word j: channel 4 * lane + j, the 4 keys of this warp
  const int4 words = make_int4(
    pack_s8x4(__fdiv_rn(x[0].x, sv.x), __fdiv_rn(x[1].x, sv.x), __fdiv_rn(x[2].x, sv.x), __fdiv_rn(x[3].x, sv.x)),
    pack_s8x4(__fdiv_rn(x[0].y, sv.y), __fdiv_rn(x[1].y, sv.y), __fdiv_rn(x[2].y, sv.y), __fdiv_rn(x[3].y, sv.y)),
    pack_s8x4(__fdiv_rn(x[0].z, sv.z), __fdiv_rn(x[1].z, sv.z), __fdiv_rn(x[2].z, sv.z), __fdiv_rn(x[3].z, sv.z)),
    pack_s8x4(__fdiv_rn(x[0].w, sv.w), __fdiv_rn(x[1].w, sv.w), __fdiv_rn(x[2].w, sv.w), __fdiv_rn(x[3].w, sv.w)));
  *reinterpret_cast<int4*>(out + ((bh * nblk + blk) * KW + w) * HEAD + 4 * lane) = words;
  if (w == 0)
    *reinterpret_cast<float4*>(scale + (bh * nblk + blk) * HEAD + 4 * lane) = sv;
}

// ---- the attention

// kInt8PV: the int8 P.V (V8, V_scale, V_mean; V unused), else the fp32 one (V [B, N, H, 128] by its strides, TV its type).
// Round P: the shipped kernel's code with the block shape and the tile buffers of Cfg<kInt8PV> (the defaults: the
// shipped kernel); every value is computed by the same operations in the same order.
template <bool kInt8PV, typename TV, typename TO>
__global__ void __launch_bounds__(Cfg<kInt8PV>::T, Cfg<kInt8PV>::MINB) qk_int8_attn_kernel(
    const int *__restrict__ Q8, const float *__restrict__ Q_scale,
    const int *__restrict__ K8, const float *__restrict__ K_scale,
    const TV *__restrict__ V,
    const int *__restrict__ V8, const float *__restrict__ V_scale, const float *__restrict__ V_mean,
    TO *__restrict__ O,
    const int nq, const int nk, const int nq_pad, const int nk_pad,
    const int64_t stride_bz_v, const int64_t stride_seq_v, const int64_t stride_h_v,
    const int64_t stride_bz_o, const int64_t stride_seq_o, const int64_t stride_h_o,
    const float sm_scale_log2)
{
  using C = Cfg<kInt8PV>;
  constexpr int NT = C::T, QB = C::Q, NW = C::W, PSW = C::PSTRIDE, NP = C::PIECES;
  constexpr int KTW = C::K_BYTES / 4, VTW = C::V_BYTES / 4;      // ints a K / v tile buffer
  __shared__ __align__(16) unsigned char smem[C::SM_END];
  int *Q8s = reinterpret_cast<int*>(smem + C::SM_Q);
  int *K8s0 = reinterpret_cast<int*>(smem + C::SM_K);           // K buffer 0 (DB: buffer 1 follows)
  float *Vs = reinterpret_cast<float*>(smem + C::SM_V);         // fp32 P.V
  float *Ps = reinterpret_cast<float*>(smem + C::SM_P);
  int *V8s0 = reinterpret_cast<int*>(smem + C::SM_V);           // int8 P.V: buffer 0's words then its scales
  int *P8s = reinterpret_cast<int*>(smem + C::SM_P);
  float *Pss = reinterpret_cast<float*>(smem + C::SM_PS);

  const int tid = threadIdx.x;
  const int lane = tid % 32, w = tid / 32;
  const int q0 = blockIdx.x * QB, h = blockIdx.y, b = blockIdx.z;
  const int nh = gridDim.y;
  const int64_t bh = (int64_t)b * nh + h;

  // The scores: lane = 8 g + c (c = 0..7, hh = c % 2) takes its own query jc = 8 w + c, the other one of its pair jo = jc ^ 1
  // and the keys 8 j + 4 hh + g (j = 0..3) of an iteration (the shipped kernel's tile).
  const int c = lane % 8, g = lane / 8, hh = c % 2;
  const int jc = 8 * w + c, jo = jc ^ 1;

  const int *Q8h = Q8 + (bh * nq_pad + q0) * (HEAD / 4);
  const int *K8h = K8 + bh * nk_pad * (HEAD / 4);
  const float *Ksh = K_scale + bh * (nk_pad / BLKK);

  for (int i = tid; i < QB * (HEAD / 16); i += NT)
  {
    const int row = i / (HEAD / 16), seg = i % (HEAD / 16);
    *reinterpret_cast<int4*>(&Q8s[row * QKS + seg * 4]) = *reinterpret_cast<const int4*>(&Q8h[row * (HEAD / 4) + seg * 4]);
  }
  // the warp's 8 queries share one q scale (16 tokens); the softmax scale and log2(e) folded in
  const float qs = Q_scale[bh * (nq_pad / WARPQ) + (q0 + 8 * w) / WARPQ] * sm_scale_log2;

  // P.V: the lane accumulates the warp's 8 queries 8 w + cc, channels 4 lane .. 4 lane + 3
  float acc[8][4];
#pragma unroll
  for (int cc = 0; cc < 8; cc++)
  {
#pragma unroll
    for (int d = 0; d < 4; d++)
      acc[cc][d] = 0.0f;
  }
  float m_own = -FLT_MAX / 2.0f, m_oth = -FLT_MAX / 2.0f;    // running maxima of jc and jo (log2 units)
  float l_own = 0.0f, l_oth = 0.0f;                          // this lane's parts of their denominators

  // the next tiles from global memory through registers: K two iterations ahead, the values one (DB: K and v both
  // two iterations ahead of their use, one ahead of their store)
  int4 kreg[NP];
  using RawV = typename Raw4<TV>::type;
  RawV vreg[BK / NW];
  int4 v8reg[NP];
  float4 vsreg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const TV *Vh = V + b * stride_bz_v + h * stride_h_v + 4 * lane;
  const int *V8h = V8 + bh * (nk_pad / 4) * HEAD;
  const float *Vsh = V_scale + bh * (nk_pad / BK) * HEAD;
#pragma unroll
  for (int i = 0; i < NP; i++)
  {
    kreg[i] = make_int4(0, 0, 0, 0);
    v8reg[i] = make_int4(0, 0, 0, 0);
  }

  auto load_K = [&](const int k0) {      // keys k0 + tid / 8 + (NT / 8) i, bytes 16 (tid % 8) ..
#pragma unroll
    for (int i = 0; i < NP; i++)
      kreg[i] = *reinterpret_cast<const int4*>(&K8h[(int64_t)(k0 + tid / 8 + (NT / 8) * i) * (HEAD / 4) + (tid % 8) * 4]);
  };
  auto store_K = [&](int *K8s) {
#pragma unroll
    for (int i = 0; i < NP; i++)
      *reinterpret_cast<int4*>(&K8s[(tid / 8 + (NT / 8) * i) * QKS + (tid % 8) * 4]) = kreg[i];
  };
  auto load_V = [&](const int k0) {
    if constexpr (kInt8PV)
    {
      const int blk = k0 / BK;          // words [key / 4][channel]: piece tid + NT i
#pragma unroll
      for (int i = 0; i < NP; i++)
        v8reg[i] = *reinterpret_cast<const int4*>(&V8h[(int64_t)blk * KW * HEAD + 4 * (tid + NT * i)]);
      if (tid < HEAD / 4)
        vsreg = *reinterpret_cast<const float4*>(&Vsh[(int64_t)blk * HEAD + 4 * tid]);
    }
    else
    {
#pragma unroll
      for (int i = 0; i < BK / NW; i++)   // keys k0 + w + NW i, channels 4 lane ..
      {
        const int t = k0 + w + NW * i;
        vreg[i] = t < nk ? *reinterpret_cast<const RawV*>(Vh + t * stride_seq_v) : RawV();
      }
    }
  };
  auto store_V = [&](int *V8s) {         // int8: the words then the scales of one buffer
    if constexpr (kInt8PV)
    {
#pragma unroll
      for (int i = 0; i < NP; i++)
        *reinterpret_cast<int4*>(&V8s[4 * (tid + NT * i)]) = v8reg[i];
      if (tid < HEAD / 4)
        *reinterpret_cast<float4*>(&V8s[KW * HEAD + 4 * tid]) = vsreg;
    }
    else
    {
#pragma unroll
      for (int i = 0; i < BK / NW; i++)
        *reinterpret_cast<float4*>(&Vs[(w + NW * i) * HEAD + 4 * lane]) = to_float4(vreg[i], V);
    }
  };

  if constexpr (C::DB)
  {
    load_K(0);
    load_V(0);
    store_K(K8s0);
    store_V(V8s0);
    if (BK < nk)
    {
      load_K(BK);
      load_V(BK);
    }
    __syncthreads();
  }
  else
  {
    load_K(0);
    load_V(0);
    store_K(K8s0);
    __syncthreads();
    if (BK < nk)
      load_K(BK);
  }

  for (int k0 = 0, it = 0; k0 < nk; k0 += BK, it++)
  {
    // DB: this iteration's buffers; the next K tile goes into the other one now (its last reader, the previous
    // iteration's score loop, finished before the previous barrier), the K tile after it into kreg
    const int *K8s = C::DB ? K8s0 + (it & 1) * KTW : K8s0;
    int *V8s = C::DB ? V8s0 + (it & 1) * VTW : V8s0;
    if constexpr (C::DB)
    {
#if !(SC3D_P61_STUDY & 4)
      if (k0 + BK < nk)
        store_K(K8s0 + ((it + 1) & 1) * KTW);
      if (k0 + 2 * BK < nk)
        load_K(k0 + 2 * BK);
#endif
    }
    const float *Vscs = reinterpret_cast<const float*>(V8s + KW * HEAD);
    const float f = qs * Ksh[k0 / BLKK];
#if SC3D_P61_STUDY & 1
    int so[4], st[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
      so[j] = MAGIC + Q8s[jc * QKS + j] + K8s[(8 * j + 4 * hh + g) * QKS];
      st[j] = MAGIC + Q8s[jo * QKS + j];
    }
#else

    // the scores of jc (so) and jo (st) with the keys 8 j + 4 hh + g, as exact int32 sums from the bits of 1.5 * 2^23
    int so[4], st[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
      so[j] = MAGIC;
      st[j] = MAGIC;
    }
#if SC3D_P61_STUDY & 512
    const int4 qa0 = *reinterpret_cast<const int4*>(&Q8s[jc * QKS]);
    const int4 qb0 = *reinterpret_cast<const int4*>(&Q8s[jo * QKS]);
#endif
#pragma unroll QK_UNROLL
    for (int kc = 0; kc < HEAD / 16; kc++)
    {
#if SC3D_P61_STUDY & 512
      const int4 qa = qa0, qb = qb0;     // timing study: the score loop without its q loads (wrong scores)
#else
      const int4 qa = *reinterpret_cast<const int4*>(&Q8s[jc * QKS + kc * 4]);
      const int4 qb = *reinterpret_cast<const int4*>(&Q8s[jo * QKS + kc * 4]);
#endif
#pragma unroll
      for (int j = 0; j < 4; j++)
      {
        const int4 kk = *reinterpret_cast<const int4*>(&K8s[(8 * j + 4 * hh + g) * QKS + kc * 4]);
        so[j] = __dp4a(qa.x, kk.x, so[j]);
        so[j] = __dp4a(qa.y, kk.y, so[j]);
        so[j] = __dp4a(qa.z, kk.z, so[j]);
        so[j] = __dp4a(qa.w, kk.w, so[j]);
        st[j] = __dp4a(qb.x, kk.x, st[j]);
        st[j] = __dp4a(qb.y, kk.y, st[j]);
        st[j] = __dp4a(qb.z, kk.z, st[j]);
        st[j] = __dp4a(qb.w, kk.w, st[j]);
      }
    }
#endif

    float xo[4], xt[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
#if SC3D_P61_STUDY & 256
      xo[j] = __int_as_float(so[j]) * 1e-6f;
      xt[j] = __int_as_float(st[j]) * 1e-6f;
#else
      xo[j] = (__int_as_float(so[j]) - MAGIC_F) * f;
      xt[j] = (__int_as_float(st[j]) - MAGIC_F) * f;
#endif
    }
    if (k0 + BK > nk)   // the last iteration: keys past the end weigh 0
    {
#pragma unroll
      for (int j = 0; j < 4; j++)
      {
        if (k0 + 8 * j + 4 * hh + g >= nk)
        {
          xo[j] = -FLT_MAX;
          xt[j] = -FLT_MAX;
        }
      }
    }
    // the block maxima of jc and jo over the 32 keys: over the lane's 4, over g, then with the partner (lane ^ 1)
    float bo = fmaxf(fmaxf(xo[0], xo[1]), fmaxf(xo[2], xo[3]));
    float bt = fmaxf(fmaxf(xt[0], xt[1]), fmaxf(xt[2], xt[3]));
#if !(SC3D_P61_STUDY & 64)
    bo = fmaxf(bo, __shfl_xor_sync(0xffffffff, bo, 8));
    bt = fmaxf(bt, __shfl_xor_sync(0xffffffff, bt, 8));
    bo = fmaxf(bo, __shfl_xor_sync(0xffffffff, bo, 16));
    bt = fmaxf(bt, __shfl_xor_sync(0xffffffff, bt, 16));
    {
      const float bo_p = __shfl_xor_sync(0xffffffff, bt, 1);   // the partner's other query is this lane's own
      const float bt_p = __shfl_xor_sync(0xffffffff, bo, 1);
      bo = fmaxf(bo, bo_p);
      bt = fmaxf(bt, bt_p);
    }
#endif
    const float mo = fmaxf(m_own, bo), mt = fmaxf(m_oth, bt);
    const float co = exp2f(m_own - mo), ct = exp2f(m_oth - mt);
    m_own = mo;
    m_oth = mt;

    __syncwarp();       // the warp's P.V reads of its P (previous iteration) are done before it is rewritten
    if constexpr (kInt8PV)
    {
      // u8 codes relative to the block maximum: the low byte of the bits of 1.5 * 2^23 + 255 e (one FFMA, no F2I)
      unsigned char *P8b = reinterpret_cast<unsigned char*>(P8s);
      float so_sum = 0.0f, st_sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++)
      {
#if SC3D_P61_STUDY & 16
        const float eo = xo[j] - bo, et = xt[j] - bt;
#else
        const float eo = exp2f(xo[j] - bo), et = exp2f(xt[j] - bt);
#endif
        so_sum += eo;
        st_sum += et;
        const int row = 2 * j + hh;     // key 8 j + 4 hh + g: word row 2 j + hh, byte g
        P8b[4 * (row * PSW + jc) + g] = (unsigned char)__float_as_uint(fmaf(255.0f, eo, MAGIC_F));
        P8b[4 * (row * PSW + jo) + g] = (unsigned char)__float_as_uint(fmaf(255.0f, et, MAGIC_F));
      }
      const float po = exp2f(bo - mo), pt = exp2f(bt - mt);
      l_own = fmaf(po, so_sum, l_own * co);
      l_oth = fmaf(pt, st_sum, l_oth * ct);
      if (g == 0)
        Pss[jc] = po * (1.0f / 255.0f);
    }
    else
    {
      float so_sum = 0.0f, st_sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++)
      {
        const int t = 8 * j + 4 * hh + g;
#if SC3D_P61_STUDY & 16
        const float po = xo[j] - mo, pt = xt[j] - mt;
#else
        const float po = exp2f(xo[j] - mo), pt = exp2f(xt[j] - mt);
#endif
        so_sum += po;
        st_sum += pt;
        Ps[t * PSW + jc] = po;          // banks 8 g + c (+ 8 w): conflict-free
        Ps[t * PSW + jo] = pt;
      }
      l_own = l_own * co + so_sum;
      l_oth = l_oth * ct + st_sum;
    }

#if !(SC3D_P61_STUDY & 32)
    // the accumulators of the queries whose maximum moved (lane cc holds query 8 w + cc's factor)
    if (!__all_sync(0xffffffff, co == 1.0f))
    {
#pragma unroll
      for (int cc = 0; cc < 8; cc++)
      {
        const float cf = __shfl_sync(0xffffffff, co, cc);
#pragma unroll
        for (int d = 0; d < 4; d++)
          acc[cc][d] *= cf;
      }
    }
#endif

    if constexpr (C::DB)
    {
      // one barrier: every warp's P is complete, this iteration's v tile and the next K tile are visible, and every
      // warp is done with the previous v tile (the other buffer), which the next one now replaces
#if !(SC3D_P61_STUDY & 8)
      __syncthreads();
#endif
#if !(SC3D_P61_STUDY & 4)
      if (k0 + BK < nk)
        store_V(V8s0 + ((it + 1) & 1) * VTW);
      if (k0 + 2 * BK < nk)
        load_V(k0 + 2 * BK);
#endif
    }
    else
    {
#if !(SC3D_P61_STUDY & 8)
      __syncthreads();    // all warps are done with K8s and the previous value tile; every warp's P is complete
#endif
      store_V(V8s0);
#if !(SC3D_P61_STUDY & 4)
      if (k0 + BK < nk)
      {
        store_K(K8s0);
        load_V(k0 + BK);
      }
      if (k0 + 2 * BK < nk)
        load_K(k0 + 2 * BK);
#endif
#if !(SC3D_P61_STUDY & 8)
      __syncthreads();
#endif
    }

    if constexpr (kInt8PV)
    {
      // per output 8 dp4a over the 32 keys into an int32 from the bits of 1.5 * 2^23 (|sum| <= 32 * 255 * 127 < 2^22)
      int isum[8][4];
#pragma unroll
      for (int cc = 0; cc < 8; cc++)
      {
#pragma unroll
        for (int d = 0; d < 4; d++)
          isum[cc][d] = MAGIC;
      }
#if !(SC3D_P61_STUDY & 2)
#pragma unroll PV_UNROLL
      for (int kw = 0; kw < KW; kw++)
      {
        const int4 vv = *reinterpret_cast<const int4*>(&V8s[kw * HEAD + 4 * lane]);
        const int4 p0 = *reinterpret_cast<const int4*>(&P8s[kw * PSW + 8 * w]);
        const int4 p1 = *reinterpret_cast<const int4*>(&P8s[kw * PSW + 8 * w + 4]);
        const int p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
        const int v4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int cc = 0; cc < 8; cc++)
        {
#pragma unroll
          for (int d = 0; d < 4; d++)
            isum[cc][d] = dp4a_us(p[cc], v4[d], isum[cc][d]);
        }
      }
#else
      isum[0][0] += P8s[8 * w] + V8s[4 * lane];
#endif
      const float4 ps0 = *reinterpret_cast<const float4*>(&Pss[8 * w]);
      const float4 ps1 = *reinterpret_cast<const float4*>(&Pss[8 * w + 4]);
      const float4 vs = *reinterpret_cast<const float4*>(&Vscs[4 * lane]);
      const float ps[8] = {ps0.x, ps0.y, ps0.z, ps0.w, ps1.x, ps1.y, ps1.z, ps1.w};
      const float vsa[4] = {vs.x, vs.y, vs.z, vs.w};
#pragma unroll
      for (int cc = 0; cc < 8; cc++)
      {
#pragma unroll
        for (int d = 0; d < 4; d++)
#if SC3D_P61_STUDY & 128
          acc[cc][d] += __int_as_float(isum[cc][d]);
#else
          acc[cc][d] = fmaf(ps[cc] * vsa[d], __int_as_float(isum[cc][d]) - MAGIC_F, acc[cc][d]);
#endif
      }
    }
    else
    {
#if !(SC3D_P61_STUDY & 2)
#pragma unroll PV_UNROLL
      for (int t = 0; t < BK; t++)
      {
        const float4 vv = *reinterpret_cast<const float4*>(&Vs[t * HEAD + 4 * lane]);
        const float4 p0 = *reinterpret_cast<const float4*>(&Ps[t * PSW + 8 * w]);
        const float4 p1 = *reinterpret_cast<const float4*>(&Ps[t * PSW + 8 * w + 4]);
        const float p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
        const float v4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int cc = 0; cc < 8; cc++)
        {
#pragma unroll
          for (int d = 0; d < 4; d++)
            acc[cc][d] = fmaf(p[cc], v4[d], acc[cc][d]);
        }
      }
#else
      acc[0][0] += Ps[8 * w] + Vs[4 * lane];
#endif
    }
  }

  // the denominators: the lane's own part plus the partner's part of the same query, then over g
  float l = l_own + __shfl_xor_sync(0xffffffff, l_oth, 1);
  l += __shfl_xor_sync(0xffffffff, l, 8);
  l += __shfl_xor_sync(0xffffffff, l, 16);

  float4 mean = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if constexpr (kInt8PV)
    mean = *reinterpret_cast<const float4*>(V_mean + bh * HEAD + 4 * lane);
  TO *Oh = O + b * stride_bz_o + h * stride_h_o + 4 * lane;
#pragma unroll
  for (int cc = 0; cc < 8; cc++)
  {
    const float lc = __shfl_sync(0xffffffff, l, cc);   // lane cc holds query 8 w + cc's
    const int row = q0 + 8 * w + cc;
    if (row < nq)
    {
      const float inv = 1.0f / lc;
      store4(Oh + row * stride_seq_o, make_float4(fmaf(acc[cc][0], inv, mean.x), fmaf(acc[cc][1], inv, mean.y),
                                                  fmaf(acc[cc][2], inv, mean.z), fmaf(acc[cc][3], inv, mean.w)));
    }
  }
}

#if SC3D_P61_PIPE
// Round P: the int8 P.V kernel software-pipelined by one iteration. Iteration i computes the scores of keys block i,
// the integer P.V sums of block i - 1 (its P and v tiles from the other buffers), block i's softmax (P codes, maxima,
// denominators), then block i - 1's P.V epilogue and block i's rescale - all in one basic block, so that ptxas can
// fill the softmax's dependent chain (maxima shuffles, exp2) with the other block's dp4a. One barrier an iteration
// (K, v and P double-buffered); the last iteration is peeled (no tail tests in the steady loop). The accumulators see
// the shipped sequence rescale(i - 1), P.V(i - 1), rescale(i), P.V(i), ...; every value is computed by the shipped
// operations in the shipped order: the shipped kernel's bits. Block -1's P.V is all zeros (P and the scales zeroed).
template <typename TO>
__global__ void __launch_bounds__(THREADS, 2) qk_int8_attn_pipe_kernel(
    const int *__restrict__ Q8, const float *__restrict__ Q_scale,
    const int *__restrict__ K8, const float *__restrict__ K_scale,
    const int *__restrict__ V8, const float *__restrict__ V_scale, const float *__restrict__ V_mean,
    TO *__restrict__ O,
    const int nq, const int nk, const int nq_pad, const int nk_pad,
    const int64_t stride_bz_o, const int64_t stride_seq_o, const int64_t stride_h_o,
    const float sm_scale_log2)
{
  constexpr int KTW = BK * QKS;                 // ints a K buffer
  constexpr int VTW = KW * HEAD + HEAD;         // ints a v buffer: the words, then the scales
  constexpr int PTW = KW * P8S;                 // ints a P buffer
  constexpr int SMI = BQ * QKS + 2 * KTW + 2 * VTW + 2 * PTW + 2 * BQ;
  static_assert(SMI * 4 <= 48 * 1024, "static shared memory: 48 KB a block");
  __shared__ __align__(16) int sm[SMI];
  int *Q8s = sm;
  int *K8b = Q8s + BQ * QKS;
  int *V8b = K8b + 2 * KTW;
  int *P8w = V8b + 2 * VTW;
  float *Pssb = reinterpret_cast<float*>(P8w + 2 * PTW);

  const int tid = threadIdx.x;
  const int lane = tid % 32, w = tid / 32;
  const int q0 = blockIdx.x * BQ, h = blockIdx.y, b = blockIdx.z;
  const int nh = gridDim.y;
  const int64_t bh = (int64_t)b * nh + h;
  const int c = lane % 8, g = lane / 8, hh = c % 2;
  const int jc = 8 * w + c, jo = jc ^ 1;

  const int *Q8h = Q8 + (bh * nq_pad + q0) * (HEAD / 4);
  const int *K8h = K8 + bh * nk_pad * (HEAD / 4);
  const float *Ksh = K_scale + bh * (nk_pad / BLKK);
  for (int i = tid; i < BQ * (HEAD / 16); i += THREADS)
  {
    const int row = i / (HEAD / 16), seg = i % (HEAD / 16);
    *reinterpret_cast<int4*>(&Q8s[row * QKS + seg * 4]) = *reinterpret_cast<const int4*>(&Q8h[row * (HEAD / 4) + seg * 4]);
  }
  const float qs = Q_scale[bh * (nq_pad / WARPQ) + (q0 + 8 * w) / WARPQ] * sm_scale_log2;

  float acc[8][4];
#pragma unroll
  for (int cc = 0; cc < 8; cc++)
  {
#pragma unroll
    for (int d = 0; d < 4; d++)
      acc[cc][d] = 0.0f;
  }
  float m_own = -FLT_MAX / 2.0f, m_oth = -FLT_MAX / 2.0f;
  float l_own = 0.0f, l_oth = 0.0f;

  int4 kreg = make_int4(0, 0, 0, 0), v8reg = make_int4(0, 0, 0, 0);
  float4 vsreg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int *V8h = V8 + bh * (nk_pad / 4) * HEAD;
  const float *Vsh = V_scale + bh * (nk_pad / BK) * HEAD;
  auto load_K = [&](const int k0) {
    kreg = *reinterpret_cast<const int4*>(&K8h[(int64_t)(k0 + tid / 8) * (HEAD / 4) + (tid % 8) * 4]);
  };
  auto store_K = [&](int *Kb) {
    *reinterpret_cast<int4*>(&Kb[(tid / 8) * QKS + (tid % 8) * 4]) = kreg;
  };
  auto load_V = [&](const int k0) {
    const int blk = k0 / BK;
    v8reg = *reinterpret_cast<const int4*>(&V8h[(int64_t)blk * KW * HEAD + 4 * tid]);
    if (tid < HEAD / 4)
      vsreg = *reinterpret_cast<const float4*>(&Vsh[(int64_t)blk * HEAD + 4 * tid]);
  };
  auto store_V = [&](int *Vb) {
    *reinterpret_cast<int4*>(&Vb[4 * tid]) = v8reg;
    if (tid < HEAD / 4)
      *reinterpret_cast<float4*>(&Vb[KW * HEAD + 4 * tid]) = vsreg;
  };
  // the integer P.V sums of one block: per output 8 dp4a from the bits of 1.5 * 2^23
  auto pv_sums = [&](const int *Vb, const int *Pb, int (&isum)[8][4]) {
#pragma unroll
    for (int cc = 0; cc < 8; cc++)
    {
#pragma unroll
      for (int d = 0; d < 4; d++)
        isum[cc][d] = MAGIC;
    }
#pragma unroll
    for (int kw = 0; kw < KW; kw++)
    {
      const int4 vv = *reinterpret_cast<const int4*>(&Vb[kw * HEAD + 4 * lane]);
      const int4 p0 = *reinterpret_cast<const int4*>(&Pb[kw * P8S + 8 * w]);
      const int4 p1 = *reinterpret_cast<const int4*>(&Pb[kw * P8S + 8 * w + 4]);
      const int p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
      const int v4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
      for (int cc = 0; cc < 8; cc++)
      {
#pragma unroll
        for (int d = 0; d < 4; d++)
          isum[cc][d] = dp4a_us(p[cc], v4[d], isum[cc][d]);
      }
    }
  };
  // the P.V epilogue of one block: the shipped fma per output
  auto pv_epilogue = [&](const int *Vb, const float *Pb_s, const int (&isum)[8][4]) {
    const float4 ps0 = *reinterpret_cast<const float4*>(&Pb_s[8 * w]);
    const float4 ps1 = *reinterpret_cast<const float4*>(&Pb_s[8 * w + 4]);
    const float4 vs = *reinterpret_cast<const float4*>(&Vb[KW * HEAD + 4 * lane]);
    const float ps[8] = {ps0.x, ps0.y, ps0.z, ps0.w, ps1.x, ps1.y, ps1.z, ps1.w};
    const float vsa[4] = {vs.x, vs.y, vs.z, vs.w};
#pragma unroll
    for (int cc = 0; cc < 8; cc++)
    {
#pragma unroll
      for (int d = 0; d < 4; d++)
        acc[cc][d] = fmaf(ps[cc] * vsa[d], __int_as_float(isum[cc][d]) - MAGIC_F, acc[cc][d]);
    }
  };

  // prologue: K(0) in buffer 0, K(1) and V(0) in registers; block -1's P and scales zero (P buffer 1, P scale buffer 1,
  // the v scales of buffer 1): its P.V adds fma(0, 0, acc) = acc
  load_K(0);
  store_K(K8b);
  if (BK < nk)
    load_K(BK);
  load_V(0);
  for (int i = tid; i < PTW; i += THREADS)
    P8w[PTW + i] = 0;
  if (tid < BQ)
    Pssb[BQ + tid] = 0.0f;
  if (tid < HEAD)
    V8b[VTW + KW * HEAD + tid] = 0;
  __syncthreads();

  auto step = [&](const int k0, const int it, auto tail_tag) {
    constexpr bool kTail = decltype(tail_tag)::value;
    const int cur = it & 1, prv = cur ^ 1;
    // stage: the next K tile into the other K buffer (its last reader, the score loop of block i - 1, finished
    // before the previous barrier), this block's v tile into its buffer (last read by block i - 2's P.V, before the
    // previous barrier); the tiles after them into registers
    if constexpr (!kTail)
    {
      store_K(K8b + prv * KTW);
      if (k0 + 2 * BK < nk)
        load_K(k0 + 2 * BK);
    }
    store_V(V8b + cur * VTW);
    if constexpr (!kTail)
      load_V(k0 + BK);
    const int *K8s = K8b + cur * KTW;
    const float f = qs * Ksh[k0 / BLKK];

    int so[4], st[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
      so[j] = MAGIC;
      st[j] = MAGIC;
    }
#pragma unroll QK_UNROLL
    for (int kc = 0; kc < HEAD / 16; kc++)
    {
      const int4 qa = *reinterpret_cast<const int4*>(&Q8s[jc * QKS + kc * 4]);
      const int4 qb = *reinterpret_cast<const int4*>(&Q8s[jo * QKS + kc * 4]);
#pragma unroll
      for (int j = 0; j < 4; j++)
      {
        const int4 kk = *reinterpret_cast<const int4*>(&K8s[(8 * j + 4 * hh + g) * QKS + kc * 4]);
        so[j] = __dp4a(qa.x, kk.x, so[j]);
        so[j] = __dp4a(qa.y, kk.y, so[j]);
        so[j] = __dp4a(qa.z, kk.z, so[j]);
        so[j] = __dp4a(qa.w, kk.w, so[j]);
        st[j] = __dp4a(qb.x, kk.x, st[j]);
        st[j] = __dp4a(qb.y, kk.y, st[j]);
        st[j] = __dp4a(qb.z, kk.z, st[j]);
        st[j] = __dp4a(qb.w, kk.w, st[j]);
      }
    }

    // block i - 1's integer P.V sums (its P and v tiles were complete at the previous barrier)
    int isum[8][4];
    pv_sums(V8b + prv * VTW, P8w + prv * PTW, isum);

    // block i's softmax: the shipped operations
    float xo[4], xt[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
      xo[j] = (__int_as_float(so[j]) - MAGIC_F) * f;
      xt[j] = (__int_as_float(st[j]) - MAGIC_F) * f;
    }
    if constexpr (kTail)
    {
#pragma unroll
      for (int j = 0; j < 4; j++)
      {
        if (k0 + 8 * j + 4 * hh + g >= nk)
        {
          xo[j] = -FLT_MAX;
          xt[j] = -FLT_MAX;
        }
      }
    }
    float bo = fmaxf(fmaxf(xo[0], xo[1]), fmaxf(xo[2], xo[3]));
    float bt = fmaxf(fmaxf(xt[0], xt[1]), fmaxf(xt[2], xt[3]));
    bo = fmaxf(bo, __shfl_xor_sync(0xffffffff, bo, 8));
    bt = fmaxf(bt, __shfl_xor_sync(0xffffffff, bt, 8));
    bo = fmaxf(bo, __shfl_xor_sync(0xffffffff, bo, 16));
    bt = fmaxf(bt, __shfl_xor_sync(0xffffffff, bt, 16));
    {
      const float bo_p = __shfl_xor_sync(0xffffffff, bt, 1);
      const float bt_p = __shfl_xor_sync(0xffffffff, bo, 1);
      bo = fmaxf(bo, bo_p);
      bt = fmaxf(bt, bt_p);
    }
    const float mo = fmaxf(m_own, bo), mt = fmaxf(m_oth, bt);
    const float co = exp2f(m_own - mo), ct = exp2f(m_oth - mt);
    m_own = mo;
    m_oth = mt;
    {
      unsigned char *P8b = reinterpret_cast<unsigned char*>(P8w + cur * PTW);
      float so_sum = 0.0f, st_sum = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; j++)
      {
        const float eo = exp2f(xo[j] - bo), et = exp2f(xt[j] - bt);
        so_sum += eo;
        st_sum += et;
        const int row = 2 * j + hh;
        P8b[4 * (row * P8S + jc) + g] = (unsigned char)__float_as_uint(fmaf(255.0f, eo, MAGIC_F));
        P8b[4 * (row * P8S + jo) + g] = (unsigned char)__float_as_uint(fmaf(255.0f, et, MAGIC_F));
      }
      const float po = exp2f(bo - mo), pt = exp2f(bt - mt);
      l_own = fmaf(po, so_sum, l_own * co);
      l_oth = fmaf(pt, st_sum, l_oth * ct);
      if (g == 0)
        Pssb[cur * BQ + jc] = po * (1.0f / 255.0f);
    }

    // block i - 1's P.V epilogue, then block i's rescale: the accumulators' shipped order
    pv_epilogue(V8b + prv * VTW, Pssb + prv * BQ, isum);
    if (!__all_sync(0xffffffff, co == 1.0f))
    {
#pragma unroll
      for (int cc = 0; cc < 8; cc++)
      {
        const float cf = __shfl_sync(0xffffffff, co, cc);
#pragma unroll
        for (int d = 0; d < 4; d++)
          acc[cc][d] *= cf;
      }
    }
    __syncthreads();    // block i's P and v tile and block i + 1's K tile complete; every warp done with block i - 1
  };

  int k0 = 0, it = 0;
  for (; k0 + BK < nk; k0 += BK, it++)
    step(k0, it, std::false_type());
  step(k0, it, std::true_type());
  {
    // the last block's P.V
    const int cur = it & 1;
    int isum[8][4];
    pv_sums(V8b + cur * VTW, P8w + cur * PTW, isum);
    pv_epilogue(V8b + cur * VTW, Pssb + cur * BQ, isum);
  }

  float l = l_own + __shfl_xor_sync(0xffffffff, l_oth, 1);
  l += __shfl_xor_sync(0xffffffff, l, 8);
  l += __shfl_xor_sync(0xffffffff, l, 16);
  const float4 mean = *reinterpret_cast<const float4*>(V_mean + bh * HEAD + 4 * lane);
  TO *Oh = O + b * stride_bz_o + h * stride_h_o + 4 * lane;
#pragma unroll
  for (int cc = 0; cc < 8; cc++)
  {
    const float lc = __shfl_sync(0xffffffff, l, cc);
    const int row = q0 + 8 * w + cc;
    if (row < nq)
    {
      const float inv = 1.0f / lc;
      store4(Oh + row * stride_seq_o, make_float4(fmaf(acc[cc][0], inv, mean.x), fmaf(acc[cc][1], inv, mean.y),
                                                  fmaf(acc[cc][2], inv, mean.z), fmaf(acc[cc][3], inv, mean.w)));
    }
  }
}
#endif

#if SC3D_P61_K64
// Round P: the int8 P.V kernel at 64 keys an iteration, as the shipped kernel's two 32-key blocks A and B in order -
// each still its own P / v scale block, its own softmax step (maxima, codes, denominators, rescale) and its own P.V -
// with one score loop for both: a lane's two q rows are read from shared memory once a 16-channel chunk for the 8
// keys of A and B (the shipped loop reads them once for 4 keys; the score loop is bound by its shared loads on the
// GTX 10xx: without its q loads it is 17 % faster per clock). Each lane still owns the shipped key slots
// (8 j + 4 hh + g of each block), so every denominator, maximum, code and output is the shipped operation on the
// shipped values in the shipped order: the shipped kernel's bits. Two barriers per 64 keys; the next K tile and this
// iteration's v tiles are loaded at the iteration's start and stored after its first barrier (no staging registers
// live in the P.V).
template <typename TO>
__global__ void __launch_bounds__(THREADS, 2) qk_int8_attn_k64_kernel(
    const int *__restrict__ Q8, const float *__restrict__ Q_scale,
    const int *__restrict__ K8, const float *__restrict__ K_scale,
    const int *__restrict__ V8, const float *__restrict__ V_scale, const float *__restrict__ V_mean,
    TO *__restrict__ O,
    const int nq, const int nk, const int nq_pad, const int nk_pad,
    const int64_t stride_bz_o, const int64_t stride_seq_o, const int64_t stride_h_o,
    const float sm_scale_log2)
{
  constexpr int BK2 = 2 * BK;                   // keys an iteration (BLKK: one k scale)
  static_assert(BK2 == BLKK, "one k scale an iteration");
  constexpr int VTW = KW * HEAD + HEAD;         // ints a block's v tile: the words, then the scales
  constexpr int PTW = KW * P8S;                 // ints a block's P
  constexpr int SMI = BQ * QKS + BK2 * QKS + 2 * VTW + 2 * PTW + 2 * BQ;
  static_assert(SMI * 4 <= 48 * 1024, "static shared memory: 48 KB a block");
  __shared__ __align__(16) int sm[SMI];
  int *Q8s = sm;
  int *K8s = Q8s + BQ * QKS;
  int *V8b = K8s + BK2 * QKS;
  int *P8w = V8b + 2 * VTW;
  float *Pssb = reinterpret_cast<float*>(P8w + 2 * PTW);

  const int tid = threadIdx.x;
  const int lane = tid % 32, w = tid / 32;
  const int q0 = blockIdx.x * BQ, h = blockIdx.y, b = blockIdx.z;
  const int nh = gridDim.y;
  const int64_t bh = (int64_t)b * nh + h;
  const int c = lane % 8, g = lane / 8, hh = c % 2;
  const int jc = 8 * w + c, jo = jc ^ 1;

  const int *Q8h = Q8 + (bh * nq_pad + q0) * (HEAD / 4);
  const int *K8h = K8 + bh * nk_pad * (HEAD / 4);
  const float *Ksh = K_scale + bh * (nk_pad / BLKK);
  for (int i = tid; i < BQ * (HEAD / 16); i += THREADS)
  {
    const int row = i / (HEAD / 16), seg = i % (HEAD / 16);
    *reinterpret_cast<int4*>(&Q8s[row * QKS + seg * 4]) = *reinterpret_cast<const int4*>(&Q8h[row * (HEAD / 4) + seg * 4]);
  }
  const float qs = Q_scale[bh * (nq_pad / WARPQ) + (q0 + 8 * w) / WARPQ] * sm_scale_log2;

  float acc[8][4];
#pragma unroll
  for (int cc = 0; cc < 8; cc++)
  {
#pragma unroll
    for (int d = 0; d < 4; d++)
      acc[cc][d] = 0.0f;
  }
  float m_own = -FLT_MAX / 2.0f, m_oth = -FLT_MAX / 2.0f;
  float l_own = 0.0f, l_oth = 0.0f;

  // staging: K rows k0 + tid / 8 + 32 i (i = 0, 1), bytes 16 (tid % 8); v words piece tid + 256 i of the iteration's
  // two blocks (block i), their scales by the first 64 threads
  int4 kreg[2], v8reg[2];
  float4 vsreg = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int *V8h = V8 + bh * (nk_pad / 4) * HEAD;
  const float *Vsh = V_scale + bh * (nk_pad / BK) * HEAD;
  auto load_K = [&](const int k0) {
#pragma unroll
    for (int i = 0; i < 2; i++)
      kreg[i] = *reinterpret_cast<const int4*>(&K8h[(int64_t)(k0 + tid / 8 + 32 * i) * (HEAD / 4) + (tid % 8) * 4]);
  };
  auto store_K = [&]() {
#pragma unroll
    for (int i = 0; i < 2; i++)
      *reinterpret_cast<int4*>(&K8s[(tid / 8 + 32 * i) * QKS + (tid % 8) * 4]) = kreg[i];
  };
  auto load_V = [&](const int k0) {       // the blocks k0 / 32 and k0 / 32 + 1 (both inside the padded layout)
    const int blk = k0 / BK;
#pragma unroll
    for (int i = 0; i < 2; i++)
      v8reg[i] = *reinterpret_cast<const int4*>(&V8h[(int64_t)(blk + i) * KW * HEAD + 4 * tid]);
    if (tid < 2 * (HEAD / 4))
      vsreg = *reinterpret_cast<const float4*>(&Vsh[(int64_t)(blk + tid / 32) * HEAD + 4 * (tid % 32)]);
  };
  auto store_V = [&]() {
#pragma unroll
    for (int i = 0; i < 2; i++)
      *reinterpret_cast<int4*>(&V8b[i * VTW + 4 * tid]) = v8reg[i];
    if (tid < 2 * (HEAD / 4))
      *reinterpret_cast<float4*>(&V8b[(tid / 32) * VTW + KW * HEAD + 4 * (tid % 32)]) = vsreg;
  };

  // one 32-key block's softmax step on its exact scores (the shipped kernel's code): maxima, rescale factor, P codes
  // and scales into buffer s, the denominators; returns co (this lane's own query's factor)
  auto softmax = [&](int (&so)[4], int (&st)[4], const int kb, const float f, const int s) -> float {
    float xo[4], xt[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
      xo[j] = (__int_as_float(so[j]) - MAGIC_F) * f;
      xt[j] = (__int_as_float(st[j]) - MAGIC_F) * f;
    }
    if (kb + BK > nk)
    {
#pragma unroll
      for (int j = 0; j < 4; j++)
      {
        if (kb + 8 * j + 4 * hh + g >= nk)
        {
          xo[j] = -FLT_MAX;
          xt[j] = -FLT_MAX;
        }
      }
    }
    float bo = fmaxf(fmaxf(xo[0], xo[1]), fmaxf(xo[2], xo[3]));
    float bt = fmaxf(fmaxf(xt[0], xt[1]), fmaxf(xt[2], xt[3]));
    bo = fmaxf(bo, __shfl_xor_sync(0xffffffff, bo, 8));
    bt = fmaxf(bt, __shfl_xor_sync(0xffffffff, bt, 8));
    bo = fmaxf(bo, __shfl_xor_sync(0xffffffff, bo, 16));
    bt = fmaxf(bt, __shfl_xor_sync(0xffffffff, bt, 16));
    {
      const float bo_p = __shfl_xor_sync(0xffffffff, bt, 1);
      const float bt_p = __shfl_xor_sync(0xffffffff, bo, 1);
      bo = fmaxf(bo, bo_p);
      bt = fmaxf(bt, bt_p);
    }
    const float mo = fmaxf(m_own, bo), mt = fmaxf(m_oth, bt);
    const float co = exp2f(m_own - mo), ct = exp2f(m_oth - mt);
    m_own = mo;
    m_oth = mt;
    unsigned char *P8b = reinterpret_cast<unsigned char*>(P8w + s * PTW);
    float so_sum = 0.0f, st_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
      const float eo = exp2f(xo[j] - bo), et = exp2f(xt[j] - bt);
      so_sum += eo;
      st_sum += et;
      const int row = 2 * j + hh;
      P8b[4 * (row * P8S + jc) + g] = (unsigned char)__float_as_uint(fmaf(255.0f, eo, MAGIC_F));
      P8b[4 * (row * P8S + jo) + g] = (unsigned char)__float_as_uint(fmaf(255.0f, et, MAGIC_F));
    }
    const float po = exp2f(bo - mo), pt = exp2f(bt - mt);
    l_own = fmaf(po, so_sum, l_own * co);
    l_oth = fmaf(pt, st_sum, l_oth * ct);
    if (g == 0)
      Pssb[s * BQ + jc] = po * (1.0f / 255.0f);
    return co;
  };
  auto rescale = [&](const float co) {
    if (!__all_sync(0xffffffff, co == 1.0f))
    {
#pragma unroll
      for (int cc = 0; cc < 8; cc++)
      {
        const float cf = __shfl_sync(0xffffffff, co, cc);
#pragma unroll
        for (int d = 0; d < 4; d++)
          acc[cc][d] *= cf;
      }
    }
  };

  load_K(0);
  store_K();
  __syncthreads();

  for (int k0 = 0; k0 < nk; k0 += BK2)
  {
    const bool has_b = k0 + BK < nk;          // block B exists (warp-uniform)
    if (k0 + BK2 < nk)
      load_K(k0 + BK2);
    load_V(k0);
    const float f = qs * Ksh[k0 / BLKK];

    // the exact scores of blocks A (so, st) and B (su, sv): per chunk the two q rows once, 4 + 4 key rows
    int so[4], st[4], su[4], sv[4];
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
      so[j] = MAGIC;
      st[j] = MAGIC;
      su[j] = MAGIC;
      sv[j] = MAGIC;
    }
#pragma unroll QK_UNROLL
    for (int kc = 0; kc < HEAD / 16; kc++)
    {
      const int4 qa = *reinterpret_cast<const int4*>(&Q8s[jc * QKS + kc * 4]);
      const int4 qb = *reinterpret_cast<const int4*>(&Q8s[jo * QKS + kc * 4]);
#pragma unroll
      for (int j = 0; j < 4; j++)
      {
        const int4 ka = *reinterpret_cast<const int4*>(&K8s[(8 * j + 4 * hh + g) * QKS + kc * 4]);
        so[j] = __dp4a(qa.x, ka.x, so[j]);
        so[j] = __dp4a(qa.y, ka.y, so[j]);
        so[j] = __dp4a(qa.z, ka.z, so[j]);
        so[j] = __dp4a(qa.w, ka.w, so[j]);
        st[j] = __dp4a(qb.x, ka.x, st[j]);
        st[j] = __dp4a(qb.y, ka.y, st[j]);
        st[j] = __dp4a(qb.z, ka.z, st[j]);
        st[j] = __dp4a(qb.w, ka.w, st[j]);
        const int4 kb = *reinterpret_cast<const int4*>(&K8s[(BK + 8 * j + 4 * hh + g) * QKS + kc * 4]);
        su[j] = __dp4a(qa.x, kb.x, su[j]);
        su[j] = __dp4a(qa.y, kb.y, su[j]);
        su[j] = __dp4a(qa.z, kb.z, su[j]);
        su[j] = __dp4a(qa.w, kb.w, su[j]);
        sv[j] = __dp4a(qb.x, kb.x, sv[j]);
        sv[j] = __dp4a(qb.y, kb.y, sv[j]);
        sv[j] = __dp4a(qb.z, kb.z, sv[j]);
        sv[j] = __dp4a(qb.w, kb.w, sv[j]);
      }
    }

    // block A's softmax step and rescale, then block B's softmax step (its rescale waits for A's P.V)
    __syncwarp();       // the warp's P.V reads of its P (previous iteration) are done before it is rewritten
    const float co_a = softmax(so, st, k0, f, 0);
    float co_b = 1.0f;
    if (has_b)
      co_b = softmax(su, sv, k0 + BK, f, 1);
    rescale(co_a);

    __syncthreads();    // every warp done with K8s and the previous v tiles; every P complete
    store_V();
    if (k0 + BK2 < nk)
      store_K();
    __syncthreads();

    // the P.V of A, then B's rescale and P.V
#pragma unroll 1
    for (int s = 0; s < (has_b ? 2 : 1); s++)
    {
      if (s == 1)
        rescale(co_b);
      const int *Vb = V8b + s * VTW;
      const int *Pb = P8w + s * PTW;
      int isum[8][4];
#pragma unroll
      for (int cc = 0; cc < 8; cc++)
      {
#pragma unroll
        for (int d = 0; d < 4; d++)
          isum[cc][d] = MAGIC;
      }
#pragma unroll PV_UNROLL
      for (int kw = 0; kw < KW; kw++)
      {
        const int4 vv = *reinterpret_cast<const int4*>(&Vb[kw * HEAD + 4 * lane]);
        const int4 p0 = *reinterpret_cast<const int4*>(&Pb[kw * P8S + 8 * w]);
        const int4 p1 = *reinterpret_cast<const int4*>(&Pb[kw * P8S + 8 * w + 4]);
        const int p[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
        const int v4[4] = {vv.x, vv.y, vv.z, vv.w};
#pragma unroll
        for (int cc = 0; cc < 8; cc++)
        {
#pragma unroll
          for (int d = 0; d < 4; d++)
            isum[cc][d] = dp4a_us(p[cc], v4[d], isum[cc][d]);
        }
      }
      const float4 ps0 = *reinterpret_cast<const float4*>(&Pssb[s * BQ + 8 * w]);
      const float4 ps1 = *reinterpret_cast<const float4*>(&Pssb[s * BQ + 8 * w + 4]);
      const float4 vs = *reinterpret_cast<const float4*>(&Vb[KW * HEAD + 4 * lane]);
      const float ps[8] = {ps0.x, ps0.y, ps0.z, ps0.w, ps1.x, ps1.y, ps1.z, ps1.w};
      const float vsa[4] = {vs.x, vs.y, vs.z, vs.w};
#pragma unroll
      for (int cc = 0; cc < 8; cc++)
      {
#pragma unroll
        for (int d = 0; d < 4; d++)
          acc[cc][d] = fmaf(ps[cc] * vsa[d], __int_as_float(isum[cc][d]) - MAGIC_F, acc[cc][d]);
      }
    }
  }

  float l = l_own + __shfl_xor_sync(0xffffffff, l_oth, 1);
  l += __shfl_xor_sync(0xffffffff, l, 8);
  l += __shfl_xor_sync(0xffffffff, l, 16);
  const float4 mean = *reinterpret_cast<const float4*>(V_mean + bh * HEAD + 4 * lane);
  TO *Oh = O + b * stride_bz_o + h * stride_h_o + 4 * lane;
#pragma unroll
  for (int cc = 0; cc < 8; cc++)
  {
    const float lc = __shfl_sync(0xffffffff, l, cc);
    const int row = q0 + 8 * w + cc;
    if (row < nq)
    {
      const float inv = 1.0f / lc;
      store4(Oh + row * stride_seq_o, make_float4(fmaf(acc[cc][0], inv, mean.x), fmaf(acc[cc][1], inv, mean.y),
                                                  fmaf(acc[cc][2], inv, mean.z), fmaf(acc[cc][3], inv, mean.w)));
    }
  }
}
#endif

// ---- host side

inline int pad_tokens(const int n) { return (n + TPAD - 1) / TPAD * TPAD; }

struct Layout
{
  int b, n, h;
  int64_t stride_bz, stride_seq, stride_h;
};

inline Layout layout_of(const torch::Tensor &x, const int64_t tensor_layout)
{
  TORCH_CHECK(tensor_layout == 0 || tensor_layout == 1, "tensor_layout must be 0 (NHD) or 1 (HND)");
  Layout l;
  l.b = x.size(0);
  l.stride_bz = x.stride(0);
  if (tensor_layout == 0)
  {
    l.n = x.size(1); l.h = x.size(2); l.stride_seq = x.stride(1); l.stride_h = x.stride(2);
  }
  else
  {
    l.h = x.size(1); l.n = x.size(2); l.stride_h = x.stride(1); l.stride_seq = x.stride(2);
  }
  return l;
}

// fp32 / fp16 / bf16, head 128, rows aligned for 4-element loads
inline void check_input(const torch::Tensor &x, const char *name)
{
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 4 && x.size(3) == HEAD, name, " must be [B, N, H, 128] or [B, H, N, 128]");
  TORCH_CHECK(x.stride(3) == 1, name, " must be contiguous in its last dimension");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half ||
              x.scalar_type() == at::ScalarType::BFloat16, name, " must be float32, float16 or bfloat16");
  const int64_t align = x.scalar_type() == at::ScalarType::Float ? 16 : 8;
  const int64_t esize = x.element_size();
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % align == 0 && (x.stride(0) * esize) % align == 0 &&
              (x.stride(1) * esize) % align == 0 && (x.stride(2) * esize) % align == 0,
              name, " must have rows aligned to ", align, " bytes");
}

template <typename F>
inline void dispatch_float(const at::ScalarType t, F &&f)
{
  if (t == at::ScalarType::Float)
    f(static_cast<float*>(nullptr));
#if SC3D_P61_F32_ONLY
  else
    TORCH_CHECK(false, "this developer build takes float32 alone");
#else
  else if (t == at::ScalarType::Half)
    f(static_cast<half*>(nullptr));
  else if (t == at::ScalarType::BFloat16)
    f(static_cast<nv_bfloat16*>(nullptr));
  else
    TORCH_CHECK(false, "float32, float16 or bfloat16 expected");
#endif
}

// the mean of x over its tokens per head and channel into mean [B, H, 128] fp32
void channel_mean(const torch::Tensor &x, const Layout &l, torch::Tensor &mean, cudaStream_t stream)
{
  const int nblk = (l.n + BLKK - 1) / BLKK;
  torch::Tensor part = torch::empty({l.b, l.h, nblk, HEAD}, x.options().dtype(torch::kFloat32));
  dispatch_float(x.scalar_type(), [&](auto *tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    block_sum_kernel<T><<<dim3(nblk, l.h, l.b), HEAD, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()), part.data_ptr<float>(), l.n, l.stride_bz, l.stride_seq, l.stride_h);
  });
  mean_kernel<<<l.b * l.h, HEAD, 0, stream>>>(part.data_ptr<float>(), mean.data_ptr<float>(), nblk, l.n);
}

template <bool kPerBlock>
void quant_int8(const torch::Tensor &x, const Layout &l, const float *mean, torch::Tensor &out, torch::Tensor &scale, cudaStream_t stream)
{
  const int npad = pad_tokens(l.n);
  dispatch_float(x.scalar_type(), [&](auto *tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    quant_int8_kernel<T, kPerBlock><<<dim3(npad / 64, l.h, l.b), 128, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()), mean, out.data_ptr<int8_t>(), scale.data_ptr<float>(), l.n, npad,
      l.stride_bz, l.stride_seq, l.stride_h);
  });
}

void quant_v(const torch::Tensor &v, const Layout &l, const float *mean, torch::Tensor &out, torch::Tensor &scale, cudaStream_t stream)
{
  const int npad = pad_tokens(l.n);
  dispatch_float(v.scalar_type(), [&](auto *tag) {
    using T = std::remove_pointer_t<decltype(tag)>;
    quant_v_kernel<T><<<dim3(npad / BK, l.h, l.b), THREADS, 0, stream>>>(
      reinterpret_cast<const T*>(v.data_ptr()), mean, out.data_ptr<int>(), scale.data_ptr<float>(), l.n,
      l.stride_bz, l.stride_seq, l.stride_h);
  });
}

// the attention kernel on quantised q / k (and v for the int8 P.V); value: the fp32 P.V's v (any float type), else
// unused (an empty tensor)
void attention(const torch::Tensor &q_int8, const torch::Tensor &q_scale, const torch::Tensor &k_int8,
               const torch::Tensor &k_scale, const torch::Tensor &value, const Layout &lv,
               const torch::Tensor *v_int8, const torch::Tensor *v_scale, const torch::Tensor *v_mean,
               torch::Tensor &output, const Layout &lo, const int nk, const double sm_scale, cudaStream_t stream)
{
  const int nq = lo.n, nq_pad = q_int8.size(2), nk_pad = k_int8.size(2);
  const float sm_scale_log2 = (float)sm_scale * LOG2E;
  const bool int8pv = v_int8 != nullptr;
  dispatch_float(output.scalar_type(), [&](auto *otag) {
    using TO = std::remove_pointer_t<decltype(otag)>;
    TO *o = reinterpret_cast<TO*>(output.data_ptr());
    if (int8pv)
    {
#if SC3D_P61_K64
      qk_int8_attn_k64_kernel<TO><<<dim3(nq_pad / BQ, lo.h, lo.b), THREADS, 0, stream>>>(
        reinterpret_cast<const int*>(q_int8.data_ptr()), q_scale.data_ptr<float>(),
        reinterpret_cast<const int*>(k_int8.data_ptr()), k_scale.data_ptr<float>(),
        v_int8->data_ptr<int>(), v_scale->data_ptr<float>(), v_mean->data_ptr<float>(), o,
        nq, nk, nq_pad, nk_pad, lo.stride_bz, lo.stride_seq, lo.stride_h, sm_scale_log2);
#elif SC3D_P61_PIPE
      qk_int8_attn_pipe_kernel<TO><<<dim3(nq_pad / BQ, lo.h, lo.b), THREADS, 0, stream>>>(
        reinterpret_cast<const int*>(q_int8.data_ptr()), q_scale.data_ptr<float>(),
        reinterpret_cast<const int*>(k_int8.data_ptr()), k_scale.data_ptr<float>(),
        v_int8->data_ptr<int>(), v_scale->data_ptr<float>(), v_mean->data_ptr<float>(), o,
        nq, nk, nq_pad, nk_pad, lo.stride_bz, lo.stride_seq, lo.stride_h, sm_scale_log2);
#else
      qk_int8_attn_kernel<true, float, TO><<<dim3(nq_pad / Cfg<true>::Q, lo.h, lo.b), Cfg<true>::T, 0, stream>>>(
        reinterpret_cast<const int*>(q_int8.data_ptr()), q_scale.data_ptr<float>(),
        reinterpret_cast<const int*>(k_int8.data_ptr()), k_scale.data_ptr<float>(),
        nullptr, v_int8->data_ptr<int>(), v_scale->data_ptr<float>(), v_mean->data_ptr<float>(), o,
        nq, nk, nq_pad, nk_pad, 0, 0, 0, lo.stride_bz, lo.stride_seq, lo.stride_h, sm_scale_log2);
#endif
    }
    else
    {
      dispatch_float(value.scalar_type(), [&](auto *vtag) {
        using TV = std::remove_pointer_t<decltype(vtag)>;
        qk_int8_attn_kernel<false, TV, TO><<<dim3(nq_pad / Cfg<false>::Q, lo.h, lo.b), Cfg<false>::T, 0, stream>>>(
          reinterpret_cast<const int*>(q_int8.data_ptr()), q_scale.data_ptr<float>(),
          reinterpret_cast<const int*>(k_int8.data_ptr()), k_scale.data_ptr<float>(),
          reinterpret_cast<const TV*>(value.data_ptr()), nullptr, nullptr, nullptr, o,
          nq, nk, nq_pad, nk_pad, lv.stride_bz, lv.stride_seq, lv.stride_h, lo.stride_bz, lo.stride_seq, lo.stride_h,
          sm_scale_log2);
      });
    }
  });
}

void check_quantised(const torch::Tensor &q_int8, const torch::Tensor &q_scale, const torch::Tensor &k_int8,
                     const torch::Tensor &k_scale, const int b, const int h, const int nq, const int nk)
{
  CHECK_CUDA(q_int8); CHECK_CUDA(q_scale); CHECK_CUDA(k_int8); CHECK_CUDA(k_scale);
  CHECK_CONTIGUOUS(q_int8); CHECK_CONTIGUOUS(q_scale); CHECK_CONTIGUOUS(k_int8); CHECK_CONTIGUOUS(k_scale);
  CHECK_DTYPE(q_int8, torch::kInt8); CHECK_DTYPE(k_int8, torch::kInt8);
  CHECK_DTYPE(q_scale, torch::kFloat32); CHECK_DTYPE(k_scale, torch::kFloat32);
  const int nq_pad = pad_tokens(nq), nk_pad = pad_tokens(nk);
  CHECK_SHAPE(q_int8, b, h, nq_pad, HEAD);
  CHECK_SHAPE(q_scale, b, h, nq_pad / WARPQ);
  CHECK_SHAPE(k_int8, b, h, nk_pad, HEAD);
  CHECK_SHAPE(k_scale, b, h, nk_pad / BLKK);
}

} // namespace sc3d_p61

using namespace sc3d_p61;

void quant_qk_int8_pascal(torch::Tensor query, torch::Tensor key, torch::Tensor query_int8, torch::Tensor query_scale,
                          torch::Tensor key_int8, torch::Tensor key_scale, torch::Tensor key_mean,
                          int64_t tensor_layout, int64_t smooth_k)
{
  check_input(query, "query");
  check_input(key, "key");
  const Layout lq = layout_of(query, tensor_layout), lk = layout_of(key, tensor_layout);
  TORCH_CHECK(lq.b == lk.b && lq.h == lk.h, "query and key must have the same batch and heads");
  TORCH_CHECK(lq.n > 0 && lk.n > 0, "empty sequence");
  check_quantised(query_int8, query_scale, key_int8, key_scale, lq.b, lq.h, lq.n, lk.n);
  CHECK_CUDA(key_mean); CHECK_CONTIGUOUS(key_mean); CHECK_DTYPE(key_mean, torch::kFloat32);
  CHECK_SHAPE(key_mean, lk.b, lk.h, HEAD);
  const at::cuda::CUDAGuard guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (smooth_k)
    channel_mean(key, lk, key_mean, stream);
  else
    key_mean.zero_();
  quant_int8<false>(query, lq, nullptr, query_int8, query_scale, stream);
  quant_int8<true>(key, lk, smooth_k ? key_mean.data_ptr<float>() : nullptr, key_int8, key_scale, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void quant_v_int8_pascal(torch::Tensor value, torch::Tensor value_int8, torch::Tensor value_scale, torch::Tensor value_mean,
                         int64_t tensor_layout, int64_t smooth_v)
{
  check_input(value, "value");
  const Layout lv = layout_of(value, tensor_layout);
  TORCH_CHECK(lv.n > 0, "empty sequence");
  const int npad = pad_tokens(lv.n);
  CHECK_CUDA(value_int8); CHECK_CUDA(value_scale); CHECK_CUDA(value_mean);
  CHECK_CONTIGUOUS(value_int8); CHECK_CONTIGUOUS(value_scale); CHECK_CONTIGUOUS(value_mean);
  CHECK_DTYPE(value_int8, torch::kInt32); CHECK_DTYPE(value_scale, torch::kFloat32); CHECK_DTYPE(value_mean, torch::kFloat32);
  CHECK_SHAPE(value_int8, lv.b, lv.h, npad / 4, HEAD);
  CHECK_SHAPE(value_scale, lv.b, lv.h, npad / BK, HEAD);
  CHECK_SHAPE(value_mean, lv.b, lv.h, HEAD);
  const at::cuda::CUDAGuard guard(value.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (smooth_v)
    channel_mean(value, lv, value_mean, stream);
  else
    value_mean.zero_();
  quant_v(value, lv, value_mean.data_ptr<float>(), value_int8, value_scale, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qk_int8_sv_f32_attn_pascal(torch::Tensor query_int8, torch::Tensor key_int8, torch::Tensor value, torch::Tensor output,
                                torch::Tensor query_scale, torch::Tensor key_scale, int64_t tensor_layout, double sm_scale)
{
  check_input(value, "value");
  check_input(output, "output");
  const Layout lv = layout_of(value, tensor_layout), lo = layout_of(output, tensor_layout);
  TORCH_CHECK(lv.b == lo.b && lv.h == lo.h && lo.n > 0 && lv.n > 0, "value and output: the same batch and heads");
  check_quantised(query_int8, query_scale, key_int8, key_scale, lo.b, lo.h, lo.n, lv.n);
  const at::cuda::CUDAGuard guard(output.device());
  attention(query_int8, query_scale, key_int8, key_scale, value, lv, nullptr, nullptr, nullptr, output, lo, lv.n,
            sm_scale, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void qk_int8_sv_i8_attn_pascal(torch::Tensor query_int8, torch::Tensor key_int8, torch::Tensor value_int8, torch::Tensor output,
                               torch::Tensor query_scale, torch::Tensor key_scale, torch::Tensor value_scale,
                               torch::Tensor value_mean, int64_t kv_len, int64_t tensor_layout, double sm_scale)
{
  check_input(output, "output");
  const Layout lo = layout_of(output, tensor_layout);
  TORCH_CHECK(lo.n > 0 && kv_len > 0, "empty sequence");
  const int nk = (int)kv_len, nk_pad = pad_tokens(nk);
  check_quantised(query_int8, query_scale, key_int8, key_scale, lo.b, lo.h, lo.n, nk);
  CHECK_CUDA(value_int8); CHECK_CUDA(value_scale); CHECK_CUDA(value_mean);
  CHECK_CONTIGUOUS(value_int8); CHECK_CONTIGUOUS(value_scale); CHECK_CONTIGUOUS(value_mean);
  CHECK_DTYPE(value_int8, torch::kInt32); CHECK_DTYPE(value_scale, torch::kFloat32); CHECK_DTYPE(value_mean, torch::kFloat32);
  CHECK_SHAPE(value_int8, lo.b, lo.h, nk_pad / 4, HEAD);
  CHECK_SHAPE(value_scale, lo.b, lo.h, nk_pad / BK, HEAD);
  CHECK_SHAPE(value_mean, lo.b, lo.h, HEAD);
  const at::cuda::CUDAGuard guard(output.device());
  attention(query_int8, query_scale, key_int8, key_scale, torch::Tensor(), lo, &value_int8, &value_scale, &value_mean,
            output, lo, nk, sm_scale, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sageattn_pascal(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output,
                     int64_t tensor_layout, double sm_scale, int64_t pv_int8)
{
  check_input(query, "query");
  check_input(key, "key");
  check_input(value, "value");
  check_input(output, "output");
  const Layout lq = layout_of(query, tensor_layout), lk = layout_of(key, tensor_layout);
  const Layout lv = layout_of(value, tensor_layout), lo = layout_of(output, tensor_layout);
  TORCH_CHECK(lq.b == lk.b && lq.b == lv.b && lq.b == lo.b && lq.h == lk.h && lq.h == lv.h && lq.h == lo.h,
              "query, key, value and output must have the same batch and heads");
  TORCH_CHECK(lk.n == lv.n && lq.n == lo.n && lq.n > 0 && lk.n > 0, "the lengths of query / output and key / value differ or are 0");
  const at::cuda::CUDAGuard guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto f32 = query.options().dtype(torch::kFloat32);
  const int nq_pad = pad_tokens(lq.n), nk_pad = pad_tokens(lk.n);
  torch::Tensor q_int8 = torch::empty({lq.b, lq.h, nq_pad, HEAD}, query.options().dtype(torch::kInt8));
  torch::Tensor q_scale = torch::empty({lq.b, lq.h, nq_pad / WARPQ}, f32);
  torch::Tensor k_int8 = torch::empty({lk.b, lk.h, nk_pad, HEAD}, query.options().dtype(torch::kInt8));
  torch::Tensor k_scale = torch::empty({lk.b, lk.h, nk_pad / BLKK}, f32);
  torch::Tensor k_mean = torch::empty({lk.b, lk.h, HEAD}, f32);
  channel_mean(key, lk, k_mean, stream);
  quant_int8<false>(query, lq, nullptr, q_int8, q_scale, stream);
  quant_int8<true>(key, lk, k_mean.data_ptr<float>(), k_int8, k_scale, stream);
  if (pv_int8)
  {
    torch::Tensor v_int8 = torch::empty({lv.b, lv.h, nk_pad / 4, HEAD}, query.options().dtype(torch::kInt32));
    torch::Tensor v_scale = torch::empty({lv.b, lv.h, nk_pad / BK, HEAD}, f32);
    torch::Tensor v_mean = torch::empty({lv.b, lv.h, HEAD}, f32);
    channel_mean(value, lv, v_mean, stream);
    quant_v(value, lv, v_mean.data_ptr<float>(), v_int8, v_scale, stream);
    attention(q_int8, q_scale, k_int8, k_scale, torch::Tensor(), lo, &v_int8, &v_scale, &v_mean, output, lo, lk.n,
              sm_scale, stream);
  }
  else
  {
    attention(q_int8, q_scale, k_int8, k_scale, value, lv, nullptr, nullptr, nullptr, output, lo, lk.n, sm_scale, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
