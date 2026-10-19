// Round J (Spellcast3D): a standalone torch extension holding ONE SageAttention kernel - the one the flows call
// through sageattn_qk_int8_pv_fp16_cuda(..., qk_quant_gran="per_warp", pv_accum_dtype="fp16+fp32") on sm_8x:
// qk_int8_sv_f16_accum_f16_attn_inst_buf for head 128, per-warp q/k scales, not causal, no LSE, bf16 output -
// built as a variant (the knobs below, as -D flags) for the speed study in
// code/devtools/pixal3d-lowmem/round-j/sagecuda (Spellcast3D hub). The quantisation stays the wheel's.
//
//   module.attn(q_int8, k_int8, v_fp16, out_bf16, q_scale, k_scale, sm_scale) -> None   (NHD tensors, [B, N, H, 128])
//   module.info() -> dict: the knobs it was built with, threads, dynamic shared memory per block
//
// Knobs (all default to the shipped kernel):
//   SV_CTA_Q          query rows per block (128 shipped; 64 = four warps). WARP_Q stays 16 (one q scale per warp).
//   SV_MIN_BLOCKS     0: no __launch_bounds__ (shipped); n: __launch_bounds__(threads, n)
//   SV_I2F_MAGIC      0: S = __int2float_rz(int32) (shipped; I2F.RZ on sm_80 SASS, I2FP on sm_86);
//                     1: S = __int_as_float(s + 0x4B400000) - 12582912.0f, exact for |s| < 2^22 (|s| <= 128 * 128 *
//                        128 = 2^21 for int8 q, k and head 128), so the same float as the conversion, on the FMA pipe
//   SV_Q_REGS         0: q read from shared memory for every block of keys (shipped); 1: held in 16 registers
//   SV_PIPE           0: the shipped loop; 1: sv_kernel_pipe.cuh - the next tile's Q.K^T beside this tile's softmax;
//                     2: sv_kernel_pipe2.cuh - the same, main loop unrolled by hand (two tiles per trip, no copies);
//                     3: sv_kernel_pp.cuh - ping-pong of two warp groups (64 KB of shared memory, CTA_Q 128 only)
//   SV_SV_ORDER       0: the P.V step as shipped (all eight 16-column tiles' fp16 buffers at once); g = 1, 2, 4: the
//                     loops swapped, g column tiles at a time (the same chains: same values; fewer registers)
//   SV_BAR3           0: four block barriers a tile (shipped); 1: three (the next K tile issued after the barrier
//                     that makes V(i) visible, which already follows every warp's Q.K^T of tile i)
//   SV_PIPE_UNROLL    the pipelined main loop's #pragma unroll (2: two tiles per trip, no register copies; 1)

#include "../../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include "../../cp_async.cuh"
#include "../../mma.cuh"
#include "../../permuted_smem.cuh"
#include "../../math.cuh"
#include "../../dispatch_utils.h"

#include "../attn_utils.cuh"

#define PACK_SIZE_QK 16 // as if it is int8
#define PACK_SIZE_V 8   // fp16
#define PACK_SIZE_O 8   // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// fp16 tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 16

#ifndef SV_CTA_Q
#define SV_CTA_Q 128
#endif
#ifndef SV_MIN_BLOCKS
#define SV_MIN_BLOCKS 0
#endif
#ifndef SV_I2F_MAGIC
#define SV_I2F_MAGIC 0
#endif
#ifndef SV_Q_REGS
#define SV_Q_REGS 0
#endif
#ifndef SV_PIPE
#define SV_PIPE 0
#endif
#ifndef SV_SV_ORDER
#define SV_SV_ORDER 0
#endif
#ifndef SV_BAR3
#define SV_BAR3 0
#endif
#ifndef SV_PIPE_UNROLL
#define SV_PIPE_UNROLL 2
#endif

#if SV_MIN_BLOCKS > 0
#define SV_LAUNCH_BOUNDS __launch_bounds__(32 * (SV_CTA_Q / 16), SV_MIN_BLOCKS)
#else
#define SV_LAUNCH_BOUNDS
#endif

#if SV_I2F_MAGIC
__device__ __forceinline__ float sv_i2f_magic(int32_t x)
{
  return __int_as_float(x + 0x4B400000) - 12582912.0f;
}
#define SV_I2F(x) sv_i2f_magic(x)
#else
#define SV_I2F(x) __int2float_rz(x)
#endif

#include "sv_kernel.cuh"
#include "sv_kernel_pipe.cuh"
#include "sv_kernel_pipe2.cuh"
#include "sv_kernel_pp.cuh"

#if SV_PIPE == 3
#define SV_KERNEL sv_attn_kernel_pp
#elif SV_PIPE == 2
#define SV_KERNEL sv_attn_kernel_pipe2
#elif SV_PIPE == 1
#define SV_KERNEL sv_attn_kernel_pipe
#else
#define SV_KERNEL sv_attn_kernel
#endif

#define SV_STR2(x) #x
#define SV_STR(x) SV_STR2(x)

static constexpr int kCTA_Q = SV_CTA_Q;
static constexpr int kCTA_K = 64;
static constexpr int kWARP_Q = 16;
static constexpr int kWARP_K = 64;
static constexpr int kHEAD = 128;
static constexpr int kThreads = 32 * (kCTA_Q / kWARP_Q) * (kCTA_K / kWARP_K);

static size_t smem_bytes()
{
  //                     smem_Q                     smem_K                    smem_V                    smem_O
  constexpr int kStages = (SV_PIPE == 3) ? 2 : 1;   // the ping-pong holds two K and two V tiles
  return std::max(kCTA_Q * kHEAD * sizeof(int8_t) + kStages * (kCTA_K * kHEAD * sizeof(int8_t) + kCTA_K * kHEAD * sizeof(half)), kCTA_Q * kHEAD * sizeof(half));
}

void attn(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output,
          torch::Tensor query_scale, torch::Tensor key_scale, double sm_scale)
{
  CHECK_CUDA(query); CHECK_CUDA(key); CHECK_CUDA(value); CHECK_CUDA(output); CHECK_CUDA(query_scale); CHECK_CUDA(key_scale);
  CHECK_CONTIGUOUS(query); CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value); CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale); CHECK_CONTIGUOUS(key_scale);
  CHECK_DTYPE(query, torch::kInt8); CHECK_DTYPE(key, torch::kInt8); CHECK_DTYPE(value, torch::kHalf);
  CHECK_DTYPE(output, torch::kBFloat16);
  CHECK_DTYPE(query_scale, torch::kFloat32); CHECK_DTYPE(key_scale, torch::kFloat32);
  CHECK_DIMS(query, 4); CHECK_DIMS(key, 4); CHECK_DIMS(value, 4); CHECK_DIMS(output, 4);
  CHECK_DIMS(query_scale, 3); CHECK_DIMS(key_scale, 3);
  TORCH_CHECK(query.size(3) == kHEAD, "head size must be 128");

  const int batch_size = query.size(0);
  const int qo_len = query.size(1), kv_len = key.size(1);
  const int num_qo_heads = query.size(2), num_kv_heads = key.size(2);
  CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, kHEAD);
  CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, kHEAD);
  TORCH_CHECK(num_qo_heads % num_kv_heads == 0, "num_qo_heads must be divisible by num_kv_heads");
  // the wheel's quantiser layout: BLKQ 128, WARPQ 16 -> 8 scales per 128 rows; BLKK 64 -> 1 per 64 keys
  CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, 128) * 8);
  CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, kCTA_K) * (kCTA_K / kWARP_K));

  const int num_kv_groups = num_qo_heads / num_kv_heads;
  const size_t smem_max = smem_bytes();

  auto kernel_func = SV_KERNEL<kCTA_Q, kCTA_K, kWARP_Q, kWARP_K, kHEAD, DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
                                    float, true, nv_bfloat16, ComputeUnit::kTensorCore, MaskMode::kNone, false, false>;
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max);

  dim3 grid(div_ceil(qo_len, kCTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (kCTA_Q / kWARP_Q) * (kCTA_K / kWARP_K));
  kernel_func<<<grid, block, smem_max, at::cuda::getCurrentCUDAStream()>>>(
      query.data_ptr<int8_t>(), key.data_ptr<int8_t>(), reinterpret_cast<half*>(value.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(output.data_ptr()), nullptr,
      reinterpret_cast<float*>(query_scale.data_ptr()), reinterpret_cast<float*>(key_scale.data_ptr()), nullptr,
      qo_len, kv_len, num_kv_groups,
      query.stride(0), query.stride(1), query.stride(2),
      key.stride(0), key.stride(1), key.stride(2),
      value.stride(0), value.stride(1), value.stride(2),
      output.stride(0), output.stride(1), output.stride(2),
      (float)sm_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

pybind11::dict info()
{
  pybind11::dict d;
  d["SV_CTA_Q"] = SV_CTA_Q;
  d["SV_MIN_BLOCKS"] = SV_MIN_BLOCKS;
  d["SV_I2F_MAGIC"] = SV_I2F_MAGIC;
  d["SV_Q_REGS"] = SV_Q_REGS;
  d["SV_PIPE"] = SV_PIPE;
  d["SV_SV_ORDER"] = SV_SV_ORDER;
  d["SV_BAR3"] = SV_BAR3;
  d["SV_PIPE_UNROLL"] = SV_PIPE_UNROLL;
  d["threads"] = kThreads;
  d["smem_bytes"] = (int)smem_bytes();
  cudaFuncAttributes attr;
  auto kernel_func = SV_KERNEL<kCTA_Q, kCTA_K, kWARP_Q, kWARP_K, kHEAD, DataType::kInt8, QuantGranularity::kPerWarp, QuantGranularity::kPerWarp,
                                    float, true, nv_bfloat16, ComputeUnit::kTensorCore, MaskMode::kNone, false, false>;
  if (cudaFuncGetAttributes(&attr, kernel_func) == cudaSuccess)
  {
    d["num_regs"] = attr.numRegs;
    d["local_bytes"] = (int)attr.localSizeBytes;
    d["static_smem"] = (int)attr.sharedSizeBytes;
    d["max_threads"] = attr.maxThreadsPerBlock;
    d["binary_version"] = attr.binaryVersion;
    d["ptx_version"] = attr.ptxVersion;
    int blocks = 0;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes());
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel_func, kThreads, smem_bytes()) == cudaSuccess)
      d["blocks_per_sm"] = blocks;
  }
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("attn", &attn, "SageAttention qk_int8_sv_f16_accum_f16_attn_inst_buf (head 128, per_warp, non-causal, bf16 out), round J variant");
  m.def("info", &info, "build knobs and kernel attributes");
}
