// Spellcast3D: the tuned fp16 P.V kernel of qk_int_sv_f16_cuda_sm80.cu (the package source itself, its sc3d = true
// instantiation for the flows: head 128, per_warp, not causal, no LSE, bf16 out; SC3D_PRESET_S_F16 as a -D flag) as a
// standalone torch extension, for the checks and timings of code/devtools/pixal3d-lowmem/round-n/kernels in the
// Spellcast3D hub. Only that instantiation is compiled (SC3D_F16_KERNEL_ONLY leaves the package's ops out).
//
//   module.attn(q_int8, k_int8, v_fp16, out, q_scale, k_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale,
//               return_lse) -> Tensor        (the op's arguments; anything but the flows' configuration raises)
//   module.info() -> dict: knobs, registers, local memory, blocks per SM

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define SC3D_F16_KERNEL_ONLY
#include "../qk_int_sv_f16_cuda_sm80.cu"

namespace {
constexpr int kCtaQ = 128, kCtaK = 64, kWarpQ = 16, kWarpK = 64, kHead = 128;
auto kernel_ptr()
{
  return qk_int_sv_f16_attn_kernel<kCtaQ, kCtaK, kWarpQ, kWarpK, kHead, DataType::kInt8, QuantGranularity::kPerWarp,
                                   QuantGranularity::kPerWarp, float, true, nv_bfloat16, ComputeUnit::kTensorCore,
                                   MaskMode::kNone, false, false, true>;
}
size_t smem_max()
{
  return std::max(kCtaQ * kHead * sizeof(int8_t) + kCtaK * kHead * sizeof(int8_t) + kCtaK * kHead * sizeof(half),
                  kCtaQ * kHead * sizeof(half));
}
}  // namespace

torch::Tensor attn(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output,
                   torch::Tensor query_scale, torch::Tensor key_scale, int64_t tensor_layout, int64_t is_causal,
                   int64_t qk_quant_gran, double sm_scale, int64_t return_lse)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);
  CHECK_DTYPE(query, torch::kInt8);
  CHECK_DTYPE(key, torch::kInt8);
  CHECK_DTYPE(value, torch::kHalf);
  CHECK_DTYPE(output, torch::kBFloat16);
  TORCH_CHECK(query.size(3) == kHead && is_causal == 0 && qk_quant_gran == static_cast<int64_t>(QuantGranularity::kPerWarp) &&
              return_lse == 0 && tensor_layout == 0, "f16_ext: the flows' configuration only (NHD, head 128, per_warp, not causal, no LSE)");
  const int batch_size = query.size(0);
  const int qo_len = query.size(1), kv_len = key.size(1);
  const int num_qo_heads = query.size(2), num_kv_heads = key.size(2);
  CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, kHead);
  CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, kHead);
  CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, kCtaQ) * (kCtaQ / kWarpQ));
  CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, kCtaK) * (kCtaK / kWarpK));
  const int num_kv_groups = num_qo_heads / num_kv_heads;
  auto kernel_func = kernel_ptr();
  cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max());
  dim3 grid(div_ceil(qo_len, kCtaQ), num_qo_heads, batch_size);
  dim3 block(32, (kCtaQ / kWarpQ) * (kCtaK / kWarpK));
  // the legacy default stream, as the package's op launches
  kernel_func<<<grid, block, smem_max()>>>(
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
  return torch::empty({0});
}

pybind11::dict info()
{
  pybind11::dict d;
  d["SC3D_PRESET_S_F16"] = SC3D_PRESET_S_F16;
  d["SC3D_MIN_BLOCKS"] = SC3D_MIN_BLOCKS;
  d["SC3D_SV_ORDER"] = SC3D_SV_ORDER;
  cudaFuncAttributes attr;
  auto kernel_func = kernel_ptr();
  if (cudaFuncGetAttributes(&attr, kernel_func) == cudaSuccess)
  {
    d["num_regs"] = attr.numRegs;
    d["local_bytes"] = (int)attr.localSizeBytes;
    d["binary_version"] = attr.binaryVersion;
    int blocks = 0;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max());
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel_func, 256, smem_max()) == cudaSuccess)
      d["blocks_per_sm"] = blocks;
  }
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("attn", &attn, "the tuned qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d kernel (the flows' configuration)");
  m.def("info", &info, "build knobs and kernel attributes");
}
