// Spellcast3D round K: the int8 P.V kernel and the v quantiser of qk_int_sv_i8_cuda_sm80.cu (the package source
// itself, its knobs I8_* as -D flags) as a standalone torch extension, for the checks and timings of
// code/devtools/pixal3d-lowmem/round-k/int8pv/kernel in the Spellcast3D hub.
//
//   module.attn(q_int8, k_int8, v_t_int8, out, q_scale, k_scale, v_scale, v_mean, tensor_layout, is_causal,
//               qk_quant_gran, sm_scale, return_lse) -> Tensor        (the op's arguments)
//   module.quant_v(v, v_t, v_scale, v_mean, tensor_layout, smooth_v) -> None
//   module.info() -> dict: knobs, registers, local memory, blocks per SM

#include <torch/extension.h>

#include "../qk_int_sv_i8_cuda_sm80.cu"

pybind11::dict info()
{
  pybind11::dict d;
  d["I8_MIN_BLOCKS"] = I8_MIN_BLOCKS;
  d["I8_I2F_MAGIC"] = I8_I2F_MAGIC;
  d["I8_STUDY"] = I8_STUDY;
  d["threads"] = (int)sc3d_i8::THREADS;
  d["smem_bytes"] = (int)sc3d_i8::SMEM_BYTES;
  cudaFuncAttributes attr;
  auto kernel_func = sc3d_i8::qk_int_sv_i8_attn_kernel<nv_bfloat16>;
  if (cudaFuncGetAttributes(&attr, kernel_func) == cudaSuccess)
  {
    d["num_regs"] = attr.numRegs;
    d["local_bytes"] = (int)attr.localSizeBytes;
    d["static_smem"] = (int)attr.sharedSizeBytes;
    d["binary_version"] = attr.binaryVersion;
    d["ptx_version"] = attr.ptxVersion;
    int blocks = 0;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, sc3d_i8::SMEM_BYTES);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel_func, sc3d_i8::THREADS, sc3d_i8::SMEM_BYTES) == cudaSuccess)
      d["blocks_per_sm"] = blocks;
  }
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("attn", &qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d, "int8 P.V attention (head 128, per_warp, not causal)");
  m.def("quant_v", &quant_per_block_int8_v_cuda, "v -> V^T int8 per 64-token block and channel, scales, mean");
  m.def("info", &info, "build knobs and kernel attributes");
}
