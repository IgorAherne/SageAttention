// Spellcast3D: the twin int8 P.V kernel of qk_int_sv_i8s_cuda_sm80.cu (the package source itself; its knobs as -D
// flags) as a standalone torch extension, for the checks and timings of code/devtools/pixal3d-lowmem/round-n/sage-sched
// in the Spellcast3D hub.
//
//   module.attn(q_int8, k_int8, v_t_int8, out, q_scale, k_scale, v_scale, v_mean, tensor_layout, is_causal,
//               qk_quant_gran, sm_scale, return_lse) -> Tensor        (the op's arguments)
//   module.info() -> dict: knobs, registers, local memory, blocks per SM

#include <torch/extension.h>

#include "../qk_int_sv_i8s_cuda_sm80.cu"

pybind11::dict info()
{
  pybind11::dict d;
  d["SC3D2_ABL"] = SC3D2_ABL;
  d["SC3D2_MAGIC"] = SC3D2_MAGIC;
  d["SC3D2_VWAIT"] = SC3D2_VWAIT;
  d["SC3D2_MINB"] = SC3D2_MINB;
  d["SC3D2_LOADS"] = SC3D2_LOADS;
  d["SC3D2_KLOAD"] = SC3D2_KLOAD;
  d["SC3D2_VBLK"] = SC3D2_VBLK;
  d["SC3D2_W16"] = SC3D2_W16;
  d["threads"] = (int)sc3d_i8s::THREADS;
  d["smem_bytes"] = (int)sc3d_i8s::SMEM_BYTES;
  cudaFuncAttributes attr;
  auto kernel_func = sc3d_i8s::qk_int_sv_i8_attn_kernel<nv_bfloat16, SC3D2_KLOAD ? 1536 : 0>;
  if (cudaFuncGetAttributes(&attr, kernel_func) == cudaSuccess)
  {
    d["num_regs"] = attr.numRegs;
    d["local_bytes"] = (int)attr.localSizeBytes;
    d["binary_version"] = attr.binaryVersion;
    int blocks = 0;
    cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, sc3d_i8s::SMEM_BYTES);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel_func, sc3d_i8s::THREADS, sc3d_i8s::SMEM_BYTES) == cudaSuccess)
      d["blocks_per_sm"] = blocks;
  }
  return d;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("attn", &qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d2, "int8 P.V attention, the twin schedule (head 128, per_warp, not causal)");
  m.def("info", &info, "build knobs and kernel attributes");
}
