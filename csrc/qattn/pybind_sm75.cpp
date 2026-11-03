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

#include <Python.h>
#include <torch/library.h>

#include "attn_cuda_sm75.h"

extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY static initializers
       below are run. */
    PyObject* PyInit__qattn_sm75(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_qattn_sm75",  /* name of module */
            NULL,           /* module documentation, may be NULL */
            -1,             /* size of per-interpreter state of the module,
                               or -1 if the module keeps state in global variables. */
            NULL,           /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

// Defines the operators. Spellcast3D round L: the Turing (sm_75) build of _qattn_sm80's flows kernel, under the same
// name and schema
TORCH_LIBRARY(sageattention_qattn_sm75, m) {
    m.def("qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "int tensor_layout, "
            "int is_causal, "
            "int qk_quant_gran, "
            "float sm_scale, "
            "int return_lse"
          ") -> Tensor");
    // the int8 P.V twin and its v quantiser (_qattn_sm80's schemas)
    m.def("qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "Tensor value_scale, "
            "Tensor value_mean, "
            "int tensor_layout, "
            "int is_causal, "
            "int qk_quant_gran, "
            "float sm_scale, "
            "int return_lse"
          ") -> Tensor");
    m.def("quant_per_block_int8_v_cuda("
            "Tensor value, "
            "Tensor(a!) value_t, "
            "Tensor(b!) value_scale, "
            "Tensor(c!) value_mean, "
            "int tensor_layout, "
            "int smooth_v"
          ") -> ()");
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(sageattention_qattn_sm75, CUDA, m) {
    m.impl("qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d", &qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d_sm75);
    m.impl("qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d", &qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d_sm75);
    m.impl("quant_per_block_int8_v_cuda", &quant_per_block_int8_v_cuda);
}
