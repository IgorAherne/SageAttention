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

#include "attn_cuda_sm61.h"
#include "attn_cuda_sm61_variants.h"

extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY static initializers
       below are run. */
    PyObject* PyInit__qattn_sm61(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_qattn_sm61",  /* name of module */
            NULL,           /* module documentation, may be NULL */
            -1,             /* size of per-interpreter state of the module,
                               or -1 if the module keeps state in global variables. */
            NULL,           /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

// Spellcast3D round L: the Pascal (sm_61) kernels, attn_cuda_sm61.h
TORCH_LIBRARY(sageattention_qattn_sm61, m) {
    m.def("quant_qk_int8_pascal("
            "Tensor query, "
            "Tensor key, "
            "Tensor(a!) query_int8, "
            "Tensor(b!) query_scale, "
            "Tensor(c!) key_int8, "
            "Tensor(d!) key_scale, "
            "Tensor(e!) key_mean, "
            "int tensor_layout, "
            "int smooth_k"
          ") -> ()");
    m.def("quant_v_int8_pascal("
            "Tensor value, "
            "Tensor(a!) value_int8, "
            "Tensor(b!) value_scale, "
            "Tensor(c!) value_mean, "
            "int tensor_layout, "
            "int smooth_v"
          ") -> ()");
    m.def("qk_int8_sv_f32_attn_pascal("
            "Tensor query_int8, "
            "Tensor key_int8, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "int tensor_layout, "
            "float sm_scale"
          ") -> ()");
    m.def("qk_int8_sv_i8_attn_pascal("
            "Tensor query_int8, "
            "Tensor key_int8, "
            "Tensor value_int8, "
            "Tensor(a!) output, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "Tensor value_scale, "
            "Tensor value_mean, "
            "int kv_len, "
            "int tensor_layout, "
            "float sm_scale"
          ") -> ()");
    m.def("sageattn_pascal("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "int tensor_layout, "
            "float sm_scale, "
            "int pv_int8"
          ") -> ()");
    // Spellcast3D round O: sageattn_pascal's twins (attn_cuda_sm61_variants.h), its arguments and its bits
#define SC3D_P61_TWIN_SCHEMA(name) m.def(name "(Tensor query, Tensor key, Tensor value, Tensor(a!) output, int tensor_layout, float sm_scale, int pv_int8) -> ()")
    SC3D_P61_TWIN_SCHEMA("sageattn_pascal_u28");
    SC3D_P61_TWIN_SCHEMA("sageattn_pascal_u44");
    SC3D_P61_TWIN_SCHEMA("sageattn_pascal_s28");
    SC3D_P61_TWIN_SCHEMA("sageattn_pascal_s24");
    SC3D_P61_TWIN_SCHEMA("sageattn_pascal_k64");   // Spellcast3D round P
    SC3D_P61_TWIN_SCHEMA("sageattn_pascal_px");    // Spellcast3D round P sage-bias: the precise Q.K^T (other bits)
#ifdef SC3D_P61_DEV_VARIANTS
    SC3D_P61_TWIN_SCHEMA("sageattn_pascal_u24");
    SC3D_P61_TWIN_SCHEMA("sageattn_pascal_s");
    SC3D_P61_TWIN_SCHEMA("sageattn_pascal_n0");
#endif
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(sageattention_qattn_sm61, CUDA, m) {
    m.impl("quant_qk_int8_pascal", &quant_qk_int8_pascal);
    m.impl("quant_v_int8_pascal", &quant_v_int8_pascal);
    m.impl("qk_int8_sv_f32_attn_pascal", &qk_int8_sv_f32_attn_pascal);
    m.impl("qk_int8_sv_i8_attn_pascal", &qk_int8_sv_i8_attn_pascal);
    m.impl("sageattn_pascal", &sageattn_pascal);
    m.impl("sageattn_pascal_u28", &sageattn_pascal_u28);
    m.impl("sageattn_pascal_u44", &sageattn_pascal_u44);
    m.impl("sageattn_pascal_s28", &sageattn_pascal_s28);
    m.impl("sageattn_pascal_s24", &sageattn_pascal_s24);
    m.impl("sageattn_pascal_k64", &sageattn_pascal_k64);
    m.impl("sageattn_pascal_px", &sageattn_pascal_px);
#ifdef SC3D_P61_DEV_VARIANTS
    m.impl("sageattn_pascal_u24", &sageattn_pascal_u24);
    m.impl("sageattn_pascal_s", &sageattn_pascal_s);
    m.impl("sageattn_pascal_n0", &sageattn_pascal_n0);
#endif
}
