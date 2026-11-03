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

#pragma once
#include <torch/all.h>

// Spellcast3D round L: the flows' kernel (qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d of _qattn_sm80) for Turing,
// sm_75 (qk_int_sv_f16_cuda_sm75.cu): the same arguments; head 128, not causal, per_warp, no LSE
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
                    int64_t return_lse);

// the int8 P.V twin (qk_int_sv_i8_cuda_sm75.cu): _qattn_sm80's qk_int8_sv_i8_accum_i32_attn_inst_buf_sc3d for Turing
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
                    int64_t return_lse);

// its v quantiser: qk_int_sv_i8_cuda_sm80.cu's, compiled for sm_75
void quant_per_block_int8_v_cuda(torch::Tensor value,
                    torch::Tensor value_t,
                    torch::Tensor value_scale,
                    torch::Tensor value_mean,
                    int64_t tensor_layout,
                    int64_t smooth_v);
