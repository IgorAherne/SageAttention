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

// Spellcast3D round L: SageAttention on Pascal (sm_61, dp4a; pascal/qattn_pascal.cu). Head 128, not causal, the same
// heads for q and k / v. q, k, v, output: [B, N, H, 128] (tensor_layout 0) or [B, H, N, 128] (1), float32, float16 or
// bfloat16, the last dimension contiguous. The quantised layouts (Npad: the tokens rounded up to 64): q / k int8
// [B, H, Npad, 128], q scales [B, H, Npad / 16], k scales [B, H, Npad / 64], k mean [B, H, 128]; v for the int8 P.V
// int32 words of 4 keys [B, H, Npad / 4, 128], its scales [B, H, Npad / 32, 128] and mean [B, H, 128] (float32).

#pragma once
#include <torch/all.h>

// q per 16 tokens, k per 64 tokens with its channel mean subtracted (smooth_k), as int8
void quant_qk_int8_pascal(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor query_int8,
                    torch::Tensor query_scale,
                    torch::Tensor key_int8,
                    torch::Tensor key_scale,
                    torch::Tensor key_mean,
                    int64_t tensor_layout,
                    int64_t smooth_k);

// v minus its channel mean (smooth_v) as int8 per 32 tokens and channel, for the int8 P.V
void quant_v_int8_pascal(torch::Tensor value,
                    torch::Tensor value_int8,
                    torch::Tensor value_scale,
                    torch::Tensor value_mean,
                    int64_t tensor_layout,
                    int64_t smooth_v);

// int8 Q.K^T, fp32 P.V (value in any of the three float types)
void qk_int8_sv_f32_attn_pascal(torch::Tensor query_int8,
                    torch::Tensor key_int8,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    int64_t tensor_layout,
                    double sm_scale);

// int8 Q.K^T, int8 P.V (P u8 per 32-key block, v from quant_v_int8_pascal); kv_len: the keys' count
void qk_int8_sv_i8_attn_pascal(torch::Tensor query_int8,
                    torch::Tensor key_int8,
                    torch::Tensor value_int8,
                    torch::Tensor output,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    torch::Tensor value_scale,
                    torch::Tensor value_mean,
                    int64_t kv_len,
                    int64_t tensor_layout,
                    double sm_scale);

// the whole attention: the quantisers, then the fp32 P.V kernel or (pv_int8) the int8 one
void sageattn_pascal(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    int64_t tensor_layout,
                    double sm_scale,
                    int64_t pv_int8);
