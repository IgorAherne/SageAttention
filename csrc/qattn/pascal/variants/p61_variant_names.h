// Spellcast3D round O: the names of one twin of the Pascal attention (pascal/qattn_pascal.cu, or the channel-slice
// source pascal/qattn_pascal_next.cu), compiled again in its own translation unit with other compile-time knobs.
// SC3D_P61_V is the twin's suffix: its namespace becomes sc3d_p61_<V> and its entry points sc3d_p61_<V>_* and
// sageattn_pascal_<V> (the op the fork calls; attn_cuda_sm61_variants.h declares it). Include this file first, then
// the kernel's source.
#pragma once
#ifndef SC3D_P61_V
#error "SC3D_P61_V (the twin's suffix) must be defined"
#endif
#define SC3D_P61_CAT2(a, b) a##b
#define SC3D_P61_CAT(a, b) SC3D_P61_CAT2(a, b)
#define sc3d_p61 SC3D_P61_CAT(sc3d_p61_, SC3D_P61_V)
#define quant_qk_int8_pascal SC3D_P61_CAT(sc3d_p61_quant_qk_, SC3D_P61_V)
#define quant_v_int8_pascal SC3D_P61_CAT(sc3d_p61_quant_v_, SC3D_P61_V)
#define qk_int8_sv_f32_attn_pascal SC3D_P61_CAT(sc3d_p61_f32_attn_, SC3D_P61_V)
#define qk_int8_sv_i8_attn_pascal SC3D_P61_CAT(sc3d_p61_i8_attn_, SC3D_P61_V)
#define sageattn_pascal SC3D_P61_CAT(sageattn_pascal_, SC3D_P61_V)
