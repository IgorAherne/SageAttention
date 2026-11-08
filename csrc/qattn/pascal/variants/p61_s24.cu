// Spellcast3D round O: the channel-slice P.V of p61_s.cu with the main loop unrolled as u24 (score loop 2, P.V 4).
// The shipped kernel's bits. Op: sageattn_pascal_s24.
#define SC3D_P61_V s24
#define SC3D_P61_PV_SLICE 1
#define SC3D_P61_I8_BK 32
#define SC3D_P61_QK_UNROLL 2
#define SC3D_P61_PV_UNROLL 4
#include "p61_variant_names.h"
#include "../qattn_pascal_next.cu"
