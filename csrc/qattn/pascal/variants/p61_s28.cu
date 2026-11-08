// Spellcast3D round O: the channel-slice P.V of p61_s.cu with the score loop unrolled by 2 (as u28): the main loop
// under Pascal's ~8 KB instruction cache. The shipped kernel's bits. Op: sageattn_pascal_s28.
#define SC3D_P61_V s28
#define SC3D_P61_PV_SLICE 1
#define SC3D_P61_I8_BK 32
#define SC3D_P61_QK_UNROLL 2
#define SC3D_P61_PV_UNROLL 8
#include "p61_variant_names.h"
#include "../qattn_pascal_next.cu"
