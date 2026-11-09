// Spellcast3D round P (pascal-attn): pp with the score loop unrolled fully (8 chunks): the whole iteration one basic
// block for ptxas's scheduler. The fp32 P.V kernel is the shipped one. The shipped kernel's bits. Op: sageattn_pascal_pp8.
#define SC3D_P61_V pp8
#define SC3D_P61_PIPE 1
#define SC3D_P61_QK_UNROLL 8
#include "p61_variant_names.h"
#include "../qattn_pascal_p.cu"
