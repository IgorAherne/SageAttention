// Spellcast3D round P (pascal-attn): the int8 P.V kernel in blocks of 4 warps (32 queries, 4 blocks an SM: the same
// 16 warps; a barrier waits for 4 warps, the K / v tiles are staged twice as often). The fp32 P.V kernel is the
// shipped one. The shipped kernel's bits. Op: sageattn_pascal_w4.
#define SC3D_P61_V w4
#define SC3D_P61_I8_WARPS 4
#include "p61_variant_names.h"
#include "../qattn_pascal_p.cu"
