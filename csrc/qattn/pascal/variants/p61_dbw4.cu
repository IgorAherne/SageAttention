// Spellcast3D round P (pascal-attn): db and w4 together - blocks of 4 warps with double-buffered K / v tiles and one
// barrier an iteration (int8 P.V; the fp32 P.V kernel is the shipped one). The shipped kernel's bits.
// Op: sageattn_pascal_dbw4.
#define SC3D_P61_V dbw4
#define SC3D_P61_DB 1
#define SC3D_P61_I8_WARPS 4
#include "p61_variant_names.h"
#include "../qattn_pascal_p.cu"
