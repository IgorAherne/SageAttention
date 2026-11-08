// Spellcast3D round O: the shipped Pascal attention (../qattn_pascal.cu, unchanged) with its P.V loop unrolled by 4
// key words / keys (shipped 8), the score loop as shipped (4): the main loop under Pascal's ~8 KB instruction cache
// with fewer loop trips than u24. The shipped kernel's bits. Op: sageattn_pascal_u44.
#define SC3D_P61_V u44
#define SC3D_P61_QK_UNROLL 4
#define SC3D_P61_PV_UNROLL 4
#include "p61_variant_names.h"
#include "../qattn_pascal.cu"
