// Spellcast3D round O: the shipped Pascal attention (../qattn_pascal.cu, unchanged) with its main loop unrolled
// less - the score loop by 2 chunks of 16 channels (shipped 4), the P.V loop by 4 key words / keys (shipped 8) - so
// that the int8 P.V kernel's main loop fits Pascal's ~8 KB instruction cache (shipped: 8,216 bytes). The same
// instructions in the same order for every value: the shipped kernel's bits. Op: sageattn_pascal_u24.
#define SC3D_P61_V u24
#define SC3D_P61_QK_UNROLL 2
#define SC3D_P61_PV_UNROLL 4
#include "p61_variant_names.h"
#include "../qattn_pascal.cu"
