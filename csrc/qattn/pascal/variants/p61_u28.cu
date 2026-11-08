// Spellcast3D round O: the shipped Pascal attention (../qattn_pascal.cu, unchanged) with its score loop unrolled by
// 2 chunks of 16 channels (shipped 4) and the P.V loop as shipped (8): the main loop under Pascal's ~8 KB
// instruction cache (shipped: 8,216 bytes) for the fewest added instructions (the fully unrolled P.V keeps its first
// dp4a's 1.5 * 2^23 start as an operand; a P.V loop needs 32 moves for it). The same instructions in the same order
// for every value: the shipped kernel's bits. Op: sageattn_pascal_u28.
#define SC3D_P61_V u28
#define SC3D_P61_QK_UNROLL 2
#define SC3D_P61_PV_UNROLL 8
#include "p61_variant_names.h"
#include "../qattn_pascal.cu"
