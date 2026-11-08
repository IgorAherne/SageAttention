// Spellcast3D round O: the Pascal attention with the P.V by channel slices (../qattn_pascal_next.cu, round L's
// candidate: a warp takes 16 channels of all 64 queries, per key 3 shared-memory wavefronts instead of 6, P
// double-buffered, the rescale factors and denominators through shared memory), 32 keys an iteration, the shipped
// unrolling. Each output is one lane's fma chain over the keys in the shipped order with the shipped factors: the
// shipped kernel's bits. Op: sageattn_pascal_s.
#define SC3D_P61_V s
#define SC3D_P61_PV_SLICE 1
#define SC3D_P61_I8_BK 32
#include "p61_variant_names.h"
#include "../qattn_pascal_next.cu"
