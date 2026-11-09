// Spellcast3D round P (pascal-attn): the int8 P.V kernel at 64 keys an iteration (../qattn_pascal_p.cu,
// qk_int8_attn_k64_kernel): the shipped kernel's two 32-key blocks in order - each its own P / v scale block, softmax
// step, rescale and P.V - with one fully unrolled score loop that reads a lane's two q rows once a 16-channel chunk for
// the 8 keys of both blocks (the shipped loop: once for 4; on a GTX 1070 Ti the q loads cost 16 % of the kernel), two
// barriers per 64 keys. The fp32 P.V kernel (ss / lr / hr block 0) is the shipped kernel's code with its score loop
// unrolled fully (the same bits; -1..-3 % a call there: under the install race's margin, not raced). The shipped
// kernel's bits.
// Op: sageattn_pascal_k64.
#define SC3D_P61_V k64
#define SC3D_P61_K64 1
#define SC3D_P61_QK_UNROLL 8
#include "p61_variant_names.h"
#include "../qattn_pascal_p.cu"
