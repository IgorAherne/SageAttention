// Spellcast3D round P (pascal-attn): the int8 P.V kernel software-pipelined by one iteration (../qattn_pascal_p.cu,
// qk_int8_attn_pipe_kernel: block i's softmax beside block i - 1's P.V, one barrier an iteration, the last iteration
// peeled), the shipped score-loop unrolling. The fp32 P.V kernel is the shipped one. The shipped kernel's bits.
// Op: sageattn_pascal_pp.
#define SC3D_P61_V pp
#define SC3D_P61_PIPE 1
#include "p61_variant_names.h"
#include "../qattn_pascal_p.cu"
