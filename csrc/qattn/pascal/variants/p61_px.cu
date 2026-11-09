// Spellcast3D round P (sage-bias): the Pascal attention with a more exact Q.K^T - NOT the shipped bits (a precision
// change, ../qattn_pascal_p.cu SC3D_P61_PRECISE): q and k each minus its mean over the tokens and one scale per token,
// q's mean times k (exact) added back to the scores per key; k64's schedule for the int8 P.V (64 keys an iteration,
// one score loop for both 32-key blocks) and the shipped kernel's code with its score loop unrolled fully for the fp32
// P.V. The P.V itself is the shipped one. Chosen over the shipped op on a GTX 10xx by the install's calibration once
// its accuracy is confirmed there (the fork's sage.py: PASCAL_PRECISE).
// Op: sageattn_pascal_px.
#define SC3D_P61_V px
#define SC3D_P61_K64 1
#define SC3D_P61_QK_UNROLL 8
#define SC3D_P61_PRECISE 1
#include "p61_variant_names.h"
#include "../qattn_pascal_p.cu"
