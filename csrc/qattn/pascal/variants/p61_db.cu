// Spellcast3D round P (pascal-attn): the int8 P.V kernel with its K and v tiles double-buffered in shared memory,
// one barrier an iteration instead of two (ptxas pairs each BAR.SYNC with a MEMBAR.CTA on sm_61; 30.2 KB a block,
// still 2 blocks an SM). The fp32 P.V kernel is the shipped one. The shipped kernel's bits. Op: sageattn_pascal_db.
#define SC3D_P61_V db
#define SC3D_P61_DB 1
#include "p61_variant_names.h"
#include "../qattn_pascal_p.cu"
