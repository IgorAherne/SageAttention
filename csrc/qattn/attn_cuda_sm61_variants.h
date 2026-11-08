// Spellcast3D round O: twins of the Pascal attention (pascal/variants/*.cu), each the shipped sageattn_pascal's
// arithmetic in the shipped order - the same bits for every input, proven by torch.equal against it - with another
// code shape (pascal/variants/p61_*.cu say which). Pascal's speed depends on the code's size and its shared-memory
// traffic in ways no card of the round could measure, so the fork's install-time calibration times each twin against
// the shipped op on the user's own card and records the faster (pixal3d/modules/sage.py: PASCAL_TWINS); the
// shipped op serves wherever there is no such record. The arguments are sageattn_pascal's.
#pragma once
#include <torch/all.h>

#define SC3D_P61_DECLARE(name)                                                                                    \
  void name(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor output,                   \
            int64_t tensor_layout, double sm_scale, int64_t pv_int8);

SC3D_P61_DECLARE(sageattn_pascal_u28)   // the score loop unrolled by 2 (main loop 7.4 KB, shipped 8.2 KB)
SC3D_P61_DECLARE(sageattn_pascal_u44)   // the P.V loop unrolled by 4 (7.2 KB)
SC3D_P61_DECLARE(sageattn_pascal_s28)   // the P.V by channel slices, the score loop unrolled by 2 (7.5 KB)
SC3D_P61_DECLARE(sageattn_pascal_s24)   // the P.V by channel slices, score loop 2, P.V 4 (6.5 KB)
#ifdef SC3D_P61_DEV_VARIANTS
// developers only (code/devtools/pixal3d-lowmem/round-o/pascal-kernels: the exactness proof covers them too)
SC3D_P61_DECLARE(sageattn_pascal_u24)   // score loop 2, P.V 4 (6.3 KB)
SC3D_P61_DECLARE(sageattn_pascal_s)     // the P.V by channel slices at the shipped unrolling (8.4 KB)
SC3D_P61_DECLARE(sageattn_pascal_n0)    // the channel-slice source with the slices off (a control)
#endif
