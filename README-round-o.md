# Spellcast3D round O: twins of the Pascal attention (sm_61), the wheel post8

Branch `spellcast3d-round-o-pascal` (from `spellcast3d` = post7's source, 1352635; the changes uncommitted when the
wheel was built - the orchestrator decides what ships). Nothing of post7's kernels changes: `_qattn_sm61` gains four
ops, each `sageattn_pascal`'s arithmetic in its order (the same bits for every input) with another code shape, because
Pascal's speed depends on code size and shared-memory traffic in ways no card of the round could measure:

| op | source | what changes | sm_61 main loop (int8 P.V / fp32 P.V) |
|---|---|---|---|
| sageattn_pascal (shipped) | csrc/qattn/pascal/qattn_pascal.cu | - | 8,216 / 8,200 B |
| sageattn_pascal_u28 | pascal/variants/p61_u28.cu | the score loop unrolled by 2 (SC3D_P61_QK_UNROLL) | 7,400 / 7,392 B |
| sageattn_pascal_u44 | pascal/variants/p61_u44.cu | the P.V loop unrolled by 4 (SC3D_P61_PV_UNROLL) | 7,160 / 6,704 B |
| sageattn_pascal_s28 | pascal/variants/p61_s28.cu + pascal/qattn_pascal_next.cu | the P.V by channel slices (round L's candidate at 32-key blocks), score loop by 2 | 7,544 / 6,944 B (48.9 KB shared) |
| sageattn_pascal_s24 | pascal/variants/p61_s24.cu + qattn_pascal_next.cu | channel slices, score 2, P.V 4 | 6,520 / 5,440 B |

`pascal/variants/p61_variant_names.h` renames a translation unit's namespace and entry points by its suffix, so the
shipped source compiles again, untouched, with other knobs (`attn_cuda_sm61_variants.h` declares the entries,
`pybind_sm61.cpp` registers them; u24, s and n0 only with -DSC3D_P61_DEV_VARIANTS, for developers).
`qattn_pascal_next.cu` refuses SC3D_P61_I8_BK != 32 (64 is another rounding). The proofs and tools:
Spellcast3D's `code/devtools/pixal3d-lowmem/round-o/pascal-kernels/` (README.md); the fork's `pixal3d/modules/sage.py`
races the twins at install (PASCAL_TWINS) and serves the shipped op without a record.

Build (the hub's tool; a throw-away venv over the Pixal3D venv's torch 2.8.0+cu128, vcvars64, nvcc 12.8,
`pip wheel --no-deps --no-build-isolation`, TORCH_CUDA_ARCH_LIST "6.1 7.5 8.0 8.6 8.9 9.0 12.0",
SAGEATTENTION_SM80_PTX=1, version suffix +cu128torch2.8.0.post8, exactly as post7):

    code\venvs\pixal3d\Scripts\python.exe code\devtools\pixal3d-lowmem\round-o\pascal-kernels\build_wheels.py sage
