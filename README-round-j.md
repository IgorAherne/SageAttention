# Spellcast3D round J edits to SageAttention (woct0rdho v2.2.0-windows.post3)

Base: https://github.com/woct0rdho/SageAttention tag `v2.2.0-windows.post3` (commit
3abe607071a3031b2a785b9a76fec05efaf69620) - the source of the wheel Spellcast3D's Pixal3D shipped until round J
(`sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl`, built by woct0rdho's GitHub workflow with
CUDA 12.8.1). Shallow clone; the edits are in the working tree and in two patches (git diff against the tag):

- `round-j-sagecuda-package.patch` - the four package files the wheel is built from (`setup.py`,
  `csrc/qattn/qk_int_sv_f16_cuda_sm80.cu`, `csrc/qattn/attn_cuda_sm80.h`, `csrc/qattn/pybind_sm80.cpp`): apply this
  one to fork and commit.
- `round-j-sagecuda.patch` - the same plus the new files (this README, the study folder `csrc/qattn/sagecuda/`,
  which setup.py never compiles).

Apply to a clone of the tag: `git apply round-j-sagecuda-package.patch` (or the full one). The edit can also be made
by `code/devtools/pixal3d-lowmem/round-j/sagecuda/apply_package_edit.py --src <clone>` and then
`apply_package_edit_post5.py --src <clone>` in the Spellcast3D hub.

Two wheels came out of round J: **post4** (the tuned kernel in place of the shipped one) and **post5** (both kernels,
the shipped one under the original op and the tuned one under a second op, so that the caller can choose per card:
see "post5" below - the current edit and the wheel Spellcast3D ships).

## post5: both kernels, two ops (what the post5 wheel carries)

The tuned kernel is -13 % a call on an RTX 3060 but unmeasured on every other card (its launch bounds force 128
registers with 64-72 bytes of spills, which may behave differently elsewhere), and a user's card must never get
slower. post5 therefore keeps the shipped kernel exactly as it was and adds the tuned one beside it:

- **`torch.ops.sageattention_qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf`** (the original op; what
  `sageattn_qk_int8_pv_fp16_cuda(..., pv_accum_dtype="fp16+fp32")` calls): the shipped kernel for every
  configuration - the code of the wheel post3 was built from. Its signature, its behaviour and its bits are post3's;
  the Python package (`sageattention/*.py`) is post3's byte for byte.
- **`torch.ops.sageattention_qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d`** (new; the same schema:
  `query, key, value, output, query_scale, key_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale,
  return_lse`): for head 128 / not causal / per_warp / no LSE / bf16 output - the configuration Pixal3D's flows call -
  the tuned instantiation (post4's tuned kernel instruction for instruction); for every other configuration the
  shipped kernel. The same bits as the original op for every input (checked: 52 / 52 cases against post3's kernel,
  and the op's whole option grid against the original op).

How it is built (`apply_package_edit_post5.py`, on top of post4's edit):

1. The kernel template `qk_int_sv_f16_attn_kernel` gets a last template parameter `bool sc3d = false`, and the
   predicate `sc3d_tuned<..., sc3d>` that switches post4's two changes on (`__launch_bounds__(256, 2)`, the P.V step
   one output tile at a time) is true only with `sc3d = true` and the flows' configuration. Every op of the file
   launches `sc3d = false` instantiations - the shipped code - so their SASS is post3's (sm_80: all 128 kernels
   identical instruction for instruction; sm_86 / sm_89: the native SASS of the same source, as in post4). The
   kernel symbols carry the extra template argument (`...ELb0EEv...` for the shipped kernels, `...ELb1EEv...` for
   the tuned one); nothing else of the module changes.
2. `qk_int8_sv_f16_accum_f16_attn_inst_buf`'s body becomes the template `qk_int8_sv_f16_accum_f16_attn_inst_buf_impl
   <bool sc3d>`, which launches `qk_int_sv_f16_attn_kernel<..., SC3D_ON>` with `SC3D_ON = sc3d_tuned<..., sc3d>::value`
   (one extra kernel in the module, not sixteen). The original op calls it with `false`, the new op with `true`.
3. `attn_cuda_sm80.h` declares the new op, `pybind_sm80.cpp` defines and registers it (CUDA implementation).

Calling it from Python - what `sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="NHD", is_causal=False,
qk_quant_gran="per_warp", pv_accum_dtype="fp16+fp32")` does at head size 128, with the second op:

    import torch, sageattention.core                      # loads _qattn_sm80 (registers the ops)
    from sageattention.quant import per_warp_int8
    op = torch.ops.sageattention_qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf_sc3d
    km = k.mean(dim=1, keepdim=True)
    q8, q_scale, k8, k_scale = per_warp_int8(q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=16, BLKK=64)
    o = torch.empty(q.size(), dtype=q.dtype, device=q.device)
    op(q8, k8, v.to(torch.float16), o, q_scale, k_scale, 0, 0, 2, q.size(-1) ** -0.5, 0)   # NHD, not causal, per_warp, no LSE

`hasattr`/`getattr` on `torch.ops.sageattention_qattn_sm80` tells whether an installed wheel has the op (post3 and
post4 do not). Spellcast3D's Pixal3D fork (`pixal3d/modules/sage.py`) does exactly this: at its first SageAttention
call of a process it runs both on its self-check table (the same bits required), times the two ops on random int8
tables at 21,504 and 8,192 tokens (12 heads x 128, interleaved, CUDA events) and keeps the tuned op only where it is
at least 2 % faster at both sizes - the shipped op otherwise, and wherever the timing cannot run.

## The package edit of post4 (kept by post5 as the `sc3d = true` code)

### `csrc/qattn/qk_int_sv_f16_cuda_sm80.cu`

ONE kernel instantiation changes: the one `sageattn_qk_int8_pv_fp16_cuda(..., qk_quant_gran="per_warp",
pv_accum_dtype="fp16+fp32")` launches for head 128, not causal, no LSE, bf16 output -
`qk_int_sv_f16_attn_kernel<128, 64, 16, 64, 128, kInt8, kPerWarp, kPerWarp, float, true, nv_bfloat16, kTensorCore,
kNone, false, false>` from `qk_int8_sv_f16_accum_f16_attn_inst_buf` (Pixal3D's flows call exactly that). The
predicate `sc3d_tuned<...>::value` selects it at compile time; the other 127 instantiations of the module compile as
before (their sm_80 SASS is post3's instruction for instruction).

1. `SC3D_MIN_BLOCKS 2`: `__launch_bounds__(256, 2)` on it - at most 128 registers a thread (ptxas spills 72 / 68
   bytes, a 40-byte stack at sm_86 / sm_89; 64 / 64, 32 bytes at sm_80) and two blocks (16 warps) an SM instead of
   one (176-184 registers). The kernel declaration gets `__launch_bounds__(SC3D_TUNED::max_threads,
   SC3D_TUNED::min_blocks)`, which is `(256, 2)` for the tuned instantiation and `(0, 0)` for every other one -
   `(0, 0)` emits no `.maxntid` / `.minnctapersm` directive at all, i.e. no launch bounds (checked in the PTX).
2. `SC3D_SV_ORDER 1`: its P.V step (`compute_fp16_sv_permuted_inst_buf` in attn_utils.cuh) with the loops swapped,
   one 16-column output tile at a time (`sc3d_sv_fv`): the tile's four 16-key fp16 MMAs (fk 0 initialises, 1-3
   accumulate), then its fp32 adds - instead of the eight tiles' fp16 buffers at once. Every tile is the same chain of
   MMA instructions on the same data in the same order followed by the same fp32 additions, so the values are the
   same; 32 fewer live registers. The shared-memory offsets are the original walk in closed form, checked for every
   lane against the walk.

Why the bits cannot change: the arithmetic instructions of the compiled kernel are the shipped kernel's (opcode
counts of FFMA, FMUL, HMMA.16816.F16/F32, HADD2.F32, IMMA, MUFU.EX2, FMNMX, I2F / I2FP, F2FP equal; only address
arithmetic, moves and spills differ), the MMA accumulation order per output value is unchanged (fp16 over each
64-key block, then fp32), and the quantisation (the wheel's quantisers) is untouched. Measured: bit-equal (torch.equal)
on Pixal3D's 32 captured q/k/v tables (sv1536 hr / tex / ss, mv1024 hr; steps 0 and 9; blocks 0/10/20/29) and random
tables at 65 ... 41,637 tokens, deterministic (round J sagecuda, `bitcheck-final.json`: 52 / 52).

Per architecture: one nvcc command compiles every `-gencode` of a file with the same `-D` flags (setup.py's
CUDAExtension), so the knobs could only differ per architecture through `__CUDA_ARCH__` in the source. They do not
need to: on an RTX 3060 the same values were the best (or within noise of the best) both as sm_86 SASS and as sm_80
SASS, so the source has one setting. sm_89 takes it as well: its SASS is the sm_86 SASS instruction for instruction
apart from the F2FP opcode spellings (`F2FP.F16.F32.PACK_AB` for `F2FP.PACK_AB`, the same conversion) - not run (no
sm_89 card here). `-DSC3D_MIN_BLOCKS=0 -DSC3D_SV_ORDER=0` compiles the original kernel.

### `setup.py`

- `_qattn_sm80` gets SASS for 8.6 and 8.9 as well when `TORCH_CUDA_ARCH_LIST` names them (woct0rdho's workflow
  passes "8.0 8.6 8.9 9.0 12.0"): before, every 8.x card ran the sm_80 SASS (the sm_86 SASS alone is -3 %; with the
  tuned kernel -13.6 % against -12.8 % for the same kernel as sm_80 SASS on the 3060).
- `SAGEATTENTION_SM80_PTX=1`: PTX of `_qattn_sm80`'s newest SASS target (compute_89) for the cards after it ("+PTX"
  in TORCH_CUDA_ARCH_LIST would drop 8.9 from the other extensions, whose allowed lists match exact strings).
- `NVCC_THREADS`: nvcc's `--threads` (default `os.cpu_count()` as before), to keep a build light.

## Measured (RTX 3060 12 GB, sm_86, 28 SMs; CUDA events, interleaved, medians of 30; kernel time only)

ms per call, 12 heads x 128, one sequence, not causal; "wheel" = post3's kernel through torch.ops:

| kernel | SASS | 4,096 | 17,446 | 21,504 | 21,547 | 41,637 |
|---|---|---:|---:|---:|---:|---:|
| post3 (wheel) | sm_80 | 2.158 | 38.067 | 57.145 | 58.318 | 215.772 |
| shipped source | sm_86 | 2.103 (-2.5 %) | 36.772 (-3.4 %) | 55.395 (-3.1 %) | 56.488 (-3.1 %) | 209.557 (-2.9 %) |
| **round J kernel** | **sm_86** | **1.827 (-15.3 %)** | **32.957 (-13.4 %)** | **49.493 (-13.4 %)** | **50.404 (-13.6 %)** | **187.218 (-13.2 %)** |
| round J kernel | sm_80 | 1.868 (-13.4 %) | 33.319 (-12.5 %) | 50.018 (-12.5 %) | 50.876 (-12.8 %) | 189.331 (-12.3 %) |

The whole study (40-odd variants: native SASS, launch bounds, CTA_Q 64, q in registers, int-to-float on the FMA
pipe, pipelined and ping-pong main loops, barrier counts, P.V orders), its scripts and tables:
`code/devtools/pixal3d-lowmem/round-j/sagecuda/` in the Spellcast3D hub (results in its git-ignored
`code/logs/profiling/pixal3d-lowmem/round-j/sagecuda/`).

## Building the post5 wheel (as the post5 in Spellcast3D's `code/pixal3d/whl/` was built)

As post4 below, with the post5 edit in the working tree and `SAGEATTENTION_WHEEL_VERSION_SUFFIX=+cu128torch2.8.0.post5`
(in the hub: `code/devtools/pixal3d-lowmem/round-j/sagecuda/build_wheel_post5.py --max-jobs 4 --nvcc-threads 1`,
which runs build_wheel.py's build and post5's own analysis) ->
`sageattention-2.2.0+cu128torch2.8.0.post5-cp39-abi3-win_amd64.whl`: 20,379,790 bytes, sha256
530027fa713c3b5dd30dd7addd17733ab3fd60efc0e946501e391846d53f1604; 532 s of compiling (Ryzen 9 7950X, 4 jobs, 1 nvcc
thread), 704 s with the analysis. Architectures (cuobjdump) as post4: `_qattn_sm80` SASS sm_80 / sm_86 / sm_89 + PTX
compute_89; `_qattn_sm89` sm_89 / sm_120a; `_qattn_sm90` sm_90a; `_fused` sm_80 / sm_89 / sm_90a / sm_120a. Against
the earlier wheels, kernel by kernel (normalised SASS; post5's names mapped back by dropping the `sc3d` argument):

| module, SASS | against | result |
|---|---|---|
| `_qattn_sm80` sm_80 | post3 | the 128 shipped kernels identical; +1 kernel (the tuned one) = post4's flows kernel = study build o1lb2_80 |
| `_qattn_sm80` sm_86 | post4 | 127 of 128 identical; the flows' kernel (tuned in post4) is now the shipped code = study build ctrl86; the tuned one = post4's flows kernel = o1lb2_86 |
| `_qattn_sm80` sm_89 | post4 | 127 of 128 identical; the flows' shipped kernel = its own sm_86 SASS (F2FP spellings mapped); the tuned one = post4's = o1lb2_89 |
| `_fused`, `_qattn_sm89`, `_qattn_sm90`, every arch | post3 | identical (44 / 224 / 64 kernels) |
| file list, the 12 `.py` files | post3 | equal, byte for byte |

## Verified after installing post5 into Pixal3D's venv (RTX 3060, sm_86, 2026-10-19)

- Every kernel of `_qattn_sm80` through the wheel's own entry (`sageattn_qk_int8_pv_fp16_cuda`'s whole option grid,
  128 calls): post3's bits (output and LSE hashes).
- Both ops against post3's kernel (study build ctrl80) on 52 cases (round I's 32 captured q / k / v + random tables
  at 65 ... 41,637 tokens x 2 seeds): shipped op 52 / 52, `_sc3d` op 52 / 52 bit-equal, both deterministic; the
  whole entry 52 / 52; Pixal3D's call of the `_sc3d` op (the wheel's preprocessing + the op) 52 / 52. The `_sc3d` op
  over the inst_buf op's option grid (fp16 / bf16, head 64 / 128, causal, per_warp / per_thread, LSE: 32 calls)
  gives the shipped op's output and LSE in every one.
- ms per call (kernel only; interleaved blocks of 6 calls after a 3 s warm-up, medians of 30; % against post3's
  kernel ctrl80):

| N | post3 (ctrl80) | shipped op | tuned op | post4's kernel (o1lb2_86) | tuned vs shipped |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 2.143 | 2.088 (-2.6 %) | 1.787 (-16.6 %) | 1.785 (-16.7 %) | -14.4 % |
| 17,446 | 37.492 | 36.435 (-2.8 %) | 32.563 (-13.1 %) | 32.509 (-13.3 %) | -10.6 % |
| 21,504 | 56.517 | 54.815 (-3.0 %) | 49.065 (-13.2 %) | 49.060 (-13.2 %) | -10.5 % |
| 21,547 | 57.637 | 55.874 (-3.1 %) | 49.907 (-13.4 %) | 49.868 (-13.5 %) | -10.7 % |
| 41,637 | 215.414 | 208.801 (-3.1 %) | 187.281 (-13.1 %) | 186.749 (-13.3 %) | -10.3 % |

  The shipped op runs its native sm_86 SASS here (= ctrl86 within 0.8 %), hence -3 % against post3's sm_80 SASS;
  the tuned op is post4's kernel to within 0.3 %.
- Pixal3D's passes (sv1536, the network as the engine ships it, 30 SageAttention calls a pass): hr 2.695 s with
  post4's kernel, 2.690 s with post5 as the fork chooses (the tuned op), 2.891 s with post5's shipped op forced
  (+7.3 %); tex 2.700 / 2.699 / 2.895 s (+7.2 %); one pass bit-equal across the three. In the passes the tuned op
  saves 6.5-6.7 ms a call against the shipped one (-11.8 %).
- Pixal3D's first-use choice (`sage._race`, 21,504 and 8,192 tokens, ABBA, 4 / 6 calls each, ~0.5 s): the tuned op
  wins every time (14 runs: -11.1 to -13.9 % at 21,504, -13.2 to -13.9 % at 8,192 tokens, against the 5 % margin);
  forced kernels, a wheel without the op, a margin it cannot reach, no memory for the tables and a failing timing all
  give the shipped call. Pixal3D's golden check with Sage off: 34 / 34.
- Not measured: any card but the RTX 3060. Every other sm_8x card chooses for itself at its first use.

## Building the post4 wheel (as the post4, now in Spellcast3D's `release/cache/wheels/keep/`, was built)

In the vcvars64 environment (Visual Studio 2022), CUDA 12.8 (nvcc V12.8.93), Python 3.11, torch 2.8.0+cu128,
setuptools 80.9.0, wheel 0.45.1, ninja 1.13.2:

    set DISTUTILS_USE_SDK=1
    set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
    set TORCH_CUDA_ARCH_LIST=8.0 8.6 8.9 9.0 12.0
    set SAGEATTENTION_WHEEL_VERSION_SUFFIX=+cu128torch2.8.0.post4
    set SAGEATTENTION_SM80_PTX=1
    set MAX_JOBS=4
    set NVCC_THREADS=1
    python -m pip wheel . --no-deps --no-build-isolation -w dist

from a checkout with LF line endings (woct0rdho's workflow clones with `core.autocrlf false`; with CRLF the wheel's
.py files differ from post3's by their line endings) -> `sageattention-2.2.0+cu128torch2.8.0.post4-cp39-abi3-win_amd64.whl`
(20,309,844 bytes, sha256 7fc41e1c2fefd93344c064769fe5d84e05ed6aa8c9af6663ead6c245178e40cc; 578 s of compiling on
a Ryzen 9 7950X with 4 jobs beside other work, 458 s for a first build). The hub's driver
`code/devtools/pixal3d-lowmem/round-j/sagecuda/build_wheel.py` does this from the clone's tracked files (index line
endings) in a throw-away build venv and writes the provenance JSON (archs by cuobjdump, every kernel's SASS against
post3's: `_fused`, `_qattn_sm89`, `_qattn_sm90` identical kernel for kernel at every architecture, `_qattn_sm80`
127 of 128 at sm_80 - the tuned one differs, as intended - plus the new sm_86 / sm_89 SASS and compute_89 PTX; the
file list and the 12 Python files byte-equal to post3's).

## Verified after installing post4 into Pixal3D's venv (RTX 3060, 2026-10-19)

- The fork's self-check (`pixal3d.modules.sage._check`): passes, kernel `sageattn_qk_int8_pv_fp16_cuda (per_warp,
  fp16+fp32)`.
- Every kernel of `_qattn_sm80` (the whole option grid of `sageattn_qk_int8_pv_fp16_cuda`: fp16 / bf16, head 64 /
  128, causal or not, per_warp / per_thread, fp32 / fp16 / fp16+fp32 / fp16+smooth_v accumulation, with and without
  LSE: 128 calls) gives post3's bits (sha256 of output and LSE; post3 ran its sm_80 SASS here, post4 its sm_86 SASS).
- The installed kernel against post3's kernel (the study build ctrl80, instruction for instruction post3's): bit-equal
  on 52 / 52 cases (32 captures + 10 token counts x 2 seeds), deterministic.
- Round I's error tables (sf_real.py, 32 captures against fp32 attention): every figure equal to the last digit.
- Pixal3D passes (sv1536, the network as the engine ships it): hr 2.936 -> 2.688 s (-8.45 %), tex 2.936 -> 2.700 s
  (-8.0 %), one pass bit-equal post3 vs post4.
- Pixal3D's golden check with Sage off (check_engine.py sv1024): 34 / 34.

## The study folder `csrc/qattn/sagecuda/` (not built into the package)

- `sv_kernel.cuh` - a copy of `qk_int_sv_f16_attn_kernel` with compile-time knobs (`SV_*` macros). With every knob at
  its default the compiled kernel is the wheel's kernel instruction for instruction (cuobjdump -sass, sm_80).
- `sv_kernel_pipe.cuh`, `sv_kernel_pipe2.cuh`, `sv_kernel_pp.cuh` - generated variants of its main loop (pipelined
  Q.K^T, hand-unrolled, a ping-pong of two warp groups; gen_pipe.py in the hub): bit-equal, not faster.
- `sv_ext.cu` - a standalone torch extension holding that one kernel, built with the knobs as `-D` flags.
