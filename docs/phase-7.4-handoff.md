# Phase 7.4 handoff, end of 2026-08-13

Working state for `anamnesis` v0.7.4 (`remember`-path caller-chosen output
dtype). Read this first, then
[`phase-7.4-bf16-perf-discrepancy.md`](phase-7.4-bf16-perf-discrepancy.md) for
the one open question.

## Where things stand

Branch `phase-7.4-remember-output-dtype`, 5 commits ahead of `main`, working
tree clean, **not pushed**. Full suite green: 498 lib tests plus every
cross-validation suite, `cargo clippy --all-targets --all-features -D warnings`
clean, `cargo fmt` clean.

| commit | what |
|---|---|
| `bc11d0f` | the four families generic over `OutputElement`; `TargetDtype::{F32,F16}`; boundary dispatch; `convert`'s v0.7.4 gate removed |
| `587fb80` | recover most of the `BF16` regression (hoist FP8 tile, register tiles, GPTQ/AWQ in place) |
| `66bb10b` | tile `BnB` `INT8`; share `VECTOR_TILE`; record the code-layout finding |
| `eac4793` | **reject** the per-element `write_one` hook, with the asm evidence |
| `36e3c8a` | whole-model threaded bench; record the perf contradiction |

## Done

- **A1–A4** kernels: `FP8`, `GPTQ`, `AWQ`, `BnB` all generic over
  `OutputElement`. Eight new generic entry points; every `*_to_bf16` name kept
  as an `#[inline]` `Bf16Out` wrapper, so **no existing caller changes**.
- **A5** investigated and *closed as not-needed*: `MAX_OUTPUT_BYTES` correctly
  stays `gguf`-gated. v0.7.3 predicted the gate would widen; the chosen design
  routes the four families through an `f32` scratch, so they never size a byte
  buffer. Doc comment records this instead of the prediction.
- **B1** output-side plumbing: `build_views::<E>` takes `E::DTYPE`,
  `hub_tensors::<E>`, and `read_safetensors` in `src/convert.rs` now dispatches
  over the dtype instead of rejecting non-`BF16` on quantised safetensors input.
  *(This step was not in the ROADMAP; without it `convert --out-dtype f32`
  would still fail on the exact input the phase exists to serve.)*

## Next up, in order

1. **Settle the perf discrepancy** (agreed for 2026-08-14). See the companion
   doc. Start with the allocation-control experiment, then CodSpeed.
2. **Vectorisation annotations.** Four `// VECTORIZED: pending` remain in
   `src/remember/` (`fp8.rs`, `gptq.rs`, `awq.rs`, `bnb.rs`) on loops this phase
   touched. Asm was already inspected once and all five kernels vectorise
   8-wide at every width, with `vcvtps2ph` for `F16`; what blocks writing
   `confirmed` is `CONVENTIONS.md`'s second requirement, a measurement showing
   at-least-parity, which is exactly what the open question governs.
3. **B2** CLI: `remember --to f32|f16`; thread the dtype into
   `run_remember_gguf`; the `.pth` / `.gguf` `--to` validation arms at
   `src/cli.rs` ~419 and ~441. Decisions already taken: honour the dtype for
   `.gguf` (7.3 made those kernels generic), accept it as vacuous for `.pth`
   (nothing is dequantised), matching `convert`'s documented `NPZ`/`.pth`
   policy.
4. **B3** `InspectOptions` builder (`InspectOptions::new().with_output_dtype()`
   plus `ParsedModel::inspect_with_options`), mirroring `RememberOptions` /
   `ConvertOptions`. Also give `InspectInfo` `#[non_exhaustive]` while it is
   open, before Phase 8 freezes its shape.
5. **B4** `to_bf16_bytes` in `src/convert.rs` must round to nearest even, not
   truncate. Decided: fix the code, not the doc.
6. **C1** `# Memory` sections and the three `peak_heap_*` tests per dtype.
7. **D1/D2** re-golden `F32` fixtures and cross-validate. Environment is ready
   (see below).
8. **D3, E1/E2, F1/F2/F3** per the plan file.

## Environment, verified 2026-08-13

Everything needed for D1 is installed; **nothing to install**.

| tool | version | for |
|---|---|---|
| Python | 3.14.0 | |
| `torch` | 2.10.0+cu130, CUDA on RTX 5060 Ti | `FP8` goldens |
| `gptqmodel` | 7.1.0 | `GPTQ` goldens (`TorchLinear`, v1→v2 conversion) |
| `autoawq` | 0.2.9 | `AWQ` goldens |
| `bitsandbytes` | 0.49.1 | `BnB` goldens |
| `numpy` / `safetensors` / `gguf` | 2.4.2 / 0.7.0 / 0.18.0 | fixture I/O |
| `pypcre` | 0.3.2 | the `pcre` shim `generate_gptq.py` needs on 3.14 Windows |

`ml_dtypes` is **absent**. Not needed for v0.7.4; it is a **Phase 8**
prerequisite (the `bfloat16` NumPy return contract).

### Two findings that shape the re-golden scripts

1. **The `BnB` `F32` golden cannot be derived from the existing one.**
   `generate_bnb.py` builds its `QuantState` with `dtype=out_dtype`, which is
   `torch.float16` for the `FP4` fixtures. Widening that f16 result to `F32` is
   **not** the canonical `F32`. Measured on this box: an f32 `QuantState` and a
   widened f16 one differ on **3811/4096 elements (93 %)**; at `BF16` width they
   differ on only 279/4096, which is why nobody noticed. The `F32` golden must
   come from a `dtype=torch.float32` `QuantState`, asked of the same canonical
   kernel.
2. **`BnB` re-goldening must run on CUDA.** The generator anchors to the GPU
   kernel deliberately: bitsandbytes 0.49's CPU kernel double-rounds and
   diverges by 1 `ULP` on ~19 % of elements. Re-goldening on CPU would silently
   re-anchor the whole family.

Expect `BnB` to be the family most likely to fail the first `F32`
cross-validation run, and treat a failure as a finding, not a test bug.

## Open decisions already made (do not re-litigate)

- Narrowing mechanism: **`write_scratch`**, not a per-element hook. `write_one`
  was implemented in full and rejected on asm evidence (`GPTQ` and `FP8`
  de-vectorised to scalar `vsubss`/`vmulss`; 4.25× and 1.46×). The reasoning is
  in `src/remember/output.rs`'s module docs so it is not re-litigated a third
  time.
- `F32` goldens: re-golden from the raw bytes already inside each committed
  `.bin` fixture, no model re-download.
- `to_bf16_bytes`: fix the code to round.
- `InspectOptions` builder rather than a parameterised `From` or an accessor.

## Still to raise with Éric

- **FAQ entries and a tutorial** for the docs pass (F1/F2). Proposal to bring:
  one new tutorial on choosing an output dtype, plus FAQ entries on which dtype
  to ask for, why `F32` output is twice the size, and why a `remember` output
  file is legitimately mixed-dtype (the passthrough policy, most likely to
  surprise).
- **ROADMAP amendments.** Four items its Phase 7.4 text does not name: the
  `convert.rs` gate (done, `bc11d0f`), `transpose_bf16` as a fifth `× 2` site
  (done), `run_remember_gguf` as a `BF16`-hardcoded duplicate (pending, B2), and
  the `MAX_OUTPUT_BYTES` gate item, which needs **rewriting rather than
  ticking** because the prediction was wrong.
- **Before v0.8.0**, unrelated to this phase: `src/lethe/bnb.rs:340` and `:921`
  still carry `// VECTORIZED: pending`, which `CONVENTIONS.md` calls a release
  blocker at the next `vX.Y.0`. One commit.
