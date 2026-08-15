# Phase 7.4 handoff (updated 2026-08-15, mid-D1)

Branch `phase-7.4-remember-output-dtype`. **There are UNCOMMITTED changes in the
working tree** — Éric asked for no commits and no pushes until the consistency
pass. Do not `git checkout`/`stash` without reading this first.

## Committed so far

**14 commits ahead of `main`**, oldest first, all green at commit time:

| # | hash | what |
|---|---|---|
| 1 | `bc11d0f` | four kernel families generic over `OutputElement`; `TargetDtype::{F32,F16}`; boundary dispatch; `convert`'s v0.7.4 gate removed |
| 2 | `587fb80` | recover most of the BF16 regression the fission split introduced |
| 3 | `66bb10b` | tile BnB INT8; share `VECTOR_TILE`; record the ~23 % code-layout finding |
| 4 | `eac4793` | **`write_one` REJECTED** on asm evidence |
| 5 | `36e3c8a` | whole-model threaded bench; record the perf contradiction |
| 6 | `c65ae63` | handoff + discrepancy docs |
| 7 | `d7bd7fa` | **cross-crate inlining fix** — the root cause, and a real downstream bug |
| 8 | `a39690d` | tile GPTQ/AWQ pass 2/3 (AWQ 5 % win, GPTQ neutral) |
| 9 | `e98e090` | B4 — `to_bf16_bytes` rounds to nearest even |
| 10 | `6ce07ba` | B2 — CLI `remember --to f32\|f16` |
| 11 | `2b28ff4` | B3 — `InspectOptions`, dtype-aware size estimate |
| 12 | `80b3b1d` | C1 — peak-heap per dtype + `dhat` parallelism fix |
| 13 | `137c413` | every `VECTORIZED: pending` resolved on asm evidence |
| 14 | `33d5151` | **FP8 F32 cross-validation** (7/7 exact, first run) |

Suite at this point: **504 tests green**, clippy clean, rustdoc clean.

## UNCOMMITTED work in the tree right now

Exactly seven modified files, all part of one coherent change plus this doc:

```
 M docs/phase-7.4-handoff.md                                  <- this file
 M tests/cross_validation_gptq.rs                             <- v2 reader + F32 arm
 M tests/fixtures/gptq_reference/generate_gptq.py             <- v2 container + drift guard
 M tests/fixtures/gptq_reference/falcon3_1b_int4.bin          <- regenerated
 M tests/fixtures/gptq_reference/falcon3_1b_int8.bin          <- regenerated
 M tests/fixtures/gptq_reference/llama_3_2_1b_int4.bin        <- regenerated
 M tests/fixtures/gptq_reference/llama_3_2_1b_gptqmodel_int8.bin <- regenerated
```

**GPTQ F32 cross-validation, complete and passing (4/4).** Full suite green
(504) with these changes in the tree, so this is a committable unit whenever
Éric lifts the hold.

- `tests/fixtures/gptq_reference/generate_gptq.py` — extended to emit a v2
  `AMNQ` container (magic + version + F32 golden appended after BF16), with a
  **drift guard** that refuses to overwrite if the regenerated BF16 differs from
  the committed one. Docstring updated.
- `tests/fixtures/gptq_reference/*.bin` — 4 fixtures regenerated to v2.
  Drift guard PASSED on all four: GPTQModel 7.1.0 + torch 2.10.0 reproduce the
  committed BF16 goldens byte for byte. 77–96 % of values are not
  BF16-representable.
- `tests/cross_validation_gptq.rs` — reader accepts v2 only (asserts magic +
  version), new `compare_gptq_f32_exact` (bit equality, no tolerance).
- **Result: 4/4 pass the exact F32 comparison, first run.**

## Method that worked, reuse it for AWQ and BnB

**All 13 source models ARE present in the HF cache** (verified). That changed
the approach from the originally-agreed "re-golden from the committed .bin" to
"extend `generate_*.py` in place" — necessary for GPTQ because the fixture does
not store `sym` / `desc_act` / `checkpoint_format`, which `TorchLinear` needs.
Tell Éric this was a means-level change; it produces a strictly more canonical
fixture. FP8 used the .bin route (its `regolden_f32.py` still exists and works).

Per family the recipe is:

1. Add `FIXTURE_MAGIC` (`AMNF`/`AMNQ`/`AMNA`/`AMNB`) + `FIXTURE_VERSION = 2`.
2. Capture the reference's f32 result **before** any `.to(bfloat16)`.
3. Print the not-BF16-representable percentage (proves the comparison has teeth).
4. Drift guard: compare regenerated BF16 against committed, refuse on mismatch.
5. Write magic, version, existing header fields, `bf16_len`, `f32_len`, payloads.
6. Rust reader: assert magic + version, add `expected_f32`, add an exact
   bit-equality comparison whose failure message says this is NOT a tolerance
   question.

## Status 2026-08-15 (end of session 2)

**Everything through F2 is done.** All ten ROADMAP Phase 7.4 bullets are ticked
and the phase has an Outcome section. Suite: **509 tests green**, clippy clean
across 12 feature combos plus MSRV 1.88, rustdoc clean across 13 combos.

Done this session, beyond the list below: AWQ + BnB `F32` cross-validation
(BnB found a real 1-`ULP` kernel defect in `INT8`, fixed); D3; E1; E2
(Experiments 14 + 15); the consistency pass, brought forward at Éric's request
(three malformed/missing `VECTORIZED` annotations, plus an `InspectInfo::Display`
bug that labelled an `F32` estimate `(BF16)`); item 6 docs; F1/F2.

**Only F3 remains**: the release gauntlet (stable + MSRV 1.88, `cargo publish
--dry-run`, the two packaging checks) and the tag. Still **no commits and no
pushes** on Éric's standing instruction.

## Still to do, in order (historical, session-2 plan)

1. **AWQ** — `tests/fixtures/awq_reference/generate_awq.py` + `cross_validation_awq.rs`.
   Magic `AMNA`. Reference is AutoAWQ `dequantize_gemm`.
2. **BnB** — the hard one. Magic `AMNB`. Two traps, both verified this session:
   - The F32 golden **must** come from a `QuantState(dtype=torch.float32)`, not
     from widening the existing f16-QuantState result. Measured: they differ on
     **93 %** of elements (vs 7 % at BF16 width, which is why nobody noticed).
   - Re-goldening **must run on CUDA** (RTX 5060 Ti present). bitsandbytes 0.49's
     CPU kernel double-rounds and diverges by 1 ULP on ~19 % of elements;
     `generate_bnb.py` deliberately anchors to the GPU kernel.
   - Expect this family to be the one that fails first. Treat a failure as a
     finding, not a test bug.
3. **D3** — determinism across thread counts {1,2,4} *per dtype*, plus
   end-to-end dtype tests on `remember` (mirror the three in `src/convert.rs`).
4. **E1** — add `remember_f32_whole_model` to `benches/convert.rs` beside the
   existing `remember_bf16_whole_model`. ADD, never rename (CodSpeed history).
5. **E2** — `docs/perf-experiments.md` entry. Numbers are settled now; see
   `docs/phase-7.4-bf16-perf-discrepancy.md` (RESOLVED) for the honest table.
6. **Docs check-in with Éric** — I owe a proposal before writing prose:
   one tutorial on choosing an output dtype; FAQ entries on which dtype to ask
   for, why F32 output is twice the size, and why a `remember` output file is
   legitimately mixed-dtype (the passthrough policy — most likely to surprise).
7. **F1/F2** — python-interop.md restated, README, cli-reference.md (several
   "bf16-only until v0.7.4" lines now live), CHANGELOG `[Unreleased]` opened,
   ROADMAP amendments **including rewriting the A5 item** (the
   `MAX_OUTPUT_BYTES` gate correctly stays; v0.7.3's prediction was wrong).
8. **STOP at the consistency pass.** Éric's instruction: proceed to it, not
   through it. No commits, no pushes before then.

## Settled decisions — do not re-litigate

- **Re-goldening extends `generate_*.py` in place** rather than deriving from
  the committed `.bin`. Confirmed by Éric 2026-08-15 as the standing rule. Used
  for GPTQ, AWQ and BnB; FP8 keeps its `.bin` route (`regolden_f32.py`) because
  it already worked. This is not merely convenient: re-goldening from the
  `.bin` **could not** have produced the BnB F32 golden at all, because that
  needs a `QuantState(dtype=torch.float32)` the fixture never stored.
- **Docs plan approved** (item 6): one tutorial (*Choosing an output dtype*)
  plus four FAQ entries — which dtype to ask for, why F32 output is twice the
  size, why a `remember` output file is legitimately mixed-dtype (the
  passthrough policy), and whether anamnesis is still bit-exact against
  `PyTorch` at F32 (where the BnB INT8 finding is told honestly).

- `write_scratch`, not `write_one`. Rejected on asm evidence (GPTQ 4.25×, FP8
  1.46× de-vectorised to scalar). Recorded in `src/remember/output.rs`.
- `MAX_OUTPUT_BYTES` stays `gguf`-gated.
- `to_bf16_bytes` rounds to nearest even.
- `InspectOptions` builder, mirroring RememberOptions/ConvertOptions.
- Perf story is RESOLVED and is a net **win**: FP8 2.2× faster, others
  1.04–1.18× slower, whole-model FP8 1.8× faster. The root cause of the earlier
  confusion was a cross-crate inlining regression (`d7bd7fa`), a real bug that
  made downstream callers ~2.9× slower.

## Environment

Python 3.14, torch 2.10.0+cu130 (CUDA, RTX 5060 Ti), gptqmodel 7.1.0,
autoawq 0.2.9, bitsandbytes 0.49.1, numpy 2.4.2, safetensors 0.7.0, pypcre.
`ml_dtypes` absent — Phase 8 prerequisite only.

## Verification one-liners

```powershell
cargo fmt; cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
$env:RUSTDOCFLAGS="-D warnings"; cargo doc --all-features --no-deps; $env:RUSTDOCFLAGS=$null
```

Run clippy **before** committing, not alongside: a missing `// CAST:`
annotation on a `usize → f64` in test code slipped into one commit this session.
