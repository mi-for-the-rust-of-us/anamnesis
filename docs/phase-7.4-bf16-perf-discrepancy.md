# Phase 7.4: the BF16 performance discrepancy (RESOLVED)

**Status:** **resolved 2026-08-15.** Root cause was a cross-crate inlining
regression that v0.7.4 introduced and that the benchmark itself was measuring.
Jump to [Resolution](#resolution).
**Branch:** `phase-7.4-remember-output-dtype`.
**Opened:** 2026-08-13. **Closed:** 2026-08-15.

---

## Resolution

**The `#[inline]` on the new `*_to_bf16` wrappers made the test crate
instantiate the generic kernel locally, where the private per-element helpers
could not inline.**

Before v0.7.4, `dequantize_fp8_to_bf16` was a plain non-generic `pub fn`: one
lib-compiled symbol, fully optimised, called opaquely from outside. v0.7.4 made
it an `#[inline]` wrapper around `dequantize_fp8::<Bf16Out>`. An external caller
now inlines the wrapper and instantiates the generic in **its own** crate, where
`e4m3_to_f32_bits`, `e4m3_to_scaled_f32`, `f32_bits_to_bf16_bits` and
`unpack_gptq` were private, non-`#[inline]`, and therefore opaque cross-crate
symbols. The result was a function call per element.

`remember` never suffered this: it calls the generic from inside the library
where everything inlines. That is the entire discrepancy. The isolated bench was
measuring external-caller codegen; the whole-model bench was measuring
production codegen; both were correct about different things.

**The fix** is `#[inline]` on those four helpers. Measured effect on the
isolated arms, same fixture, same process:

| arm | without the fix | with the fix |
|---|---:|---:|
| `kernel_fresh_alloc` (4 × 4096²) | 1.934 ms/Melem | **0.672** |
| `kernel_single_call` | 2.013 | **0.731** |

**This was a real production bug, not just a benchmark artefact.** Any
downstream crate calling `anamnesis::dequantize_*_to_bf16` would have got the
un-inlined version, roughly 2.9× slower than v0.7.3 for the same call.

### The honest v0.7.4 numbers, after the fix

Isolated kernels, baseline binary at `6697599`, interleaved, min of 4:

| family | before | after | ratio |
|---|---:|---:|---:|
| `fp8_fg` 4096 × 11008 | 65.60 ms | 29.67 | **0.452× (2.2× faster)** |
| `fp8_fg` 4096 × 4096 | 24.50 | 10.87 | **0.444×** |
| `bnb_nf4` | 25.26 | 26.72 | 1.058× |
| `bnb_int8` | 15.23 | 16.78 | 1.102× |
| `awq_int4` | 28.22 | 31.43 | 1.114× |
| `gptq_int4` | 19.96 | 22.88 | 1.146× |

Whole-model `remember`, 4 × 4096² fine-grained `FP8`:

| threads | before | after | ratio |
|---|---:|---:|---:|
| 1 | 120.61 ms | 66.70 | 0.553× |
| 4 | 49.93 | 38.57 | 0.772× |

These are now mutually consistent: `FP8` gains 2.2× at the kernel, diluted to
1.8× end to end by serialization. The `VECTOR_TILE` restructure vectorises
better than the old per-element fused loop it replaced. The other three families
pay a genuine 1.06–1.15×, which is the honest cost of the
arithmetic/narrowing split and is the number that belongs in
`docs/perf-experiments.md`.

### Lesson for the next ad-hoc bench

An ad-hoc bench in `tests/` is an **external** crate. Once a public entry point
becomes a thin generic wrapper, that bench stops measuring what the library
executes internally. Either mark every hot-loop helper `#[inline]`, or benchmark
through a whole-model entry point that exercises the internal call path.

---

## Original investigation (kept for the record)

---

## The question in one line

Two benchmarks of the same two binaries disagree about whether v0.7.4's
arithmetic/narrowing split makes the `BF16` default path slower or faster, and
the disagreement is a factor of ~1.9.

## The numbers

Idle machine, `RUSTFLAGS="-C target-cpu=native"`, release, binaries built once
and then run **alternately** (before, after, before, after, ...), min of 3 to 5
rounds. Baseline binary built at `6697599`, the commit before the phase started.

### Isolated kernel, `bench_bf16_all_families`

| fixture | before | after | ratio |
|---|---:|---:|---:|
| `fp8_fg` 4096 × 11008 | 1.480 ms/Melem | 1.561 | 1.055× |
| `fp8_fg` 4096 × 4096 | 1.466 | 1.547 | 1.055× |
| `bnb_int8` 4096 × 11008 | 15.02 ms | 16.70 | 1.112× |
| `bnb_nf4` 45 M elements | 25.49 | 26.25 | 1.030× |
| `gptq_int4` 4096 × 11008 | 19.70 | 22.03 | 1.118× |
| `awq_int4` 4096 × 11008 | 28.48 | 30.18 | 1.060× |

Stable across two shapes for `FP8`, so **shape is not the variable**.

### Whole model, `bench_remember_whole_model_threaded`

4 × 4096 × 4096 fine-grained `FP8` (67 M elements) through
`remember_to_bytes_with_options`.

| threads | before | after | ratio |
|---|---:|---:|---:|
| 1 | 120.31 ms | 67.32 | **0.560×** |
| 4 | 50.00 | 38.20 | **0.764×** |

Reproduced on a freshly rebuilt, hash-verified binary pair.

## The tell

End to end runs at **1.003 ms/Melem**. The isolated kernel it *contains* runs at
**1.547 ms/Melem**. A path that does strictly more work (parse, per-tensor
dispatch, `build_views`, `safetensors::serialize` into one contiguous `Vec`)
cannot be faster per element than one of its own components.

So at least one bench is not measuring what its name says. The leading
hypothesis is that **`bench_bf16_all_families` is dominated by output-buffer
allocation**: each iteration calls a `dequantize_*` entry point that does
`vec![0u8; n]` for a 90 MB output, so every iteration pays fresh pages from the
OS plus first-touch faults. The whole-model bench allocates too, but in 33 MB
pieces that the allocator can recycle across iterations.

If that is right, the ~5 % "regression" chased for most of 2026-08-13 is a
property of the harness, not of the code.

## What is already ruled out

- **Machine noise.** Spreads are 0.4–1.6 % on an idle box. The one wild session
  (`fp8` reading 1.43×, a non-monotonic tile sweep) was traced to a background
  job Éric was running; re-measured clean afterwards.
- **Tensor shape.** 4096 × 11008 and 4096 × 4096 both give 1.055× isolated.
- **Correctness.** `BF16` output is byte-identical. All 498 lib tests and every
  cross-validation suite (`FP8`, `GPTQ`, `AWQ`, `BnB`, `GGUF`, convert, pth,
  npz, ollama, safetensors) pass unchanged, and those compare against the
  canonical libraries' own bytes.
- **Stale binaries.** The last comparison rebuilt both from verified trees and
  checked their hashes differ.

## What is NOT ruled out

1. **Allocation/page-fault dominance in the isolated bench** (leading
   hypothesis, see above).
2. **Whole-program code layout.** Separately established this session: `FP8`
   with byte-identical source measured 1.060× in one build and 1.308× in
   another, the only difference being that an unrelated module changed size.
   ~23 % swings from layout alone. This is real and is recorded in `66bb10b`.
   It bounds the precision of any per-kernel claim on this hardware.
3. **Something genuinely faster in the v0.7.4 orchestration.** Not identified in
   the diff, but not excluded either.

## How to settle it tomorrow

In rough order of cost:

1. **Take allocation out of the isolated bench.** Add a variant that allocates
   the output buffer once outside `time_best_of_5` and calls a
   `*_into(&mut out)`-shaped entry point, or simply subtract a
   `vec![0u8; n]`-only control loop at the same size. If the 1.055× collapses,
   the hypothesis is confirmed and the isolated numbers are discarded.
2. **Profile one iteration of each bench** (Windows: WPA / `xperf`, or
   `superluminal` if available) and compare time in the kernel symbol versus
   time in `RtlAllocateHeap` / page-fault handling.
3. **CodSpeed.** Push the branch; `benches/dequant.rs` already covers all four
   families at `BF16` with history from *before* these kernels were touched, and
   `benches/convert.rs` has `remember_bf16_whole_model` at 1 and 4 threads. The
   macro runners give a directly comparable baseline. This is plan step E1.

## Reproduce

```powershell
# build both binaries (the ad-hoc bench uses only pre-v0.7.4 API on purpose,
# so the identical file compiles on the baseline commit)
$env:RUSTFLAGS = "-C target-cpu=native"
cargo test --release --all-features --test bench_dequant_adhoc --no-run
# ...copy the exe aside, git checkout 6697599, copy tests/bench_dequant_adhoc.rs
#    back in, rebuild, copy that exe aside as the baseline, return to the branch

# then run them alternately, never A-then-B:
.\before.exe bench_bf16_all_families            --nocapture --ignored
.\after.exe  bench_bf16_all_families            --nocapture --ignored
.\before.exe bench_remember_whole_model_threaded --nocapture --ignored
.\after.exe  bench_remember_whole_model_threaded --nocapture --ignored
```

**Method notes, learned the hard way:**

- Alternate the binaries. A-then-B on this host gave inverted results.
- Take the **min** across rounds, not the median of one round.
- Check the machine is idle first (`Get-CimInstance Win32_Processor`).
- Rebuilding one module changes another module's timing by up to 23 %. Never
  attribute a sub-20 % per-kernel difference to the source you edited without
  an independent control.

## Why it matters, and why it does not block

The phase claims **exactness**, not speed, and `CLAUDE.md` scopes the "no
measured win, no commit" rule to perf-claim commits. `feedback-capability-before-speed`
says a measured slowdown does not block a capability feature. So v0.7.4 can
ship either way; what it cannot do is publish a number this investigation shows
is unreliable.
