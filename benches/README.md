# anamnesis benchmarks

This directory holds `criterion`-driven runtime benchmarks for the
crate's hot paths. Phase 6.5 of the ROADMAP introduces them as
dev-only infrastructure (zero impact on the published crate) for
two purposes:

1. **Regression detection** — `criterion` baselines the first run,
   then statistical-significance-tests every subsequent run against
   that baseline. Significant slowdowns are flagged in the report.
2. **Public credibility** — every "anamnesis is X× faster than
   <tool>" claim in `README.md` and `docs/perf-experiments.md` should
   trace back to a number measurable from this directory.

The benches do **not** ship in the published crate (`[[bench]]`
entries in `Cargo.toml` are dev-only by Cargo convention) and do
**not** depend on any external runtime fixture beyond what is already
checked into `tests/fixtures/`.

---

## File layout

| File | Scope | Run command |
|---|---|---|
| `dequant.rs` | Dequantisation kernels (decode side) — `FP8` per-tensor / fine-grained, `GPTQ` `INT4`, `AWQ` `INT4`, `BnB` `NF4`, `BnB` `INT8`, `GGUF` `Q4_K`, plus the real-world `GGUF` `Q8_0` slice from the `Ollama`-distributed `llama3.2:1b` fixture | `cargo bench --features gptq,awq,bnb,gguf --bench dequant` |
| `parsing.rs` | Header / metadata-only parses for the four supported tensor formats vs an `fs::read` baseline at the same fixture | `cargo bench --features npz,pth,gguf --bench parsing` |
| `convert.rs` | **Whole-model, multi-threaded** paths — quantised `GGUF` → safetensors `convert()`, and `FP8` safetensors → `BF16` `remember_to_bytes` — each measured at **1 and 4 threads** | `cargo bench --features gguf --bench convert` |

### Why `convert.rs` benchmarks each path twice

`dequant.rs` times one kernel on one tensor and `parsing.rs` times a
header parse; neither spawns a thread. So until Phase 7.2 the ~3–4×
that Phase 7 shipped had **no continuous regression guard** — it was
only ever verified by hand.

The `threads_1` / `threads_4` pair is the point. A change that
silently serialises the dispatch keeps every absolute time plausible
but collapses the ratio, which a single-budget benchmark would report
as a uniform slowdown indistinguishable from a slower kernel.

Two caveats worth knowing before reading the numbers:

- **`convert_gguf_to_safetensors` includes the output file write.**
  `convert()` is the only public entry to the `GGUF` reader, and it
  writes a file. That write is a large constant both budgets pay, so
  the visible ratio (~1.25×) understates the dequant stage's own
  ~1.9×. The stage-isolated figure comes from the `#[ignore]`d
  `src/convert.rs::hub_scaling_bench`, which can reach the private
  `read_hub` — a bench target cannot. See
  [`docs/perf-experiments.md`](../docs/perf-experiments.md)
  Experiment 12.
- **`remember_bf16_whole_model` touches no filesystem** (it returns a
  `Vec<u8>`), so it is the cleaner of the two and the one that tracks
  Phase 7's headline number.

> **If you retune `parallel::MIN_PARALLEL_BYTES`, re-check the fixture
> sizes in `convert.rs`.** The dispatch runs sequentially below that
> threshold, so an undersized fixture makes *both* budgets take the
> sequential path — the benchmark then looks healthy while measuring
> nothing. An early draft of the file did exactly that.

**Phase 7.5 forecast** — when the encode-side kernels land (`FP8`,
`GGUF` legacy / K-quants / IQ / TQ / MXFP4), a new `encode.rs` will
sit alongside `dequant.rs` rather than letting one file balloon past
~600 LOC. See the [ROADMAP](../ROADMAP.md) Phase 7.5 entry.

---

## Running

### Full statistical run (default)

```sh
cargo bench --features gptq,awq,bnb,gguf,npz,pth
```

Runs **both** bench files (`dequant.rs` + `parsing.rs`) with
`criterion`'s default settings: 100-sample groups, ~5 s measurement
time per group, plus warm-up. Total wall-clock: **~10–15 minutes**
on the reference machine below. Reports land in
`target/criterion/`; open `target/criterion/report/index.html` for
the HTML index.

### Quick sanity / smoke

```sh
cargo bench --features gptq,awq,bnb,gguf,npz,pth -- --quick
```

`--quick` forces `criterion` into reduced-sample mode (≤10 samples,
≤2 s per group). Total wall-clock: **~30–60 seconds**. Output is
noisy and not statistically valid; use this only to confirm the
benches **execute** end-to-end after a code change, then run the
full statistical version for an actual baseline.

### Run a single group

`criterion` accepts substring filters as positional arguments:

```sh
cargo bench --features bnb --bench dequant -- dequant_bnb_nf4
cargo bench --features gguf --bench dequant -- gguf_q8_0_ollama
cargo bench --features gguf --bench parsing -- inspect_gguf
```

---

## Baseline (reference machine)

Captured 2026-05-21 on the development machine. Run the same
commands locally to get **your** baseline; absolute numbers are
machine-specific and should not be compared across machines.

**Hardware**
- CPU: AMD Ryzen 9 5950X (16 cores @ 3.40 GHz)
- OS: Windows 11 Pro x64

**Toolchain**
- `rustc 1.95.0 (2026-04-14)` (release build, `target-cpu=native` via
  default cargo profile + `rustflags` not set explicitly — see note
  on `target-cpu=native` below)

> **Note on `target-cpu=native`**: this project's CI builds *do not*
> set `target-cpu=native` (CI builds run on Ubuntu without
> CPU-specific code generation). The numbers below were taken with
> the default release profile, so they reflect what CI would produce
> if it ran the benches (it currently does not — `cargo bench` is
> developer-driven, not CI-driven). To get the maximum-throughput
> numbers the README claims for dequant kernels, set
> `RUSTFLAGS='-C target-cpu=native'` before invoking `cargo bench`.

### Dequant — synthetic `4096 × 11008` layer

| Kernel | Median time | Throughput |
|---|---:|---:|
| `dequant_fp8_per_tensor` | 41.1 ms | 1.10 Gelem/s |
| `dequant_fp8_fine_grained` | 68.6 ms | 657 Melem/s |
| `dequant_gptq_int4` (g128) | 22.7 ms | 1.99 Gelem/s |
| `dequant_awq_int4` (g128) | 43.9 ms | 1.02 Gelem/s |
| `dequant_bnb_nf4` (b64) | 37.6 ms | 1.20 Gelem/s |
| `dequant_bnb_int8` | 25.6 ms | 1.76 Gelem/s |
| `dequant_gguf_q4_k` | 22.5 ms | 2.01 Gelem/s |

These are `--quick` numbers (`criterion --quick`, ~10 samples each).
The full statistical run produces tighter confidence intervals but
the medians shift by less than measurement noise. Refresh from the
full run before quoting in a release PR.

### Dequant — real-world `Ollama` fixture

| Kernel | Fixture | Median time | Throughput |
|---|---|---:|---:|
| `dequant_gguf_q8_0_ollama` | `llama3.2:1b` `blk.0.attn_q.weight` slice (65 536 elements, 68 KiB Q8_0) | 16.7 µs | 3.92 Gelem/s |

This is the same slice the `cross_validation_ollama` test validates
bit-exactly against `gguf-py`'s reference dequant — so the
throughput number is paired with a correctness guarantee on real
`Ollama` distribution data.

### Parsing — header-only throughput

| Bench | Fixture | Median time | Throughput |
|---|---|---:|---:|
| `baseline_fs_read` (divisor) | Synthetic safetensors, 128× F32 [4096] | 1.82 ms | 4.30 GiB/s |
| `parse_safetensors_header` | Same fixture, header-only | 193 µs | 40.6 GiB/s* |
| `inspect_npz` | Synthetic `.npz`, 128 F32 [4096] arrays | 1.56 ms | 5.02 GiB/s |
| `inspect_pth` | `algzoo_rnn_small.pth` (~2 KB) | 167 µs | (small fixture; latency is the real metric) |
| `inspect_gguf` | Synthetic `.gguf`, 128× F32 [4096] | 97.8 µs | 79.9 GiB/s* |

\* Throughput numbers marked with an asterisk are misleading at face
value: header-only parses do **not** read the full tensor-data
section, but `criterion`'s `Throughput::Bytes(file_size)` divisor
uses the full file size. The apparent "throughput" therefore looks
much higher than `fs::read`'s baseline. The honest metric for
header-only parses is the absolute median time, not the throughput;
the README cites both.

---

## How to interpret regressions

`criterion` writes baselines into `target/criterion/<group>/base/`
on first run. Every subsequent run compares against the last
baseline:

- **No change** — within the noise floor (`criterion` reports
  "Performance has not regressed").
- **Improvement** — a green "Performance has improved" line.
- **Regression** — a red "Performance has regressed" line with the
  p-value of the Welch t-test.

If a regression shows up:

1. **Verify it on a clean run** — rerun without other CPU-bound
   processes in the background.
2. **Find the change** — the last commit that modified the kernel,
   or one of its dependencies (`half`, `float8`, `safetensors`).
3. **Decide** — accept the regression (and re-baseline with
   `cargo bench -- --save-baseline new-base`) or revert / fix.

A regression in `dequant_*` paired with a passing
`cross_validation_*` test means the kernel is still correct but
slower — which is the case for several of the recent loop-fission
refactors documented in `docs/perf-experiments.md`. Keep both
artefacts in mind when reading a regression message.

---

## Updating this README

This file holds the **latest reference numbers from one machine** —
not a history. When you produce a meaningful new baseline (after a
SIMD pass, a refactor, a `criterion` upgrade, etc.), replace the
tables above. The git history is the historical record.

For phase-defining performance claims (the kind that show up in the
crate's main `README.md`), update both this file and the README at
the same time so the two stay in sync.
