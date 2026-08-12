# anamnesis peak-heap validation

This directory holds three `dhat-rs`-instrumented heap-assertion test
files that catch peak-memory regressions in the hot dequantisation
kernels. Phase 6.5 of the [ROADMAP](../ROADMAP.md) introduces them as
dev-only infrastructure (zero impact on the published crate) so that:

1. **Regressions are caught at commit time, not at deployment** — if
   a future refactor accidentally turns a `O(out_features)` scratch
   into `O(num_groups × out_features)`, or introduces an
   intermediate byte-serialisation buffer, the assertion fires with
   a decomposed error message identifying the regression shape.
2. **Public claims trace back to numbers** — every "peak heap is
   `output_size + O(out_features)`"-style claim in the per-function
   `# Memory` doc sections (per
   [CONVENTIONS.md §`# Memory`](../CONVENTIONS.md))
   is now backed by an assertion that runs on demand.

The tests do not ship in the published crate (`tests/` is dev-only by
Cargo convention) and do not depend on any external fixture beyond
deterministic synthetic data.

---

## File layout

| File | Kernel under test | Claim validated | Run command |
|---|---|---|---|
| `peak_heap_gptq.rs` | `dequantize_gptq_to_bf16` | `peak < output_size + O(out_features)` (NOT `O(num_groups × out_features)`) | `cargo test --release --features gptq --test peak_heap_gptq -- --ignored --nocapture` |
| `peak_heap_awq.rs` | `dequantize_awq_to_bf16` | same as `GPTQ` (identical scratch shape) | `cargo test --release --features awq --test peak_heap_awq -- --ignored --nocapture` |
| `peak_heap_bnb_dq.rs` | `dequantize_bnb4_double_quant_to_bf16` | "no intermediate byte-serialisation allocation" | `cargo test --release --features bnb --test peak_heap_bnb_dq -- --ignored --nocapture` |
| `peak_heap_gguf.rs` | `dequantize_gguf::<E>` / `dequantize_gguf_blocks::<E>` | `peak == output_size` exactly, **per output dtype**; streaming peak `== 0` | `cargo test --release --features gguf --test peak_heap_gguf -- --ignored --nocapture` |

> **`peak_heap_gguf.rs` asserts a stronger claim than its siblings.** `GPTQ` and
> `AWQ` allow `output_size + O(out_features)` of scratch; `GGUF` allows **no heap
> scratch at all**, because both kernel runners keep their `[f32; QK]` scratch
> and their block output buffer on the *stack*. Its ceiling is therefore the
> output size plus a fixed constant, deliberately not a term that scales with the
> tensor: if heap scratch that grows with the input ever appears, the assertion
> must fail rather than absorb it.
>
> It is also the only one parameterised over the **output dtype**, because since
> v0.7.3 the `GGUF` output width is caller-chosen and the `# Memory` claim is
> stated in terms of `E::BYTES`. Measured on 4096 × 11008 `Q4_K` (45 M elements):
> peak equals output exactly at `BF16` (90 177 536 B), `F16` (same) and `F32`
> (180 355 072 B, exactly twice), with **0 B** of overhead in every case, and the
> streaming entry point peaks at **0 B**.

Each file declares `dhat::Alloc` as `#[global_allocator]` for its
own test binary (`tests/<file>.rs` becomes one binary per Cargo's
auto-discovery), so the dhat allocator overhead is scoped per-binary
and does not leak into other test runs. The tests are `#[ignore]`d
so default `cargo test` skips them entirely — opt in via `--ignored`.

---

## Assertion shape — two distinct forms

### GPTQ + AWQ — ratio with `K = 5` slack

Both kernels allocate **three reused `Vec<f32>[out_features]`
scratch buffers** (unpacked weights, zeros, scales) refilled lazily
when the cached group changes — see
[src/remember/gptq.rs:355-357](../src/remember/gptq.rs) and
[src/remember/awq.rs:263-265](../src/remember/awq.rs). The assertion:

```rust
peak_heap <= output_size + K × out_features × 4    // K = 5
```

`K = 5` covers the documented `3 × out_features × 4` scratch with
~2× headroom for allocator overhead and minor future drift. Trips on:

- **Eager precomputation** — if scratch grows to `num_groups ×
  out_features × 4` (32 groups × 11008 elements × 4 bytes = 1.4 MiB
  at the layer fixture), peak exceeds the `K = 5` ceiling by ~10×.
- **Catastrophic double-allocation** — if output is allocated twice,
  peak doubles, way over.

`K = 5` does **not** catch a single extra `Vec<f32>[out_features]`
allocation at the small fixture (4 KiB extra, under the ~16 KiB
slack). The layer fixture (44 KiB extra) does catch it.

### BnB double-quant — noise tolerance with 4 KiB constant

The kernel allocates exactly two scratch buffers:
- `dequantized_absmax: Vec<f32>[num_blocks]` — recovered absmax,
  computed directly as `f32`, **never serialised to bytes**
- `Vec<f32>[block_size]` — block-iteration scratch inside the core
  dequant loop

See [src/remember/bnb.rs:386-426](../src/remember/bnb.rs). The
assertion:

```rust
expected = output_size + num_blocks × 4 + block_size × 4
peak_heap - expected <= 4 KiB                          // noise tolerance
```

The 4 KiB tolerance is **tighter** than the GPTQ/AWQ ratio form
because the claim under test ("no intermediate byte-serialisation")
is precisely a `1× scratch_bytes` regression — adding a
`Vec<u8>[num_blocks × 4]` intermediate buffer would push observed
scratch from `1×` to `2×` expected. A ratio assertion with `K ≥ 2`
would let that slide. The 4 KiB constant catches it immediately
(+65 KiB at the small fixture, +2.8 MiB at the layer fixture, both
way over).

---

## Baseline (reference machine)

Captured 2026-05-21 on the development machine. Run the same
commands locally to verify your own numbers; absolute byte counts
are machine-specific.

**Hardware / toolchain**: AMD Ryzen 9 5950X (16C @ 3.40 GHz), Windows
11 Pro x64, `rustc 1.95.0` (release build, `target-cpu=native` via
default cargo profile).

### GPTQ — `dequantize_gptq_to_bf16`

| Fixture | Peak | Output | Observed scratch | Expected scratch | Ratio |
|---|---:|---:|---:|---:|---:|
| Small (1M, 1024×1024, g128) | 2,109,440 B | 2,097,152 B | 12,288 B | 3 × 1024 × 4 = 12,288 B | **1.000×** |
| Layer (45M, 4096×11008, g128) | 90,309,632 B | 90,177,536 B | 132,096 B | 3 × 11008 × 4 = 132,096 B | **1.000×** |

### AWQ — `dequantize_awq_to_bf16`

| Fixture | Peak | Output | Observed scratch | Expected scratch | Ratio |
|---|---:|---:|---:|---:|---:|
| Small (1M, 1024×1024, g128) | 2,109,440 B | 2,097,152 B | 12,288 B | 3 × 1024 × 4 = 12,288 B | **1.000×** |
| Layer (45M, 4096×11008, g128) | 90,309,632 B | 90,177,536 B | 132,096 B | 3 × 11008 × 4 = 132,096 B | **1.000×** |

### BnB double-quant — `dequantize_bnb4_double_quant_to_bf16`

| Fixture | Peak | Output | Observed scratch | Expected scratch | Excess |
|---|---:|---:|---:|---:|---:|
| Small (1M, b64) | 2,162,944 B | 2,097,152 B | 65,792 B | 16,384 × 4 + 64 × 4 = 65,792 B | **0 B** |
| Layer (45M, b64) | 92,995,840 B | 90,177,536 B | 2,818,304 B | 704,512 × 4 + 64 × 4 = 2,818,304 B | **0 B** |

**Every kernel's observed heap allocation matches the documented
claim to the byte.** `dhat::HeapStats::max_bytes` sums live
user-side allocation sizes; allocator-internal bookkeeping is not
included.

---

## Interpreting an assertion failure

The assertions emit decomposed failure messages so the next reader
can spot the regression shape without re-running with extra logging.

### GPTQ / AWQ failure example

```
GPTQ peak heap 91234567 bytes exceeded ceiling 90397568 bytes
 (output=90177536 bytes, out_features=11008, K=5 × 4-byte scratch slack).
 Excess of 1057031 bytes suggests 24 × out_features regression —
 likely eager precomputation has crept in.
```

Read the "Excess of N bytes suggests M × out_features" line — that
`M` multiplier tells you whether the regression is one extra `Vec`
(`M = 4` total, vs the documented `3`) or full eager precomputation
(`M = num_groups × 3`). If `M ≈ num_groups × 3`, look for whether a
`for group in 0..num_groups { precompute(group) }` loop has crept
into the hot path.

### BnB-DQ failure example

```
BnB-DQ peak heap 95814144 bytes exceeded expected 92995840 bytes
 by 2818304 bytes (tolerance 4096 bytes).
 Expected: output=90177536 bytes + scratch=2818304 bytes
 (num_blocks × 4 + block_size × 4).
 Excess > tolerance suggests an intermediate byte-serialization
 regression — most likely a Vec<u8>[num_blocks × 4] crept in.
```

If `observed_excess ≈ num_blocks × 4`, look for whether the recovered
absmax is being serialised to a `Vec<u8>` somewhere before the core
dequant loop consumes it. The cleanest BnB-DQ implementation keeps
absmax as `Vec<f32>` end-to-end.

---

## Updating this README

This file holds the **latest reference numbers from one machine** —
not a history. When you produce a meaningful new baseline (after a
kernel refactor, an allocator upgrade, etc.), replace the calibration
tables above. The git history is the historical record.

If a future kernel grows new scratch buffers (e.g., a SIMD-aligned
working tensor at Phase 9), update both the kernel's `# Memory` doc
section and this README's expected-scratch formula at the same time
so the assertion threshold + documented claim stay in sync.

For broader perf-validation context — including criterion runtime
benchmarks that cover the same kernels' throughput — see
[`benches/README.md`](../benches/README.md).
