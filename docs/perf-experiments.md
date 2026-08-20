# Performance & Correctness Experiments — Tested and Rejected

This file is the **case-study log of perf or correctness hypotheses that were
tested and either rejected, partially confirmed, or contradicted by measurement**.
It exists so future audits and reviews don't re-propose the same ideas without
first reading what already happened. The primary scope is perf (Experiments 1–6);
Experiment 7 onwards extends the template to correctness-invariant changes that
were initially framed in absolute terms ("impossible") and only narrowed after
empirical observation on a real fixture — the same "measure before claiming"
discipline applies.

The binding rule for any new perf-claim commit lives in [`CLAUDE.md`'s
Performance Changes section](../CLAUDE.md). This file is the historical record
backing it.

## Why this file exists

In late April 2026 a multi-finding "algorithmic-weakness audit" was run against
the crate. Several findings were framed in absolute-sounding terms ("saves
~30 % on Gemma Scope", "saves ~10 % on every dequant kernel") but turned out to
be wrong in direction or much smaller than claimed once measured on real
fixtures and real hardware. After the second consecutive revert
([commit `5f2632b`](../README.md), then a never-committed FP8 refactor),
the project adopted the rule: **measure on a real fixture before committing any
perf-claim change**. This file catalogs what's been tested.

## Experiment index

| # | Experiment | Verdict | Commit / status |
|---|---|---|---|
| 1 | NPZ `read_array_data` memset elimination | **Regressed −33 %** | Committed as `67d6db0`, reverted in `5f2632b` |
| 2 | FP8 per-tensor chunked extend | **Regressed −23 %** | Never committed (this session, branch is clean) |
| 3 | v0.4.0 GGUF refactor re-validation | **Split: Q4_0 wins ~8 %, Q8_0 loses ~6 %** | Re-measurement only — current code unchanged |
| 4 | `parse()`: `fs::read` → `memmap2::Mmap` | **~3000× faster on 11 GiB safetensors** | Shipped |
| 5 | `inspect_gguf_from_reader`: internal `BufReader<R>` | **~52× faster on `File` substrate, mmap parity** | Shipped |
| 6 | `inspect_pth_from_reader` vs `parse_pth(path).inspect()` vs `torch.load` | **Reader 1.36× mmap median across 6 960 AlgZoo files; mmap 4.07× / reader 2.99× faster than `torch.load`** | Shipped |
| 7 | Sign-of-zero preservation rule (`BnB FP4` decode tweak) — *correctness experiment* | **"Impossible" was conditional, not absolute** — narrow tweak recovers byte-exact round-trip on 0.2 % of FP4 elements with no NF4/INT8 side-effect | Shipped commits `a5c452d` / `24cba42` / `ab4e735` (v0.5.0) |
| 8 | Vendored `ZIP` reader: container-metadata footprint vs `zip::ZipArchive::new` | **337 → 41 B/entry resident, 8.07×** (3.12× peak) on a 50 001-entry archive — projected ~12×, ceiling was ~8.4× | Shipped (v0.6.7, Phase 6.12) |
| 9 | `convert` copy-elimination pass (avoid hub-sized copies + one `O(P·N)` scan) | **Confirmed: peak −39 % (bnb), −25 % (gguf KV), −49.6 % (npz)** — each drop equals exactly the one eliminated copy | Shipped (v0.6.9, Phase 6.14) |
| 10 | Phase 7 SIMD exhaustion — is explicit AVX2 worth it anywhere in the dequant kernels? | **No. Bit-exact hand-written AVX2 `f32x8_to_bf16x8` gains 1.02×/1.04× (null) — writer is bandwidth-bound. FP8 already auto-AVX2 yet compute-bound; GPTQ/AWQ arithmetic already vectorizes; slowest kernels (IQ2/IQ3) are gather-bound (AVX2 gather slow on Zen 3). Compiler already captures the SIMD; limits are memory bandwidth + codebook gathers. Real headroom = multi-threading (single-threaded on 16 cores)** | SIMD exhausted with evidence, no product `unsafe` shipped (branch `phase-7-cpu-simd`); bench harness kept as proof; next = multi-threaded dequant |
| 11 | Phase 7 multi-threaded dequant — prototype scaling (`std::thread::scope`) | **Real but bounded, ~3–4× (not core-count-linear): FP8 (memory-heavy) 2.9×, IQ3_S (compute/gather-heavy) 4.0× at 16 threads — both plateau (bandwidth + all-core-clock ceilings). Per-thread `Vec` alloc caps FP8 at 2.2×; the disjoint `split_at_mut` output pattern (no alloc, no zero-fill) recovers it to 2.9×** | Prototype (branch `phase-7-cpu-simd`) — confirms multi-threading is the real lever and that the disjoint-output-slice pattern matters; corrects an earlier "4–16×" over-estimate |
| 13 | Phase 7.3 caller-chosen output dtype — what `F32` and `F16` cost | **A capability, not a perf claim: `F32` is *expected* to be slower and is. Kernel level 1.79× slower than `BF16` against 2.00× of output bytes, so the cost is the doubled write and nothing else. End to end 1.54×/1.61× (1/4 threads), less than the kernel figure because fixed parse cost does not scale. The Phase 7.2 threading ratio shifts 1.19× → 1.14×, as a more bandwidth-bound path should. `BF16` default did not regress (`p = 0.41`, `p = 0.13`). All three writers vectorise: `Bf16Out` 8-wide AVX2, `F32Out` 8-wide stores, `F16Out` 4-wide F16C. Peak heap equals output exactly at every width, streaming peak is 0 B** | Shipped (v0.7.3, Phase 7.3) |
| 14 | Phase 7.4 `remember`-path output dtype — what the arithmetic/narrowing split cost, and a bench that was measuring the wrong crate | **Two findings, one of them a real production bug. (a) The isolated `tests/` bench was measuring *external-caller* codegen: making `dequantize_*_to_bf16` an `#[inline]` generic wrapper let a downstream crate instantiate the kernel locally, where four private helpers could not inline — ~2.9× slower for every downstream caller, invisible to the whole-model bench. `#[inline]` on those four helpers fixed it (1.934 → 0.672 ms/Melem). (b) Post-fix the split is a net **win**: `FP8` 2.2× faster at the kernel, 1.8× whole-model; `BnB`/`AWQ`/`GPTQ` pay an honest 1.06–1.15×. Separately: whole-program code layout alone moves a kernel timing up to **23 %** on this host, which bounds every sub-20 % per-kernel claim made on it** | Shipped (v0.7.4, Phase 7.4) |
| 15 | Phase 7.4 `BnB` `INT8` association fix — bit-exactness at `F32` versus one packed multiply | **A correctness fix with a measured price, kept anyway. `w × (SCB/127)` and `bitsandbytes`' `(w × SCB) × (1/127)` are the same real number and the same `BF16`, but differ by 1 `ULP` on **26.9 %** of elements at `F32` — which is why 0/65536 `BF16` cross-validation never caught it. Matching the canonical association costs **16.82 → 17.48 ms** at 4096 × 11008 (1.04×, min of 10 interleaved rounds) for one extra `vmulps` per 8 lanes. Not recoverable by hoisting: `w × (SCB × c)` is a *third* association, measured **worse** (17299/65536 mismatches)** | Shipped (v0.7.4, Phase 7.4) |
| 12 | Phase 7.2 `GGUF`-reader parallelisation — product scaling, `MIN_PARALLEL_BYTES` calibration, and a `gguf-py` baseline | **Stage-isolated `read_hub` 1.90×/1.95× at 4 threads (plateau ~2.1× at 16) — lands at Experiment 11's *`Vec`-per-tensor* ceiling (2.2×), not its disjoint-slice 2.9×, because the hub is a `Vec` per tensor by construction. End-to-end `convert()` only 1.26×/1.36× locally, and **0.99× on CodSpeed macro runners** (see the Experiment 14 postscript) — the output write is ~60 % of the wall clock (Amdahl) and is slower still on CI storage, so treat the end-to-end threading figure as host-dependent rather than a property of the code. Pool cost measured at 236 µs (4 workers, Windows) → break-even ~0.5 MiB, threshold set to 4 MiB. vs `gguf-py` 0.18.0: 17.3–27.9× single-threaded, 33.8–52.9× at 4** | Shipped (v0.7.2, Phase 7.2) |
| 16 | Phase 7.6 `GGUF` `remember` moved out of the CLI — what the duplicated sequential path cost | **A duplication, not an optimisation: `amn remember model.gguf` ran a 121-line CLI transcription of `convert`'s reader that was sequential and silently ignored `--threads`, while `convert --to safetensors` on the same file ran the threaded library path and produced `SHA-256`-identical bytes. Routing the CLI at the library gives **1.24×** on `SmolLM2-135M-Q6_K` (241 → 194 ms) and **2.23×** on `Qwen2.5-1.5B-IQ2_M` (5854 → 2631 ms), medians of 5, `target-cpu=native`, output byte-identical before and after. The gap widens with model size because the fixed parse/write cost amortises, which is also why the small fixture understates it** | Shipped (v0.7.6, Phase 7.6) |

---

## Experiment 1 — NPZ `read_array_data` memset elimination

**Audit finding:** "`vec![0u8; data_bytes]` zero-inits the buffer immediately
before `read_exact` overwrites every byte — pure dead work. Switching to
`Vec::with_capacity(data_bytes)` + `reader.take(data_bytes).read_to_end(...)`
should save ~30 % of the parse time on Gemma Scope `params.npz` (302 MB)."

**Method:** [`tests/bench_npz_adhoc.rs`](../tests/bench_npz_adhoc.rs),
best-of-5 release-mode median, target-cpu=native, warmed FS cache. Compared
the two versions by `git checkout`-ing only `src/parse/npz.rs` between runs.

**Result:**

| Variant | Median | Range (min/max) |
|---|---|---|
| pre-#4 (`vec![0u8;n]` + `read_exact`) | **82.9 ms** | 82.2–83.2 (σ≈0.4) |
| post-#4 (`Vec::with_capacity` + `take().read_to_end`) | **110.8 ms** | 104.3–131.8 (σ≈11) |

A **+33 %** regression, opposite direction from the audit's prediction.

**Why the prediction was wrong:**

- A SIMD-optimised memset on a fresh allocation runs at ~25 GB/s on modern x86,
  so `vec![0u8; 302_000_000]` costs ~10 ms — not the ~25 ms the audit implied.
- `read_to_end` reads in ~8 KiB chunks via `read_buf`; for 302 MB that's ~37 000
  `read` syscalls vs the **single** `read_exact` syscall the old code issued.
  Even with `Vec::with_capacity` pre-allocating exactly the right size (so no
  reallocations), the iteration overhead dominates and swamps the memset
  saving.

**Disposition:** reverted in [`5f2632b`](../CHANGELOG.md). The full pre/post
numbers and analysis are preserved in that commit's message.

**Re-attempting this requires:** a safe-Rust replacement that beats
`read_exact` over a pre-allocated buffer. The only mechanism that would work is
`unsafe { buf.set_len(n) }` + `read_exact`, which requires amending
[`CONVENTIONS.md`](../CONVENTIONS.md)'s accepted-`unsafe` table. Not justified
for a single read site that saves ~10 ms.

---

## Experiment 2 — FP8 per-tensor chunked extend

**Audit finding:** "`vec![0u8; out_byte_len]` in
`dequantize_per_tensor_fp8_to_bf16` is dead work; the v0.4.0 GGUF refactor
saved ~10–15 % on `Q8_0`/`Q4_0` with the same change." Predicted ~10 % win on
FP8 per-tensor.

**Method:** [`tests/bench_dequant_adhoc.rs`](../tests/bench_dequant_adhoc.rs)
`bench_fp8_per_tensor`. 4096 × 11008 = 45 M FP8 elements, ~90 MB BF16 output
(typical Llama-class FFN layer). Best-of-5 release-mode median.

**Replacement design:** Chunked extend with a 2048-element stack scratch
buffer ([CONVENTIONS.md](../CONVENTIONS.md) SIMD-friendly loop rules
preserved: `chunks_exact` outer loop, vectorisable inner zip into
`[u8; 4096]`, single `extend_from_slice` per chunk).

**Result:**

| Variant | Median | Range (min/max) |
|---|---|---|
| BEFORE (`vec![0u8;n]` + zip) | **39.63 ms** | 39.41–42.89 (σ≈1.0) |
| AFTER (`Vec::with_capacity` + chunked extend) | **48.63 ms** | 48.59–48.79 (σ≈0.07) |

A **+23 %** regression, opposite direction from the audit's prediction. The
post-refactor σ is ~14× tighter, suggesting the regression is a stable cost
attribution, not measurement noise.

**Why the prediction was wrong:**

1. **The memset cost the audit assumed wasn't actually paid.** `vec![0u8; n]`
   on Windows allocates via `HeapAlloc` → `VirtualAlloc` with `MEM_COMMIT`. The
   kernel returns *demand-zero pages* — virtual addresses that map to a magic
   zero page lazily, then get individually zero-filled on first write. So the
   "memset" we thought we were eliminating wasn't a separable cost; it was a
   constant per-page tax that any allocation pays. (Linux and macOS also use
   demand-zero pages.)
2. **The chunked structure adds a doubled memory pass.** In the original, each
   element does `read 1 input byte → arithmetic → write 2 output bytes` in one
   tight zip the compiler interleaves. In the chunked refactor, each element
   does `read 1 input byte → arithmetic → write 2 bytes to scratch (L1) →
   memcpy 4096 bytes from scratch to output`. Even though scratch lives in L1,
   the additional memcpy is a measurable secondary cost.

**Disposition:** never committed. `src/remember/fp8.rs` remains on the
pre-refactor pattern.

**Re-attempting this requires:** evidence that one of the lazy-zero-page
absorbing assumptions doesn't hold (e.g., a target where the memset actually
runs eagerly), AND a refactor that doesn't add a second memory pass.

---

## Experiment 3 — v0.4.0 GGUF refactor re-validation

**Background:** The v0.4.0 CHANGELOG ([2026-04-12](../CHANGELOG.md))
described the `Vec::with_capacity` + `extend_from_slice` GGUF dequant
refactor as **"~10–15 % of dequant wall time on `Q8_0`/`Q4_0` saved on
platforms without lazy zero pages"**. The "platforms without lazy zero pages"
caveat is doing a lot of work — Windows, Linux, and macOS all *have* lazy zero
pages. After Experiment 2's null result, this claim looked suspect and was
re-measured.

**Method:** [`tests/bench_dequant_adhoc.rs`](../tests/bench_dequant_adhoc.rs)
`bench_gguf_size_sweep`. Same kernel logic driven two ways via the public
streaming API `dequantize_gguf_blocks_to_bf16`:
- **NEW** — current `dequantize_gguf_to_bf16` (`Vec::with_capacity` +
  per-block `extend_from_slice`).
- **OLD** — bench-local replay of the pre-refactor pattern: pre-allocate
  `vec![0u8; out_byte_len]`, drive the streaming API with a sink that tracks
  an offset and writes via indexed `copy_from_slice`.

Sweep across four output sizes: 2 MB (L3-resident) → 16 MB (L3 boundary) →
90 MB (Llama-class FFN) → 200 MB (deeply DRAM-bound). Best-of-5 release-mode
median per cell.

**Result (NEW vs OLD median delta, negative = NEW faster):**

| Output BF16 size | Q8_0 | Q4_0 |
|---|---|---|
| 2 MB | **+3.0 %** slower | **−6.9 %** faster |
| 16 MB | **+9.5 %** slower | **−8.2 %** faster |
| 90 MB | **+6.3 %** slower | **−7.2 %** faster |
| 200 MB | **+3.9 %** slower | **−9.1 %** faster |
| **Average** | **+5.7 %** slower | **−7.9 %** faster |

The sign is stable across all 4 sizes for each kernel — the result is a
structural property of the kernels, not size-dependent measurement noise.

**Verdict:** the v0.4.0 CHANGELOG claim was **partially wrong**:

- **`Q4_0`** is a real win, but the magnitude was overstated (~8 % measured vs
  10–15 % claimed).
- **`Q8_0`** is a real **regression** (~6 % slower than the pre-refactor
  pattern). The CHANGELOG asserted a uniform improvement; reality is a
  net wash across the two kernels.

**Why `Q8_0` and `Q4_0` disagree** (best understanding):

- Both kernels emit BF16 through the same `dispatch_streaming` → sink-closure
  pipeline. The OLD vs NEW difference is just
  `out[offset..].copy_from_slice(block_out); offset += len;` versus
  `out.extend_from_slice(block_out)` — almost identical machine code.
- **`Q8_0`** is bandwidth-bound (`d × i8 → BF16`, no bit unpacking). The
  output-write bandwidth is the bottleneck. Anything that adds even small
  overhead per block (e.g., extra Vec metadata bookkeeping) shows up.
- **`Q4_0`** does packed-nibble unpacking (`(q & 0xF) - 8`,
  `(q >> 4) - 8`), so the kernel has more CPU work per output byte. The
  per-block overhead is amortised across that work, and `Vec::extend_from_slice`'s
  internal length update apparently has slightly less overhead than the
  manual `offset += ...` pattern for this kernel.

**Disposition:** **current code unchanged.** Two reasons:

1. The deltas roughly cancel: a 5.7 % regression on `Q8_0` and a 7.9 % win on
   `Q4_0`. Splitting the dispatch by kernel (Q4_0 keeps the new pattern, Q8_0
   reverts) would add complexity for a 1–2 ms saving on a 22 ms kernel.
2. The bench file is now the audit-trail. The next person tempted to "fix
   `Q8_0`" can read this entry and Experiment 2 first.

**The principle:** "save the memset" is **not a reliable rationale** in this
codebase. Three of four perf-claim experiments based on it have measured null
or regression. Future audit findings using this framing should be treated as
hypotheses to disprove with measurement, not as actionable.

---

## Experiment 4 — `parse()`: `fs::read` → `memmap2::Mmap`

**Audit finding:** "[`src/model.rs:90`](../src/model.rs) calls `std::fs::read(path)`, materialising the entire safetensors file into a `Vec<u8>` before the header is even parsed. On a 70 GiB shard this peaks at 70 GiB even when the caller only intends to `inspect()`. Switching to `memmap2::Mmap::map(&file)` would let the kernel page bytes in lazily — `parse()` + `inspect()` would then only fault in the header (~1 MiB), and full `remember()` paths gain OOM-resilience because file-backed pages can be dropped by the kernel under memory pressure (whereas `Vec<u8>` pages cannot, they need swap)."

**Method:** [`tests/bench_parse_adhoc.rs`](../tests/bench_parse_adhoc.rs)
`bench_parse_safetensors_large`. Fixture: a locally-cached 11 560 MiB
single-file safetensors model (`bigcode/starcoder2-3b/model.safetensors`).
Best-of-5 release-mode median, 2-iteration warmup to populate the OS file
cache. Compared `parse()` alone and `parse()` + `inspect()`.

**Result:**

| | BEFORE (`fs::read` + `Vec<u8>`) | AFTER (`memmap2::Mmap::map`) | Delta |
|---|---|---|---|
| `parse()` median | **2881.93 ms** (range 2787.82–2887.74, σ ≈ 40 ms) | **0.89 ms** (range 0.86–0.91, σ ≈ 0.02 ms) | **~3236× speedup** |
| `parse()` + `inspect()` median | 2715.84 ms | 0.94 ms | ~2889× speedup |
| `inspect()` overhead | (within noise) | 0.05 ms | ✓ as expected |

The "before" parse() rate is ~4 GiB/s — consistent with `memcpy` from
the warm OS file cache to a fresh `Vec<u8>`. The "after" rate is
file-size-independent: `mmap` setup + parsing the ~1 MiB header.

**Why the prediction was right (and the magnitude):**

`std::fs::read` is `open + read_exact(n) + close` where `n` is the file
size. The dominant cost on a warm cache is the `memcpy` from the FS
cache to the freshly-allocated `Vec<u8>` — ~4 GiB/s on this hardware,
linear in file size.

`memmap2::Mmap::map` is `open + mmap + close` where `mmap` is a
kernel call that establishes virtual address translations without
copying anything — constant time, file-size-independent. Subsequent
reads through the mapping fault in pages on demand. For
`parse_safetensors_header`, only the first ~1 MiB is touched, so for
the inspect-only path the resident-set growth is bounded by header
size, not file size.

The ~3000× speedup is the ratio of (file size / `memcpy` bandwidth)
to (constant `mmap` setup + header parse). It scales with file size:
on a 70 GiB shard the speedup would be larger still.

**Disposition:** **Shipped**. Commit hash recorded in this entry's
index row when the commit lands. All 320 unit tests + every
cross-validation suite (FP8, GPTQ, AWQ, BnB, GGUF, NPZ, PTH) still
pass — the refactor is semantically equivalent because the public
API surface (`ParsedModel::inspect`, `ParsedModel::remember`,
`tensor_data`) all consume the buffer through `&[u8]` slices, and
`memmap2::Mmap` derefs to `[u8]` so callers see no change.

**Trade-offs accepted:**

- `memmap2` becomes a mandatory dependency (was optional, gated behind
  `pth`/`gguf`). This adds ~one small crate to the dependency tree of
  every build, including the safetensors-only minimal build. Justified
  by the always-on speedup.
- Concurrent file modification by another process is now undefined
  behaviour — the same assumption every other tensor parser in this
  crate (`parse_pth`, `parse_gguf`) and the upstream `safetensors`
  crate's mmap path already rely on. Documented in the `// SAFETY:`
  comment and the [CONVENTIONS.md](../CONVENTIONS.md) accepted-`unsafe`
  table.

**Re-attempting this requires:** N/A — this is the success case. If
the change ever needs to be reverted, the `bench_parse_adhoc` harness
is in place to detect a regression.

## Experiment 5 — `inspect_gguf_from_reader`: internal `BufReader<R>` (Tier 1)

**Audit finding:** The Phase 4.9 substrate-equivalence test surfaced
that `inspect_gguf_from_reader(File::open(path)?)` was 30–100× slower
than `parse_gguf(path).inspect()` on the same file (e.g., 213 ms vs.
3.0 ms on a 2.7 GiB Mistral-7B-IQ3_XXS). Diagnosis: the parser issues
many small `read_exact` calls (4–8 B per typed primitive, variable per
`gguf_string_t`), and on a `File` substrate every one is a syscall.
Hypothesis: wrapping the user's reader in a `std::io::BufReader<R>`
(64 KiB buffer) inside `inspect_gguf_from_reader` collapses those into
one underlying read per buffer-fill, with no API change and no
correctness risk (the only `Seek` calls happen at `GgufReader::new`
*before* any reads, so the buffer is empty when seek is issued — no
invalidation cost).

**Method:** [`tests/bench_gguf_inspect_adhoc.rs`](../tests/bench_gguf_inspect_adhoc.rs)
(`bench_gguf_inspect_paths`), best-of-5 release-mode median per file
with min/max range, target-cpu=native (`$env:RUSTFLAGS = "-C target-cpu=native"`),
1 warm-up iteration before timing. Compared baseline (no `BufReader`)
vs. post-Tier-1 (`BufReader::with_capacity(64 * 1024, reader)`) by
running the bench, applying the patch, running again. 17 real `GGUF`
files from `tests/fixtures/gguf_reference/models/` spanning 4
architectures × 11 distinct dtypes × 84 MiB to 2.7 GiB:

- `bartowski/SmolLM2-135M-Instruct` (8 quants: `Q2_K`, `Q3_K_M`, `Q4_0`,
  `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ4_XS`)
- `bartowski/Mistral-7B-Instruct-v0.3` (5 quants: `IQ1_S`, `IQ1_M`,
  `IQ2_XXS`, `IQ2_XS`, `IQ3_XXS`)
- `bartowski/Qwen2.5-{0.5,1.5}B-Instruct-IQ2_M`
- `TheBloke/TinyLlama-1.1B-chat-v1.0` (`Q2_K`, `Q5_0`)

**Result:**

| Aggregate | Baseline reader/mmap ratio | Post-Tier-1 reader/mmap ratio |
|---|---|---|
| Min | 46.6× slower | 0.9× (slightly **faster** than mmap) |
| Median | 51.7× slower | 1.0× (parity) |
| Mean | 56.6× slower | 1.0× (parity) |
| Max | 71.4× slower | 1.0× (parity) |

Per-file reader medians (μs), best-of-5:

| File | Baseline | Tier 1 | Reader speedup |
|---|---:|---:|---:|
| Mistral-7B-Instruct-v0.3-IQ1_M | 209,452 | 2,845 | **73.6×** |
| Mistral-7B-Instruct-v0.3-IQ1_S | 213,157 | 2,856 | **74.6×** |
| Mistral-7B-Instruct-v0.3-IQ2_XS | 214,694 | 2,826 | **76.0×** |
| Mistral-7B-Instruct-v0.3-IQ2_XXS | 214,768 | 2,881 | **74.5×** |
| Mistral-7B-Instruct-v0.3-IQ3_XXS | 213,215 | 2,829 | **75.4×** |
| Qwen2.5-0.5B-Instruct-IQ2_M | 1,228,412 | 25,712 | **47.8×** |
| Qwen2.5-1.5B-Instruct-IQ2_M | 1,229,113 | 25,424 | **48.3×** |
| SmolLM2-135M-Instruct-IQ4_XS | 399,473 | 7,538 | **53.0×** |
| SmolLM2-135M-Instruct-Q2_K | 397,048 | 8,338 | **47.6×** |
| SmolLM2-135M-Instruct-Q3_K_M | 400,154 | 7,753 | **51.6×** |
| SmolLM2-135M-Instruct-Q4_0 | 400,510 | 8,054 | **49.7×** |
| SmolLM2-135M-Instruct-Q4_K_M | 399,283 | 7,578 | **52.7×** |
| SmolLM2-135M-Instruct-Q5_K_M | 397,638 | 7,558 | **52.6×** |
| SmolLM2-135M-Instruct-Q6_K | 398,046 | 7,641 | **52.1×** |
| SmolLM2-135M-Instruct-Q8_0 | 430,908 | 7,560 | **57.0×** |
| TinyLlama-1.1B-chat-v1.0.Q2_K | 440,615 | 6,961 | **63.3×** |
| TinyLlama-1.1B-chat-v1.0.Q5_0 | 437,530 | 7,132 | **61.4×** |

The `parse_gguf(path).inspect()` (mmap-backed) numbers are unchanged
across the two runs — Tier 1 only touches the reader-generic entry
point, by design. Median mmap times: ~3.0 ms for Mistral-7B,
~26.4 ms for Qwen2.5, ~8.0 ms for SmolLM2, ~7.9 ms for TinyLlama.

**Why the prediction was right and the headline result was bigger
than expected:** On a `File` substrate with cold-then-warm fs cache,
the post-Tier-1 reader path occasionally measures *faster than mmap*
(0.9× ratio). The likely explanation: BufReader does one syscall per
64 KiB of metadata, while mmap incurs one minor page fault per 4 KiB
page touched (the front matter is ~few MiB on these fixtures, so
dozens of syscalls vs. a few hundred page faults). Both backends
ultimately read the same OS-cached pages — but BufReader's larger
batch granularity wins on this access pattern.

The 30–100× baseline ratio was an underestimate of the syscall cost
on Windows; the 47–76× per-file speedups are the empirical answer.
The Qwen2.5 fixtures are the slowest in absolute terms (1.23 s
baseline) because their `tokenizer.ggml.tokens` arrays are larger
than the SmolLM2/TinyLlama equivalents (Qwen has a 152K-entry
vocabulary vs. SmolLM2's 49K), giving more per-element reads to
amortise.

**Disposition:** **Shipped.** All 28 GGUF parser unit tests + the
real-fixture substrate-equivalence test (17/17) still pass — every
field of `GgufInspectInfo` is identical pre- and post-Tier-1 because
the bytes read are identical, only the syscall granularity changed.
The `# Performance` rustdoc on `inspect_gguf_from_reader` was updated
to reflect the new numbers and to remove the now-stale "use mmap for
local files" guidance.

**Trade-offs accepted:**

- **+~64 KiB heap per call** for the BufReader's internal buffer.
  Negligible vs. the parsed metadata `HashMap` (often hundreds of KiB
  to a few MiB for the tokenizer arrays).
- **Caller can no longer pass a non-buffered `Read + Seek` and rely
  on its own buffering decisions** — but the type signature is
  unchanged (`R: Read + Seek` in, `Result<GgufInspectInfo>` out), so
  this is a strictly internal optimisation. Callers that want to
  control buffering can pass any `Read + Seek`; the internal
  `BufReader` will wrap it (mostly redundantly for an in-memory
  `Cursor`, but the per-call memcpy cost is dwarfed by the parsing
  work).

**Tier 2 not pursued:** The original analysis identified a "bulk-read
typed arrays in `read_typed_array`" optimisation (collapse the
per-element `Vec::push` loop into one `read_into` + `chunks_exact`
convert) as a Tier 2 follow-up. With Tier 1 closing the gap to mmap
parity, Tier 2's added complexity (security guard for "fail-before-
allocate" on adversarial array-length headers) is no longer
justified. The geometric-growth `Vec::push` pattern stays.

**Re-attempting this requires:** N/A — this is the success case.
[`tests/bench_gguf_inspect_adhoc.rs`](../tests/bench_gguf_inspect_adhoc.rs)
is in place to detect any regression. If a future change to
`GgufReader` reintroduces per-element reads on top of `BufReader`
(e.g., dropping `read_into` for some other pattern), the bench will
catch it.

## Experiment 6 — `inspect_pth_from_reader` reader vs mmap vs `torch.load`

**Question:** Does Phase 4.10's reader-generic
[`inspect_pth_from_reader`](../src/parse/pth.rs) match the mmap-backed
`parse_pth(path).inspect()` in throughput, and how does each compare to the
closest Python equivalent (`torch.load(weights_only=True)` + iterate the
returned `state_dict` to compute the same summary fields)? Phase 4.10's
PTH parser does not adopt the GGUF Tier-1 `BufReader` win because the I/O
pattern is structurally different (bulk reads through `zip`, not many small
`read_exact` calls) — does the measurement confirm that the parity claim
holds without buffering?

**Method:**
[`tests/bench_pth_inspect_adhoc.rs`](../tests/bench_pth_inspect_adhoc.rs)
(`bench_pth_inspect_paths`, `#[ignore]`-gated) plus the Python script at
[`tests/fixtures/pth_reference/bench_python_inspect.py`](../tests/fixtures/pth_reference/bench_python_inspect.py).
Best-of-5 release-mode median per file, `target-cpu=native`, warmed FS
cache, one warm-up iteration before timing. Three AlgZoo fixtures (the
only `.pth` fixtures checked into the repo): `algzoo_rnn_small.pth`
(2.0 KiB), `algzoo_transformer_small.pth` (3.5 KiB), `algzoo_rnn_blog.pth`
(3.3 KiB). PyTorch 2.10.0+cu130 on the Python side; same machine, same
files.

**Result — Rust mmap vs Rust reader (this commit):**

| Fixture | mmap median | reader median | reader / mmap |
|---|---:|---:|---:|
| `algzoo_rnn_small.pth` (2.0 KiB) | 134.4 µs | 220.1 µs | 1.64× |
| `algzoo_transformer_small.pth` (3.5 KiB) | 154.0 µs | 236.1 µs | 1.53× |
| `algzoo_rnn_blog.pth` (3.3 KiB) | 133.1 µs | 151.5 µs | 1.14× |
| **median across fixtures** | — | — | **1.53×** |

**Result — Rust paths vs Python `torch.load`:**

| Fixture | torch.load median | mmap speedup | reader speedup |
|---|---:|---:|---:|
| `algzoo_rnn_small.pth` (2.0 KiB) | 532.7 µs | 4.0× | 2.4× |
| `algzoo_transformer_small.pth` (3.5 KiB) | 858.7 µs | 5.6× | 3.6× |
| `algzoo_rnn_blog.pth` (3.3 KiB) | 530.6 µs | 4.0× | 3.5× |
| **median across fixtures** | — | **4.0×** | **3.5×** |

**Why the reader path is slower than mmap on these fixtures:** the AlgZoo
files are 2–4 KiB — at this scale, *fixed* costs dominate over per-byte
work. The reader path pays for:

1. `seek(End(0))` to capture total length (one syscall).
2. The 4-byte magic probe + rewind (one read, one seek — kept so the
   *"legacy pre-1.6 raw pickle"* diagnostic remains distinct from the
   generic *"not a valid ZIP"* diagnostic).
3. `zip::ZipArchive::new` doing its own EOCD scan (several reads near the
   end of the file, with `zip`'s internal buffering).
4. One `Vec::with_capacity(pkl_size)` + `read_to_end` for `data.pkl`.

The mmap path skips (1) and (3)'s syscall costs because the file is already
in the page cache; it pays one minor page fault for the 4 KiB containing
the EOCD and central directory. On a 2 KiB fixture the entire archive fits
in one page, so the mmap path is essentially free past the initial mmap
syscall.

On larger files the relative overhead collapses. Linear extrapolation from
the Phase 4.9 GGUF benchmark (where reader and mmap reached parity ~3 ms on
multi-GB models) plus the per-byte work being dominated by the pickle VM
(which is identical across substrates and takes O(`pkl_size`) ≈ O(tens of
microseconds for tens of KiB)) gives reader/mmap parity at a few hundred
KiB of `data.pkl` — which is the realistic range for torchvision-class
models (~50 KiB of `data.pkl` on ResNet-50 / ViT-B/16).

**Why both Rust paths are faster than `torch.load`:** PyTorch has no
separate inspect-only primitive — `torch.load(weights_only=True)`
materialises every tensor as a `torch.Tensor` on CPU before the caller
can iterate the `state_dict` for summary stats. Even on a 2 KiB fixture
that involves tens of `torch.Tensor` constructions plus the surrounding
Python overhead. The Rust paths skip *all* tensor materialisation — only
the pickle metadata is interpreted.

The 3.5× median speedup on tiny fixtures is a **lower bound** for the
reader path: scaling to a torchvision-class 300 MB `.pth`, `torch.load`'s
time grows linearly with the total `data/N` size while
`inspect_pth_from_reader`'s time stays bounded by `data.pkl` size (tens
of KiB). On 300 MB models the reader path would beat `torch.load` by
multiple orders of magnitude, mirroring the 11.2–30.8× full-parse
speedups already documented in the project README for the mmap path.

**Disposition:** **Shipped.** The reader-generic path lands without an
internal `BufReader` because:

- The parity gap on KiB fixtures (1.14–1.64× of mmap) is small in absolute
  terms (~20–90 µs across the 3 in-tree fixtures; ~45 µs at the corpus
  median, see Follow-up below) and would not benefit meaningfully from
  buffering — the `zip` crate already does its own buffering on the
  central-directory scan, and our two payload reads are bulk
  `read_to_end` calls.
- Adding `BufReader<R>` would introduce one extra memcpy per buffer-fill
  on every substrate (including `Cursor`), with no syscall reduction (the
  fixed-cost path is dominated by `seek + EOCD scan + central-directory
  parse`, not per-element reads).
- The 3.5× speedup vs `torch.load` is already comfortable; the remaining
  ~80 µs gap to the mmap path on the 3-fixture median (~45 µs on the
  6 960-file corpus median — see Follow-up) is a fixed cost of the
  ZIP-archive abstraction, not a syscall pattern that buffering would
  amortise.

The Phase 4.9 GGUF rationale for adding `BufReader<R>` (collapsing many
4–8 B `read_exact` calls on a `File` substrate into one syscall per buffer
fill) does not apply here.

**Re-attempting this requires:** evidence that on a torchvision-class
real `.pth` (≥45 MB, ≥100 tensors, ≥50 KiB `data.pkl`) the reader path is
more than ~1.5× slower than the mmap path — at which point the parity
claim in the rustdoc would need to be revised and `BufReader<R>`
reconsidered. The current bench harness
([`bench_pth_inspect_paths`](../tests/bench_pth_inspect_adhoc.rs)) accepts
arbitrary additional fixtures dropped into
`tests/fixtures/pth_reference/`.

### Follow-up — 6 960-file `AlgZoo` corpus sweep

The 3 in-tree fixtures are a sanity check, not a population estimate. To
back the rustdoc parity claim with a broader sample, the bench harness
grew two new tests: `bench_pth_inspect_algzoo_sweep` (Rust) and the
`ANAMNESIS_ALGZOO_DIR` sweep mode of `bench_python_inspect.py` (Python).
Both walk every `*.pth` file under a configurable directory and report
aggregate distributions plus per-task-family breakdown.

**Corpus:** `algzoo_weights/` — the full `AlgZoo` model set imported for
`candle-mi` v0.1.9's `stoicheia` module: **6 960 files**, 22.6 MiB total,
median file size 2.5 KiB (range 2.0–7.7 KiB), grouped into four
algorithmic-task families:

| Task family | File count |
|---|---:|
| `2nd_argmax` | 3 360 |
| `argmedian` | 1 200 |
| `longest_cycle` | 1 200 |
| `median` | 1 200 |

**Method:** same as above (best-of-5 release-mode median per file,
`target-cpu=native`, warmed FS cache, one warm-up iteration), but with
34 800 timed measurements per substrate (6 960 files × 5 iterations)
instead of 15 (3 fixtures × 5 iterations). Rust wall-clock 13.5 s
(516 files/s); Python wall-clock 24.4 s (286 files/s) — both
single-process, single-threaded.

**Global distribution (per-file medians, µs):**

| Substrate | min | p25 | median | p75 | mean | max |
|---|---:|---:|---:|---:|---:|---:|
| `parse_pth(path).inspect()` (mmap)        | 117.4 | 122.3 | **124.0** | 128.4 | 127.7 | 415.1 |
| `inspect_pth_from_reader(File)` (reader)  | 160.0 | 165.9 | **168.7** | 173.1 | 177.9 | 612.5 |
| `torch.load(weights_only=True)` (PyTorch) | 489.3 | 500.6 | **504.3** | 512.7 | 559.2 | 951.3 |
| reader / mmap                             | 0.54  | 1.34  | **1.36**  | 1.39  | 1.39  | 4.63  |

**Cross-language speedups (median across all 6 960 files):**

- `parse_pth(path).inspect()` is **4.07× faster than `torch.load`** (504.3 / 124.0).
- `inspect_pth_from_reader` is **2.99× faster than `torch.load`** (504.3 / 168.7).
- `inspect_pth_from_reader` is **1.36× the time of the mmap path** (168.7 / 124.0).

**Per-family breakdown (medians, µs):**

All ratios are computed as `ratio-of-medians` from the row's µs values,
matching the global section's method. Each `reader/mmap` cell therefore
equals (reader median) / (mmap median) in the same row; small differences
from the bench harness's `median-of-per-file-ratios` (also reported in
the bench output) are expected at this rounding precision.

| Family | Count | `torch.load` | mmap | reader | mmap×torch.load | reader×torch.load | reader/mmap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `2nd_argmax`    | 3 360 | 503.3 | 123.7 | 168.3 | 4.07× | 2.99× | 1.36× |
| `argmedian`     | 1 200 | 502.5 | 123.8 | 169.1 | 4.06× | 2.97× | 1.37× |
| `longest_cycle` | 1 200 | 818.1 | 144.1 | 223.8 | 5.68× | 3.66× | 1.55× |
| `median`        | 1 200 | 502.0 | 121.3 | 164.2 | 4.14× | 3.06× | 1.35× |

**Reading the numbers:**

- **The parity claim holds.** The 3-fixture median of 1.53× was a small
  sample biased upward by the middle fixture's 1.53× ratio sitting well
  above the corpus distribution: only one of the three in-tree fixtures
  (1.14×) fell within the 6 960-file p25–p75 band of 1.34×–1.39×. The
  6 960-file median is **1.36×**, with p25 = 1.34× and p75 = 1.39× — a
  tight distribution that says the reader/mmap gap is structurally
  fixed at ~45 µs on KiB-scale `.pth` files (`168.7 − 124.0 = 44.7 µs`
  at the median). The 1.5× re-attempt threshold in this experiment's
  *Re-attempting* clause stands.
- **`longest_cycle` is the outlier**, with median timings ~17 % slower
  than the other three families on the mmap path, ~34 % slower on the
  reader path, and ~63 % slower on `torch.load`. The task itself is
  structurally heavier — `longest_cycle` `AlgZoo` models use the
  Transformer architecture (more tensors per file) while the other
  three families use simpler RNN-style models, so the pickle
  interpreter does more work per file. All three substrates rank
  `longest_cycle` slowest, confirming it's task-driven, not
  substrate-driven. The compounding on `torch.load` (Rust paths add
  ~17–34 %, Python adds ~63 %) is consistent with the extra tensors
  triggering extra Python-side `torch.Tensor` materialisation on top of
  the extra pickle-VM work.
- **The cross-language speedup tightens.** Earlier 3-fixture median
  reader-vs-`torch.load` was **3.5×**; the 6 960-file median is
  **2.99×**. The drop reflects the larger sample averaging out
  `torch.load`'s tail (the 3-fixture set included
  `algzoo_transformer_small.pth` whose 858 µs `torch.load` time is well
  above the corpus p75 of 512.7 µs — an upper-decile case). The 4.07×
  mmap speedup is essentially unchanged from the 3-fixture 4.0× median.
- **The `torch.load` distribution is narrow.** p25/p75 are 500.6/512.7
  µs — within ±2 % of the 504.3 µs median — so the speedup ratios are
  not an artefact of a fat-tailed Python distribution.
- **None of the conclusions of the 3-fixture experiment are changed.**
  Specifically: the reader path is still **shipped without `BufReader`**;
  the rustdoc parity claim "~1.14–1.64× the time of the mmap-backed
  `parse_pth(path).inspect()`" is updated to "~1.36× median across 6 960
  AlgZoo files (p25=1.34×, p75=1.39×)" in this commit.

**Disposition (follow-up):** Numbers added to the rustdoc on
`inspect_pth_from_reader`'s `# Performance` section; CHANGELOG entry
updated. No behavioural change to the code itself — only the empirical
evidence base widened from 3 fixtures to 6 960.

## How to add an entry

When you ship (or attempt to ship) a perf-claim change, add a row to the index
table and a section below. The minimum content is:

- **Audit finding** (one paragraph) — what was claimed and why.
- **Method** — bench file, fixture, hardware/OS, harness type (best-of-N
  median, etc.).
- **Result** — a table of before/after numbers with σ or range.
- **Why the prediction was right or wrong** — root-cause analysis,
  preferably citing measured behaviour rather than asymptotic argument.
- **Disposition** — committed (with hash), reverted (with hash), or never
  committed.
- **Re-attempting this requires** — what new evidence would make the
  experiment worth retrying.

Keep entries even when the experiment *succeeds*: a successful experiment with
documented before/after numbers is the strongest possible defense against
future regressions.

---

## Experiment 7 — Sign-of-zero preservation rule (`BnB FP4` decode tweak)

**Scope note:** Unlike Experiments 1–6, this is a **correctness** experiment, not a perf one. Included in this file because the template is identical: an initial framing claimed a property was impossible; empirical observation on a real fixture narrowed the framing; a targeted code change recovered the desired invariant; cross-architecture validation confirmed the change generalises. Future encode kernels (`FP8`, `GGUF`, `IQ`, `TQ`, `MXFP4` in Phase 8.5) may surface analogous "codebook-quirk-driven" findings — this entry sets the precedent.

**Initial claim during Phase 5 step 1 design discussion:** "*Byte-level round-trip of `BnB FP4` is mathematically impossible — bitsandbytes' Python on-disk `quant_map` stores `+0.0` at both index 0 and index 8 (collapsing the `±0` pair), so decoding nibble 8 produces `+0.0` BF16, indistinguishable from nibble 0, and no encoder can recover which original nibble produced it.*" Conclusion drawn at the time: "the operative contract for FP4 is decode-equivalence (`decode(re_encoded) == decode(weight_data)` at the BF16 level), not byte-exact round-trip."

**What surfaced the over-claim:** The user pushed back on the word "impossible". The accurate framing is conditional: byte-level round-trip is impossible *under the existing decode contract* (which required our decode to bit-exactly match bitsandbytes' Python decode, which is itself lossy on the sign of zero). Drop the constraint and the loss is recoverable.

**Method:** Three measurements on the existing Llama-1B `FP4` fixture (`HF-Quantization/Llama-3.2-1B-BNB-FP4`, 2048-byte slice, 4096 elements):

1. **Baseline:** decode → encode without any tweak. Result: 8 / 2048 byte mismatches (0.39 %), all of the form "we output nibble 0 where the fixture has nibble 8".
2. **Decode tweak applied:** in `dequantize_bnb4_to_bf16`, when `codebook[nibble].to_bits() == 0` AND `nibble & 0x8 != 0`, substitute `-0.0` for `+0.0` in the BF16 output. Re-run round-trip. Result: still 8 / 2048 byte mismatches — the decode side now emits `-0.0` at the relevant positions, but the encoder's nearest-search treats `-0` and `+0` as equidistant and picks the lower index.
3. **Decode tweak + encode-side mirror applied:** in `encode_bnb4_core`, when the source value `is_sign_negative()` AND the nearest-search returned a lower-half nibble AND `codebook[lower].to_bits() == codebook[upper].to_bits()`, shift to the upper-half nibble. Re-run. Result: **0 / 2048 byte mismatches** — byte-exact round-trip recovered.

**Cross-architecture validation** (Phase 5 steps 1b / 1c, against fixtures from different orgs to confirm the codebook-collapse quirk is not Llama-fixture-specific):

| Fixture | Before tweak | After tweak (decode + encode mirror) |
|---|---|---|
| `HF-Quantization/Llama-3.2-1B-BNB-FP4` (Llama) | 8 / 2048 byte diffs (measured) | **0 / 2048** |
| `ema1234/qwen_mcqa_bnb_fp4` (Qwen3) | not measured pre-tweak; post-tweak byte-exact round-trip is consistent with the tweak firing on this fixture's codebook too | **0 / 2048** |
| `medmekk/Llama-3.2-1B-Instruct-bnb-nf4` (NF4, plain) | 0 / 2048 (tweak inactive on NF4 — `codebook[8] = 0.0795 …`, no `+0/+0` collision) | **0 / 2048** (unchanged) |
| `medmekk/Llama-3.2-1B-Instruct-bnb-nf4-double-quant` (NF4, DQ) | 0 / 2048 (tweak inactive on NF4) | **0 / 2048** (unchanged) |
| `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` (NF4 DQ, Qwen2.5) | not measured pre-tweak (tweak inactive on NF4) | **0 / 2048** |
| `unsloth/Phi-3.5-mini-instruct-bnb-4bit` (NF4 DQ, Phi-3.5) | not measured pre-tweak (tweak inactive on NF4) | **0 / 2048** |
| `HF-Quantization/Llama-3.2-1B-BNB-INT8` (INT8) | 0 / 65536 (no codebook) | **0 / 65536** (unchanged) |

Tweak fires on the **2 FP4 fixtures** (Llama + Qwen3), recovers byte-exactness in both; **no-op** on every NF4 / INT8 fixture. Tested on 4 architecture families.

**Why the initial framing was wrong:**

The "impossibility" was reasoned from a fixed-point assumption: that anamnesis's decode must bit-exactly match bitsandbytes' Python decode on every element, forever. Under that constraint, the lossy collapse in bitsandbytes' Python codebook becomes a lossy intermediate in our pipeline, and information lost in the intermediate cannot reappear downstream. Drop the constraint — specifically, allow our decode to emit `-0.0` BF16 at sign-bit positions where bitsandbytes' Python decode emits `+0.0` BF16 — and the round-trip is recoverable.

The downstream cost of dropping the constraint is bounded:

- **IEEE 754 arithmetic** treats `+0.0` and `-0.0` as equal (`+0.0 == -0.0` is `true`). Any subsequent multiply / add / matmul on the decoded `BF16` produces identical results modulo the sign bit on the zero output (which is itself an IEEE 754 equivalence class).
- **Decode bit-exactness test breakage** was prevented by a one-line tweak to `compare_bf16` in `tests/cross_validation_bnb.rs`: treat `±0` as IEEE-equivalent when computing ULP distance. Documented, principled, narrow.
- **No-op for every codebook whose upper-half indices have non-zero entries** — NF4 (`codebook[7] = 0.0`, `codebook[15] = 1.0`), every GGUF codebook, FP8 (no codebook collapse), etc. The tweak is FP4-specific by construction even though it's expressed as a general "if codebook entry is `+0` AND nibble high bit set" rule.

**Disposition: Shipped**. Three commits (use `git log` or [`CHANGELOG.md`](../CHANGELOG.md)'s `[0.5.0]` entry for context):

- `a5c452d` (Phase 5 step 1a) — tweak introduced in `dequantize_bnb4_to_bf16` + mirror in `encode_bnb4`; `compare_bf16` updated; unit tests `apply_sign_magnitude_zero_flips_only_when_codebook_is_plus_zero` + `apply_sign_magnitude_encode_correction_lifts_to_upper_when_duplicated` lock the behaviour.
- `24cba42` (Phase 5 step 1b) — Qwen3 FP4 fixture proves cross-architecture generalisation.
- `ab4e735` (Phase 5 step 1c) — `encode_bnb4_double_quant` extends the same tweak through the double-quant path.

**Trade-offs accepted:**

- Anamnesis's decode is no longer a bit-exact mirror of bitsandbytes' Python decode on the `0.2 %` of `FP4` elements where the codebook collapse fires. The deviation is arithmetically invisible (`+0` vs `-0` IEEE 754 equivalence), documented in [`src/remember/bnb.rs`](../src/remember/bnb.rs)'s `dequantize_bnb4_to_bf16` rustdoc, and unit-tested.
- A future bitsandbytes Python release that fixes the `quant_map` collapse (storing `-0.0` at index 8 instead of `+0.0`) would re-establish bit-exactness on both sides — our tweak would become a no-op on the fixed codebook because `codebook[8].to_bits() != codebook[0].to_bits()` would short-circuit the condition. Forward-compatible by construction.

**Re-attempting this requires:** N/A — this is the success case. If the change ever needs to be reverted, the cross-architecture FP4 round-trip tests in [`tests/cross_validation_bnb_encode.rs`](../tests/cross_validation_bnb_encode.rs) will surface a byte regression on the Llama fixture (the 8 / 2048 originally measured) and on the Qwen3 fixture (count not pre-measured but at least 1 since the post-tweak round-trip is byte-exact) within the next test run.

**Cross-reference:** The full design discussion that led to this rule is summarised in [`ROADMAP.md`](../ROADMAP.md)'s Phase 5 "Boundary-pushing finding (sign-of-zero preservation)" paragraph and in commit `a5c452d`'s commit-message body.

**Template for future encode-side correctness findings:** when adding a new encode kernel family in Phase 8.5 (FP8, GGUF legacy/K/IQ/TQ/MXFP4), check whether the on-disk codebook has any collapsed-entry pairs of the form `codebook[i].to_bits() == codebook[j].to_bits()` for `i != j`. If so, the same template applies: (1) measure baseline round-trip error, (2) identify whether decode could disambiguate via some carrier the existing kernel ignores, (3) apply the narrowest possible decode + encode tweak pair, (4) verify on cross-architecture fixtures.

---

## Experiment 8 — Vendored `ZIP` reader: container-metadata footprint

**Hypothesis (Phase 6.8 "reopened by measurement" → Phase 6.12).** `zip::ZipArchive::new`
eagerly materialises the whole central directory into a fat per-entry
`ZipFileData` record, estimated at **~500 B/entry (~5.7× the file)** for a
many-tiny-entry archive, versus the **~40 B/entry** anamnesis needs (a
`name → (offset, size)` index). Replacing it with a vendored, read-only
central-directory reader was projected to cut resident container metadata
**~12×** (500 → 40 B/entry).

**Method.** `tests/peak_heap_zip_metadata.rs` (dev-only, `#[ignore]`), `dhat`
global allocator, release build, on a 50 001-entry archive (50 000 tiny STORED
`archive/data/N` entries + an empty-`state_dict` `data.pkl`). Both readers go
through what they actually expose: the vendored reader via the public
`parse_pth` (mmap path, empty pickle so the pickle VM contributes ~nothing),
the `zip` crate via `ZipArchive::new`. `dhat` tracks the global allocator, so
the mmap'd file body is not counted — only container metadata heap.
Run: `cargo test --release --features pth --test peak_heap_zip_metadata -- --ignored --nocapture`.

**Result — Shipped (v0.6.7).**

| Reader | Resident | B/entry | Peak |
|---|---:|---:|---:|
| `zip::ZipArchive::new` | 16 856 982 B | **337** | 27 257 038 B |
| vendored `parse_pth` | 2 088 930 B | **41** | 8 745 004 B |
| **reduction** | | **8.07×** | **3.12×** |

The realised resident figure (41 B/entry) **hits the ~40 B/entry target**. The
projected ~12× did not materialise for one measured reason: the `zip` crate
costs **337 B/entry on these short entry names**, not the estimated ~500 (the
fixed `ZipFileData` fields dominate and are lighter than assumed), so the
ceiling here is `337 / 40 ≈ 8.4×` — and the vendored index sits at it.

**Getting to the ceiling took two index-representation iterations** (each
re-measured against the same fixture):

| Index representation | Resident B/entry | Reduction |
|---|---:|---:|
| `HashMap<String, (usize, usize)>` (first cut) | 63 | 5.31× |
| `Vec<(Box<str>, usize, usize)>`, sorted, binary-searched | 51 | 6.52× |
| … + `shrink_to_fit` (reclaim push-growth slack) | **41** | **8.07×** |

The `HashMap` lost to (a) its power-of-two bucket array (65 536 buckets for
50 001 entries — ~31 % slack) and (b) `String`'s 8-byte capacity word per key.
A sorted `Vec` of `Box<str>` keys removes both; `shrink_to_fit` after the
build (the index is immutable thereafter) reclaims the `Vec`'s own
push-growth over-allocation — that last step alone moved 51 → 41 B/entry.

**Peak (3.12×) is unaffected by the later micro-optimisations** (it stays
8 745 004 B): the global peak lands during `EntryIndex` construction
(`Vec<ZipEntry>` + the sorted index coexisting), so the zero-copy
central-directory borrow added on the mmap path lowers an earlier, non-dominant
transient without moving the headline peak — a real allocation-pressure win the
peak metric simply doesn't capture.

**Re-attempting this requires:** N/A — success case. The `#[ignore]` test is a
committed regression guard (it asserts a resident reduction); re-run it against
the parent commit to reproduce the before/after.

---

## Experiment 9 — `convert` copy-elimination pass (Phase 6.14)

**Audit finding (self-review before the Python bindings expose `convert()`):**
the BF16-hub `convert` path carried four avoidable costs — (1) `hub_tensors`
recovered each passthrough tensor's dtype with a linear `find` over all tensors
(`O(passthrough × N)`); (2) `to_bf16_bytes` did a full `data.to_vec()` even when
the tensor was already `BF16`, allocating a second full-model buffer alongside
the hub on the `bnb-nf4` path; (3) `write_gguf_target` deep-cloned the inherited
source KV — including a multi-thousand-entry tokenizer array — even with no
caller KV to merge; (4) `read_npz` cloned every tensor's bytes instead of moving
out of the owned map. None changes output bytes; all are pure copy/scan
elimination.

**Method:** [`tests/bench_convert_adhoc.rs`](../tests/bench_convert_adhoc.rs), a
`dhat`-instrumented `#[ignore]` harness (synthetic fixtures, one `dhat::Profiler`
scope per route, fixtures built *before* the profiler so only `convert()`'s own
allocations are counted). Release build. Compared parent vs patched on the same
binary:
`cargo test --release --features npz,gguf,bnb,pth --test bench_convert_adhoc -- --ignored --nocapture`.

**Result (peak = `dhat` `max_bytes`; total = cumulative allocated):**

| Route (fixture) | Metric | Before | After | Δ |
|---|---|---:|---:|---:|
| #2 `BF16` st → `bnb-nf4` (8192×8192, 128 MiB hub) | peak | 328.0 MiB | 200.0 MiB | **−128.0 MiB (−39.0 %)** |
| | total | 332.0 MiB | 204.0 MiB | −128.0 MiB |
| #3 `gguf → gguf` (256 K-token tokenizer KV) | peak | 25.3 MiB | 19.0 MiB | **−6.3 MiB (−25.0 %)** |
| | total | 47.1 MiB | 37.6 MiB | −9.5 MiB (−20.2 %) |
| | blocks | 786 504 | 524 353 | −262 151 (= one 256 K-entry KV copy) |
| #4 `NPZ` → safetensors (2×4096×4096 F32, 128 MiB) | peak | 256.0 MiB | 129.0 MiB | **−127.0 MiB (−49.6 %)** |
| | total | 257.1 MiB | 129.1 MiB | −128.0 MiB |

**Why the numbers land exactly on the eliminated copy:** each drop equals one
model-sized (or one KV-sized) buffer, confirming the hypothesis precisely rather
than approximately. #2 removes the 128 MiB `to_bf16_bytes` copy of the
already-`BF16` hub; #4 removes the 128 MiB per-tensor NPZ clone (peak halves
because the owned parse map and the hub no longer coexist at full size); #3's
`blocks` count falls by exactly 262 144 = 256 K — the tokenizer array is now
copied twice (parse + the still-necessary reader-side owning clone) instead of
three times (the write-side merge clone is gone).

**#1 (the `O(P·N)` → `O(N)` dtype lookup)** is **not** in the table: it changes
no allocation these routes exercise (a `find` and a one-pass index both touch the
same bytes), and at current model sizes the scan-count reduction is far below
wall-clock noise (the dequant dominates). It is an asymptotic guard for
many-tensor models (a 70 B checkpoint has 1 000+ tensors), verified by inspection
and the existing `convert` round-trip tests, not by measurement — claimed as a
complexity improvement only, per the "measure before claiming a speed win" rule.

**Disposition:** shipped in Phase 6.14 (v0.6.9). The harness is committed
`#[ignore]`; re-run it against the parent commit to reproduce before/after.

**Re-attempting this requires:** N/A — success case. Related deeper wins left for
later: the hub is still materialised in full (`O(model)`; streaming it down to
`O(largest tensor)` is Phase 10), and the reader-side GGUF KV clone in
`read_gguf` (necessary because the hub outlives the mmap-backed parse) could be
moved rather than cloned if `ParsedGguf` gained an `into_metadata()`.

---

## Experiment 10 — Phase 7 Stage-0 SIMD roofline (which loops are worth vectorizing?)

**Scope note:** This is the measurement-first Stage 0 of Phase 7 (CPU SIMD). The
ROADMAP's Phase-7 thesis was *"replace the scalar pass-2 `f32 → BF16` writer with
runtime-dispatched AVX2/NEON for a 4–8× speedup on the hot path."* Before writing any
`unsafe` intrinsic, Stage 0 asks the two questions that decide whether that thesis
holds: **(1)** is the shared pass-2 writer compute-bound (SIMD can help) or
bandwidth-bound (SIMD cannot beat DRAM)? and **(2)** what does the compiler already
auto-vectorize today?

**Method:**

- **Roofline bench** — [`tests/bench_pass2_adhoc.rs`](../tests/bench_pass2_adhoc.rs)
  (`#[ignore]`), best-of-9 median **and min-time** (the min is the least-perturbed
  sample — the honest ceiling on an interactive box), 3-iteration warmup, ~45 M-element
  (Llama-FFN-sized) synthetic fixtures. Two builds of the same binary:
  `-C target-cpu=native` (the local CLAUDE.md perf-gate baseline) and
  `-C target-cpu=x86-64` (SSE2 — the default-PyPI-wheel baseline). `bench_memcpy_ceiling`
  establishes the practical single-core streaming ceiling; each kernel's GB/s is compared
  against it. Traffic counted as read + write bytes per element.
- **asm audit** — `cargo asm --release` (cargo-show-asm 0.2.62) on each public dequant
  entry, counting packed (`vmulps`/`ymm`) vs scalar (`vmulss`) float arithmetic.
- **Machine:** AMD Ryzen 9 5950X (Zen 3 — AVX2 + FMA, no AVX-512),
  `x86_64-pc-windows-msvc`, warmed working set.

**Result — roofline (min-time GB/s, read+write traffic):**

| Kernel | native | SSE2 | fraction of memcpy ceiling | classification |
|---|---:|---:|---:|---|
| `memcpy_ceiling` | 9.7 | 11.3 | 1.00 | (the ceiling) |
| **pure pass-2 writer** (isolated `write_scratch_to_bf16` loop) | 8.4 | 9.4 | **0.83–0.87** | **bandwidth-bound** |
| GGUF `Q8_0` (light unpack) | 3.3 | 5.2 | 0.34–0.46 | compute-bound |
| GGUF `Q4_0` (nibble unpack) | 2.7 | 3.5 | 0.28–0.31 | compute-bound |
| FP8 per-tensor (fused decode) | 3.2 | 3.3 | 0.29–0.33 | compute-bound |

**Result — asm audit (`-C target-cpu=native`, per-kernel arithmetic in the loop body):**

Read via `cargo asm --release "<fn>" 0` — selecting the **function body by index** is
required: bare `cargo asm "<fn>"` prints a *candidate list* (the function plus its iterator
closures) that a naive `grep` mistakes for disassembly. The first pass of this experiment
made exactly that error and briefly recorded "0 `ymm`, fully scalar" for AWQ/GPTQ; the
corrected readings below select index 0.

| Kernel loop | Packed AVX2 arithmetic? | Evidence (index-0 body) |
|---|---|---|
| FP8 per-tensor fused kernel | **yes, main loop** | `vmulps` (vectorized body) + `vmulss` (scalar tail) |
| GPTQ pass-2 `(qw−zero)×scale → bf16` | **yes, partial** | 3×`vsubps`/`vmulps` + 5×`vsubss`/`vmulss` (scalar tails / loop versions) |
| AWQ pass-2 (same shape) | **yes, partial** | 3×`vsubps`/`vmulps` + 5×`vsubss`/`vmulss` |

So the decode arithmetic is **already partially auto-vectorized in all three kernels** — the
premise that the four-way `Zip` chain leaves it fully scalar is **false**. A naive fission
of AWQ's pass-2 into a separate indexed arithmetic loop *removed* the packed ops (dropped to
`1×vsubss`/`1×vmulss`) — a regression — and was reverted.

**Why the ROADMAP's thesis does not hold as stated (and where the headroom actually is):**

1. **The isolated pass-2 writer is bandwidth-bound** — 83–87 % of the memcpy ceiling on both
   builds. It reads 4 B (f32) and writes 2 B (BF16) per element with trivial compute; an AVX2
   rewrite cannot exceed DRAM bandwidth. Replacing *only* the shared writer (the ROADMAP's
   marquee `f32x8_to_bf16x8` helper) yields ~0 on a `native` build. Its remaining value is
   narrow but real and distribution-side: the Phase 8 **SSE2-fallback wheel** cannot use
   `target-cpu=native`, so a runtime AVX2 dispatcher keeps that wheel fast on capable CPUs.
2. **The decode kernels are compute-bound (0.3–0.46× ceiling), but not because the arithmetic
   is scalar** — it is already vectorized. The cost lives in the parts that *don't* vectorize:
   the per-element **unpack** (byte/nibble extraction; AWQ additionally scatters through
   `AWQ_ORDER`) and the **f32→BF16 convert + 2-byte strided store** (`.to_bits()` → integer
   RNE → `to_le_bytes()` → `chunks_exact_mut(2)`), plus scalar remainder loops. Squeezing
   these is genuinely harder than "refactor a scalar zip-chain" — it needs either careful
   restructuring that survives asm verification, or explicit `#[target_feature]` intrinsics.
3. The inversions where the SSE2 build **beats** native on bandwidth-bound loops (writer 9.4
   vs 8.4; `Q8_0` 5.2 vs 3.3) reinforce point 1: on Zen 3, wider AVX2 codegen buys nothing
   when the loop is memory-bound.

**Disposition:** the **roofline conclusion stands** (writer bandwidth-bound; decode
compute-bound), but the Stage-1 plan built on the erroneous "decode arithmetic is scalar"
reading is **withdrawn** — the arithmetic already vectorizes, so there is no free
zip-chain-refactor win, and one such attempt regressed the asm. Direction for Phase 7 is
re-opened with the corrected evidence: the realistic levers are (a) vectorizing the
**convert + strided store** and/or the **unpack** (harder; needs measured, asm-verified
restructuring or explicit intrinsics), and (b) the runtime-dispatched writer for the SSE2
wheel (small, distribution-motivated). No source change ships from Stage 0; the only Stage-0
artifact is `tests/bench_pass2_adhoc.rs` (the committed roofline harness).

**Re-attempting this requires:** N/A for the roofline — it is the gating measurement. Any
future kernel-vectorization claim must (i) read asm via the **index-0 selector**, (ii) hold
0 ULP against the PyTorch cross-validation fixtures, and (iii) show a best-of-N release
median win on a real fixture, or be reverted and logged here.

### Part 2 — explicit-AVX2 exhaustion (the ROADMAP centerpiece, measured)

**Question:** with the roofline saying the writer is bandwidth-bound, does a *real*
hand-written AVX2 `f32x8_to_bf16x8` — the ROADMAP's marquee helper — beat the (already
auto-vectorized) scalar writer? This is the definitive test before committing any product
`unsafe`.

**Method:** [`tests/bench_pass2_adhoc.rs`](../tests/bench_pass2_adhoc.rs) —
`avx2_write_bf16` (RNE via the same `0x7FFF + lsb` bias, `_mm256_packus_epi32` +
`permute4x64` pack), **validated 0-ULP** against the scalar oracle over 100 003 elements
(prime → exercises the scalar tail) in `avx2_writer_is_bit_exact`. Timed **within one job on
one buffer** (scalar then AVX2 back-to-back → the ratio cancels the runner-CPU confound),
best-of-9 min-time, under `-C target-cpu=native` and `-C target-cpu=x86-64` (SSE2).

**Result:**

| Build | scalar | AVX2 | AVX2 speedup |
|---|---:|---:|---:|
| `native` (scalar already auto-AVX2) | 20.2 GB/s | 20.5 GB/s | **1.02×** |
| `x86-64` / SSE2 (scalar has no AVX2 — the wheel case) | 19.9 GB/s | 20.7 GB/s | **1.04×** |

**Verdict: null.** A bit-exact explicit AVX2 writer gains ~2–4 %, inside the noise, on both
builds. The writer is bandwidth-bound, so even the SSE2 case — where the scalar path cannot
auto-vectorize — sees no benefit, which also **undercuts the SSE2-wheel rationale** for the
helper. The ROADMAP's "SIMD the pass-2 writer for 4–8×" is disproven with a real intrinsic.

**Corroborating evidence that explicit SIMD is exhausted across the kernel set:**

- **FP8 fused kernel** already carries `// VECTORIZED: confirmed ... AVX2 vmulps+vpackusdw`
  ([`src/remember/fp8.rs`](../src/remember/fp8.rs)) — the compiler already emits AVX2 — yet it
  stays compute-bound (3.2 GB/s), so the cost is the ~dozen-op-per-element E4M3 decode, not a
  missing vectorization intrinsics could add.
- **GPTQ/AWQ** pass-2 arithmetic already partially vectorizes (`vsubps`/`vmulps`, Part 1).
- **GGUF kernel sweep** (`bench_gguf_kernel_sweep`, 8.4 M elems, BF16-output MB/s): the slow
  kernels are the **IQ2/IQ3 lattice codebooks** (IQ2_XS 2.1, IQ3_S 2.1 GB/s vs ~5 for
  legacy/K-quants) — they are **gather-bound**, and AVX2 gather (`vpgatherdd`) is slow on
  Zen 3, so SIMD does not help the slowest kernels either.

**Disposition: SIMD conclusively exhausted; no product `unsafe` shipped** (per CLAUDE.md
"no measured win → no commit"). The AVX2 prototype stays in the `#[ignore]` bench as the
reproducible exhaustion proof, not in the library. The measured headroom lives elsewhere:
the kernels are **single-threaded on a 16-core CPU** and embarrassingly parallel — the next
investigation (multi-threaded dequant) targets that.

**Re-attempting this requires:** a kernel that is measured compute-bound AND *not* already
auto-vectorized AND *not* gather-bound — none found in the current dequant set.

---

## Experiment 11 — multi-threaded dequant, prototype scaling

**Question:** Experiment 10 identified multi-threading as the real lever (kernels are
single-threaded on a 16-core 5950X, embarrassingly parallel). How far does it actually
scale — is it the "4–16×" a naive core-count argument suggests?

**Method:** [`tests/bench_pass2_adhoc.rs`](../tests/bench_pass2_adhoc.rs), `std::thread::scope`
(no dependency), best-of-9 median, `-C target-cpu=native`, 5950X. Two kernels bracketing the
arithmetic-intensity range, plus a deliberate allocation-strategy contrast:

- **FP8 per-tensor** (memory-heavy: 1 B in → 2 B out, light compute) — two variants: one
  where each thread allocates its own output `Vec` (the naive API-per-chunk shape), and one
  where a single output buffer is **pre-allocated and split via `split_at_mut`** (the
  `CONVENTIONS.md` disjoint-output-region pattern — no per-thread alloc, no zero-fill).
- **IQ3_S** (compute/gather-heavy: lattice-codebook lookups) via the public Vec API.

Determinism is asserted (8-way parallel concat == sequential bytes) before timing.

**Result (speedup vs 1 thread, median):**

| threads | FP8, `Vec`-per-thread | FP8, disjoint slices | IQ3_S (compute/gather) |
|---:|---:|---:|---:|
| 1 | 1.00× (30.6 ms) | 1.00× (20.6 ms) | 1.00× (8.3 ms) |
| 2 | 1.79× | 1.93× | 1.75× |
| 4 | 1.70× | 2.72× | 3.11× |
| 8 | 1.79× | 2.86× | 3.86× |
| 16 | 2.23× | **2.92×** | **4.02×** |

**Findings:**

1. **The realistic win is ~3–4×, not core-count-linear.** Both kernels plateau far below 16
   threads — FP8 (memory-heavy) at ~4 threads (aggregate memory bandwidth ~19 GB/s here),
   IQ3_S (compute-heavy) a bit higher at ~8 threads (~4×). Aggregate DRAM bandwidth and the
   all-core turbo-clock drop are the ceilings, not the core count. This **corrects an earlier
   "4–16×" over-estimate** — the honest figure is **~3× (memory-bound) to ~4× (compute-bound)**.
   Still a large, worthwhile win versus SIMD's measured 1.0×.
   - **The limiting resource is DRAM bandwidth, not disk or ALU** — the buffers are in-RAM
     (no I/O in the timed loop) and larger than the 64 MB L3, so every element round-trips to
     main memory. On this 5950X the ceiling is dual-channel DDR4 through one memory controller,
     amplified by write read-for-ownership (each output store ≈ 2× bus traffic). It is therefore
     **host-dependent**: an 8-channel DDR5 server has 5–10× the bandwidth and would scale
     further — the reason the thread budget is a caller-tunable default (`min(cores, 4)`), not a
     hard-coded core count. **Design implication:** 4 threads captures ~90 % of the desktop win
     while leaving the host's other cores free — a good-citizen default for an embeddable library.
2. **The allocation strategy matters — the `CONVENTIONS.md` disjoint-slice rule is load-bearing.**
   Naive `Vec`-per-thread caps FP8 at **2.23×** (each thread `vec![0; n]` zero-fills its output,
   double-writing memory and serializing on the allocator); pre-allocating once and writing
   disjoint `split_at_mut` slices lifts it to **2.92×** and drops the 1-thread time 30.6→20.6 ms
   (no zero-fill). A real implementation must write into caller-provided disjoint output
   regions, not allocate per task.
3. **Compute-heavier kernels scale better** (IQ3_S 4.0× > FP8 2.9×), as predicted: lower memory
   traffic per unit work means they hit the bandwidth wall later.

**Disposition:** prototype only — no product code yet. Confirms multi-threading is the
worthwhile Phase-7 lever (~3–4×) and validates the disjoint-output-slice design the new
[`CONVENTIONS.md`](../CONVENTIONS.md) "When Parallelizing Work" section mandates. A product
implementation needs a slice-writing (not `Vec`-returning) internal dequant entry, a
caller-controlled thread budget (never derived from file-declared counts), and a
thread-count-invariant determinism test.

**Re-attempting this requires:** N/A — this is the gating prototype. The product
implementation's per-kernel scaling lands here as measured, against the sequential baseline.

---

## Experiment 12 — Phase 7.2 GGUF-reader parallelisation (product), threshold calibration, and the `gguf-py` baseline

**Questions.** Three, answered in order:

1. Does bringing `convert::read_gguf` under the shared thread pool actually pay, and how much?
2. What should `parallel::MIN_PARALLEL_BYTES` — the named size threshold `CONVENTIONS.md`
   rule 3 requires — be set to, measured rather than guessed?
3. Since Phase 8 puts this path in front of a Python audience, how does it compare to the
   Python stack a user would otherwise reach for?

**Method.** 5950X (16 cores), Windows 11, `-C target-cpu=native`, release, best-of-5 median
per `CLAUDE.md` § Performance Changes. Two harnesses, because the interesting quantity is not
the one the public API exposes:

- [`src/convert.rs::hub_scaling_bench`](../src/convert.rs) — times **`read_hub` alone**. It is
  an in-crate `#[ignore]`d test because `read_hub` is private; an integration test cannot reach it.
- [`tests/bench_gguf_convert_adhoc.rs`](../tests/bench_gguf_convert_adhoc.rs) — times the public
  end-to-end `convert()`, and separately measures scoped-pool spawn/join cost.

Real fixtures (gitignored, see `tests/fixtures/gguf_reference/generate_gguf.py`):
`SmolLM2-135M-Instruct-Q4_K_M.gguf` (100.6 MiB) and `tinyllama-1.1b-chat-v1.0.Q5_0.gguf`
(731.5 MiB). **Determinism is asserted before any timing** — the 1-thread and 8-thread outputs
must be byte-identical (269 MB and 2.2 GB respectively) or the test fails instead of reporting
numbers. `threads = 1` *is* the honest "before": the reader was unconditionally sequential
through v0.7.1.

### Result 1 — stage-isolated `read_hub` scaling

| threads | SmolLM2 Q4_K_M | speedup | TinyLlama Q5_0 | speedup |
|---:|---:|---:|---:|---:|
| 1 | 103.88 ms | 1.00× | 694.57 ms | 1.00× |
| 2 | 72.28 ms | 1.44× | 502.40 ms | 1.38× |
| 4 | 54.70 ms | **1.90×** | 355.76 ms | **1.95×** |
| 8 | 72.24 ms | 1.44× | 331.58 ms | 2.09× |
| 16 | 59.45 ms | 1.75× | 319.71 ms | 2.17× |

(The SmolLM2 8-thread row is background-load noise on an interactive machine — its *min* was
62.3 ms, in line with the 4- and 16-thread rows. The medians either side bracket it.)

### Result 2 — end-to-end `convert()` is much flatter, and that is not a bug

| threads | SmolLM2 → safetensors | TinyLlama → safetensors |
|---:|---:|---:|
| 1 | 219.99 ms (1.00×) | 1886.38 ms (1.00×) |
| 4 | 184.55 ms (1.19×) | 1482.59 ms (1.27×) |
| 16 | 180.01 ms (1.22×) | 1391.54 ms (1.36×) |

**Amdahl, measured.** `convert()` = read + dequantise (parallelised) + **write the whole hub to
disk** (not parallelised, and not parallelisable — it is one sequential file). For TinyLlama the
output is 2.2 GB; at 1 thread the dequant stage is 695 ms of the 1886 ms total, so ~63 % of the
wall clock is the write. Parallelising the remaining 37 % perfectly would cap the end-to-end gain
at ~1.6×; the measured 1.36× is that ceiling minus allocator and page-fault cost.

**Host-dependence, added 2026-08-15.** This figure does not survive a change of
machine, and that is worth stating where the number lives rather than only in a
later entry. CodSpeed walltime on macro runners measured the same benchmark at
**167.60 → 168.41 ms** from 1 to 4 threads: **0.99×, no scaling at all**. The
explanation is consistent with the Amdahl argument above rather than in tension
with it — if the write is ~60 % of wall clock on this host, and CI storage is
slower, the write is simply ~100 % of it there. The stage-isolated `read_hub`
figures (1.90×/1.95×) are the ones that describe the *code*; the end-to-end
figure describes the code **plus the storage it was measured on**. Quote them
accordingly, and do not put a single blended threading number on a README or a
PyPI page.

**This is the number to quote for `convert()`, and the 1.9× is the number to quote for the
reader.** Conflating them would overstate what a user sees.

### Result 3 — why ~2× and not Experiment 11's ~3–4×

Experiment 11 measured two allocation strategies and found `Vec`-per-thread capped FP8 at
**2.23×** while pre-allocated disjoint `split_at_mut` slices reached **2.92×**. The `GGUF` hub
is a `Vec<u8>` **per tensor** by construction — `HubTensor` owns its bytes — so this path is
structurally the `Vec`-per-task case, and it lands exactly where Experiment 11 predicted that
case would: ~2×, not ~3×.

Reaching the disjoint-slice number would mean pre-allocating one contiguous buffer for the whole
model and handing out sub-slices — i.e. redesigning `Hub` away from per-tensor ownership. That is
a real option, but it is a **memory-layout change with an API blast radius**, not a threading
tweak, and it trades against the Phase 6.14 copy-elimination work. Not attempted here; recorded
as the known ceiling.

### Result 4 — `MIN_PARALLEL_BYTES` calibration

Scoped-pool spawn **and** join, empty workers, 200 cycles per sample:

| workers | cost / pool cycle |
|---:|---:|
| 2 | 118.6 µs |
| 4 | **236.0 µs** |
| 8 | 329.2 µs |
| 16 | 655.3 µs |

Windows thread creation is expensive — roughly 2× what a Linux figure would suggest. With the
1-thread `read_hub` rate of ~1.05 GB/s and a 1.9× parallel speedup, the break-even is
`work × (1 − 1/1.9) > 236 µs` → **~0.5 MiB** of input.

**Set to 4 MiB** — break-even with an ~8× margin. At exactly the threshold the pool still nets
~1.7× and costs under ~6 % overhead, while a mis-set value can only ever waste a few hundred
microseconds. v0.7.0's dispatch had no byte threshold at all (only `count > 1`), so an
eight-tensor toy model paid four thread spawns to dequantise a few KiB.

> **Consequence for tests.** The dispatch runs sequentially below the threshold, so any test or
> benchmark meant to exercise the *parallel* path must use a fixture above it. `benches/convert.rs`
> was initially written with a 2.64 MiB GGUF fixture and reported **identical** times at 1 and 4
> threads — a healthy-looking benchmark measuring nothing. The in-crate determinism tests size
> their fixture off the constant itself (`fixture_crosses_the_parallel_threshold` asserts it) so
> they cannot silently decay this way.

### Result 5 — versus the Python baseline

Baseline: **`gguf-py` 0.18.0** (`gguf` on PyPI, maintained in the llama.cpp tree). It is the
right comparison because it is the only general-purpose GGUF dequantiser on PyPI, its kernels
are NumPy-vectorised rather than naive Python loops (so this is the *strong* baseline), and the
two libraries provably agree on the numbers.

**What "agree" means, precisely — this is not a like-for-like output.** `gguf-py`'s `dequantize()`
returns `float32`; anamnesis returns `BF16` (`TargetDtype` has exactly one variant). The
cross-validation therefore takes gguf-py's `float32`, rounds it to `BF16` round-to-nearest-even
(`f32_array_to_bf16_bytes` in the fixture generators), and asserts anamnesis matches at
**`max_ulp = 0` across all 22 kernels** (`tests/cross_validation_gguf.rs`,
`tests/cross_validation_ollama.rs`). So anamnesis is bit-identical to *gguf-py correctly rounded
to `BF16`* — same values, narrower delivered type. Both sides do their block arithmetic in `f32`
internally, so the difference is the **output width**, not the kernel's precision. Script:
[`tests/fixtures/gguf_reference/generate_gguf_dequant_timings.py`](../tests/fixtures/gguf_reference/generate_gguf_dequant_timings.py);
the sidecar JSONs are committed, so the Rust harness prints the comparison without a Python
environment.

| Model | `gguf-py` | anamnesis 1 thread | anamnesis 4 threads |
|---|---:|---:|---:|
| SmolLM2-135M Q4_K_M | 2894.6 ms | 103.9 ms — **27.9×** | 54.7 ms — **52.9×** |
| TinyLlama-1.1B Q5_0 | 12031.4 ms | 694.6 ms — **17.3×** | 355.8 ms — **33.8×** |

**Caveats, both to be stated wherever these numbers are quoted:**

1. **Output width.** gguf-py writes **2× the output bytes** (`float32` vs `BF16`) — a real handicap
   on a bandwidth-bound workload that is *not* attributable to Python. Halving gguf-py's time as a
   deliberately generous width correction still leaves **~8.7–14× single-threaded** and
   **~17–26× at the default budget**.
2. **Threading.** gguf-py is single-threaded, so the like-for-like row is the 1-thread column.

And one **functional** difference that is not a caveat on the measurement but matters for Phase 8:
a caller who needs `float32` cannot get it from anamnesis today — `BF16` is the only target dtype.
They would upcast on the Python side (cheap, but their cost, and it gives back part of the 2×
byte advantage). `docs/python-interop.md` already frozen the `ml_dtypes.bfloat16` return contract
this implies.

**Disposition:** shipped. The stage gain (~1.9× at the default budget) is a measured win in the
expected direction on real fixtures, byte-identical across `{1, 2, 4, 8, 16}` threads and the
hardware-resolved default, with the end-to-end figure reported separately rather than folded in.

**Re-attempting this requires:** a `Hub` redesign toward one contiguous model-wide buffer with
per-tensor sub-slices (the Experiment 11 disjoint-output pattern) — the only identified route
past the ~2× `Vec`-per-tensor ceiling. Measure against the 4-thread medians above, and weigh the
result against the Phase 6.14 copy-elimination gains it would disturb.

---

## Experiment 13 — Phase 7.3 caller-chosen output dtype: what `F32` and `F16` cost

**Question.** Phase 7.3 makes the `GGUF` dequantisation output width a caller-chosen parameter.
`F32` doubles the output bytes on a path Experiment 12 showed is bandwidth-bound, so it is
*expected* to be slower. How much slower, does the default `BF16` path regress, and does the
Phase 7.2 threading ratio hold?

**This is a capability, not a perf claim.** `CLAUDE.md`'s "no measured win, no commit" rule
governs *perf-claim* commits. This phase claims **exactness**: `F32` output removes anamnesis's
own narrowing step, so the emitted value is the reference's own `f32`. A correct-but-slower
`F32` path is the point of it. The numbers below are recorded as the honest cost of the new
option, not as a result to be optimised before shipping.

**Method.** Criterion, release, `RUSTFLAGS="-C target-cpu=native"`, 5950X / Windows 11.
Kernel level: `cargo bench --features "gptq awq bnb gguf" --bench dequant -- dequant_gguf_q4_k`
on a synthetic 4096 × 11008 `Q4_K` tensor (45 088 768 elements). End to end:
`cargo bench --features gguf --bench convert -- convert_gguf_to_safetensors` on the ~10.6 `MiB`
quantised `GGUF` fixture, which includes the output file write.

### Kernel level (`dequant_gguf_q4_k`)

| Output dtype | median | range | throughput | vs `BF16` |
|---|---:|---|---:|---:|
| `BF16` | **20.300 ms** | [19.575, 21.142] | 2.2211 Gelem/s | 1.00× |
| `F32` | **36.298 ms** | [34.508, 38.283] | 1.2422 Gelem/s | **1.79× slower** |

**1.79× against 2.00× of output bytes.** The cost is very nearly the doubled write and nothing
else, which is what "the kernels already compute in `f32`, `F32` output simply stops narrowing"
predicts: pass 1 is unchanged, and pass 2 writes twice as much. The shortfall from 2.00× is the
shared input read and unpack.

### End to end (`convert_gguf_to_safetensors`, includes the output write)

| Output dtype | 1 thread | 4 threads | threading gain |
|---|---:|---:|---:|
| `BF16` | 17.939 ms | 15.134 ms | 1.19× |
| `F32` | 27.694 ms | 24.376 ms | 1.14× |
| **`F32` vs `BF16`** | **1.54×** | **1.61×** | |

Two things worth reading off this table. End-to-end `F32` costs **less** than the kernel-level
1.79×, because the fixed parse and file-open costs do not scale with output width. And the
**threading ratio shifts down**, 1.19× → 1.14×, exactly as predicted: `F32` moves more bytes per
unit of compute, so the path is more bandwidth-bound and threading buys slightly less. Phase 7.2's
ratios remain valid for `BF16`; do not quote them for `F32`.

### The `BF16` default did not regress

The generic refactor makes `Bf16Out::write_scratch` byte-for-byte the loop that shipped as
`write_scratch_to_bf16`, so the default path's codegen should be unchanged by construction. The
measurements agree: the two end-to-end `BF16` arms report `p = 0.41` and `p = 0.13` against their
stored baselines, i.e. **no statistically significant change**. The kernel arm reported a nominal
−9.8 % (`p = 0.02`); that is recorded rather than claimed as a win, since nothing in the change
plausibly makes `BF16` faster and run-to-run variation on an interactive machine is the more
likely explanation.

### Vectorisation, verified per implementation

`RUSTFLAGS="-C target-cpu=native --emit=asm"`, inspecting each `dispatch_streaming::<E, _>`
monomorphisation. This is why the trait carries a block-level `write_scratch` rather than a
per-element hook: three independent loops, three independent verdicts.

| Impl | Evidence | Width |
|---|---|---|
| `Bf16Out` | `vpaddd`, `vpsrld`, `vpand`, `vmovdqu` on `%ymm` | 8-wide AVX2 |
| `F32Out` | `vmovups` on `%ymm` | 8-wide AVX2 stores |
| `F16Out` | `vcvtps2ph $0, %xmm, %xmm` (F16C) | 4-wide packed |

All three are packed; none fell back to scalar. `F16Out` is narrower because `vcvtps2ph` takes a
128-bit source, which is a hardware property rather than a missed vectorisation. Its `$0`
immediate is round-to-nearest-even, so the documented `F16` rounding is enforced by the
instruction itself.

### No `F16` throughput arm, deliberately

`F16` is the same width as `BF16`, so there is no bandwidth story to tell, and a benchmark would
only re-measure `BF16`'s number with a different conversion instruction. `F16`'s interesting
properties are accuracy (11 significand bits against 8) and exponent range (saturating to infinity
above 65504, where `BF16` shares `f32`'s range), and both are **tests**, in
`src/remember/output.rs`, not benchmarks.

### Memory, verified to the byte

`tests/peak_heap_gguf.rs` (`dhat`, release, 45 M-element `Q4_K`):

| Path | `BF16` | `F16` | `F32` |
|---|---:|---:|---:|
| owned `Vec` peak | 90 177 536 B | 90 177 536 B | 180 355 072 B |
| overhead above output | **0 B** | **0 B** | **0 B** |
| streaming peak | **0 B** | — | **0 B** |

Peak equals output exactly at every width, and `F32` is exactly twice `BF16`. The streaming entry
point allocates nothing at all on the heap for a 45 M-element tensor, which is its `# Memory`
claim confirmed literally rather than argued.

### Monomorphisation cost

Three output types across the two block runners duplicate two small writer loops, not the 24
kernel functions, because every kernel funnels through one shared pass-2 writer. Release
`libanamnesis` binary size is unchanged at the resolution `ls` reports; the `.crate` stayed at
0.60 MiB. This economy is specific to `GGUF` and is precisely why Phase 7.4 (the `remember`
path's four fused-narrowing families) is a separate tag.

**Verdict:** shipped (v0.7.3, Phase 7.3). `F32` costs 1.79× at the kernel and 1.54–1.61×
end to end, for output that is bit-identical to `gguf-py`'s own `f32`. Optimising it is a
follow-up, not a gate.

---

## Experiment 14 — Phase 7.4 `remember`-path output dtype: the split's cost, and a bench measuring the wrong crate

**Hypothesis under test:** that making the four `remember`-path kernel families
(`FP8`, `GPTQ`, `AWQ`, `BnB`) generic over the output element — splitting each
fused per-element loop into an arithmetic pass plus a `write_scratch` narrowing
pass — leaves the `BF16` default path's performance essentially unchanged.

**Method:** two ad-hoc benches, both release, `target-cpu=native`, binaries
built once and run **alternately** (never A-then-B), min across rounds. Baseline
binary at `6697599`, the commit before the phase opened.
[`tests/bench_dequant_adhoc.rs`](../tests/bench_dequant_adhoc.rs)
`bench_bf16_all_families` (isolated kernels) and
`bench_remember_whole_model_threaded` (whole model, 4 × 4096² fine-grained
`FP8` through `remember_to_bytes_with_options`).

### The two benches disagreed by ~1.9×, and the disagreement was the finding

The isolated bench reported the split costing ~1.06×; the whole-model bench
reported it *gaining* 0.56×. The tell was arithmetic, not statistical: end to
end ran at **1.003 ms/Melem** while the isolated kernel it *contains* ran at
**1.547**. A path doing strictly more work (parse, per-tensor dispatch,
`build_views`, `serialize`) cannot beat one of its own components.

**Root cause: the ad-hoc bench in `tests/` is an *external* crate.** Before
v0.7.4, `dequantize_fp8_to_bf16` was a plain non-generic `pub fn` — one
lib-compiled, fully optimised symbol called opaquely from outside. v0.7.4 made
it an `#[inline]` wrapper around `dequantize_fp8::<Bf16Out>`, so an external
caller inlined the wrapper and instantiated the generic *in its own crate*,
where `e4m3_to_f32_bits`, `e4m3_to_scaled_f32`, `f32_bits_to_bf16_bits` and
`unpack_gptq` were private, non-`#[inline]` and therefore opaque cross-crate
symbols: a function call per element. `remember` never suffered it, calling the
generic from inside the library where everything inlines. Both benches were
correct about different things.

**This was a real production bug, not a harness artefact.** Any downstream crate
calling `anamnesis::dequantize_*_to_bf16` would have got the un-inlined version,
roughly **2.9× slower than v0.7.3** for the same call. Fixed by `#[inline]` on
those four helpers (commit `d7bd7fa`):

| arm | without the fix | with the fix |
|---|---:|---:|
| `kernel_fresh_alloc` (4 × 4096²) | 1.934 ms/Melem | **0.672** |
| `kernel_single_call` | 2.013 | **0.731** |

### The honest numbers, after the fix

Isolated kernels, interleaved, min of 4:

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

Mutually consistent at last: `FP8` gains 2.2× at the kernel, diluted to ~1.8×
end to end by serialization, because the `VECTOR_TILE` restructure vectorises
better than the per-element fused loop it replaced. The other three families pay
a genuine **1.06–1.15×**, the honest cost of the arithmetic/narrowing split.

### A second-order result worth more than the first

**Whole-program code layout moves a kernel's timing by up to 23 % on this host.**
`FP8` with byte-identical source measured 1.060× in one build and 1.308× in
another, the only difference being that an *unrelated* module changed size
(recorded in `66bb10b`). This bounds the precision of every per-kernel claim
made on this machine: a sub-20 % difference cannot be attributed to the source
that was edited without an independent control.

**Method notes, learned the hard way, and binding on the next ad-hoc bench:**

- Alternate the binaries. A-then-B on this host gave *inverted* results.
- Take the **min** across rounds, not the median of one round.
- Watch for per-iteration allocation dominating: these kernels `vec![0u8; n]` a
  90 MB output per call, so a naive loop measures page faults as much as maths.
- Once a public entry point becomes a thin generic wrapper, a bench in `tests/`
  stops measuring what the library executes internally. Either mark every
  hot-loop helper `#[inline]`, or bench through a whole-model entry point.

**Verdict:** shipped (v0.7.4, Phase 7.4). The split is a net win, and the
investigation that looked like chasing a 5 % regression surfaced a 2.9×
downstream bug instead.

---

## Experiment 15 — Phase 7.4 `BnB` `INT8`: bit-exactness at `F32` versus one packed multiply

**Hypothesis under test:** that anamnesis' `BnB` `INT8` dequant, validated at
0/65536 `BF16` mismatches against `bitsandbytes` since v0.5.0, is also
bit-exact at `F32`.

**It is not, and the reason is instructive.** anamnesis hoisted a per-row scale
and computed `w × (SCB / 127)`. `bitsandbytes`' `int8_vectorwise_dequant` is
`A * stats.view(-1, 1) * 7.874015718698502e-3` — the same real number, a
different `f32` evaluation. Measured on the committed `llama_1b_int8` fixture:

| comparison | mismatches vs the canonical `f32` output |
|---|---:|
| `(w × SCB) × c` — `bitsandbytes`' own association | **0 / 65536** |
| `w × (SCB × c)` — the obvious hoist | 17299 / 65536 |
| `w × (SCB / 127)` — what anamnesis shipped | **17610 / 65536 (26.9 %)** |

Every mismatch was exactly **1 `ULP`**, and every one vanished on narrowing to
`BF16` — which is precisely why five releases of `BF16` cross-validation never
saw it. The constant is *not* the variable: `f32(7.874015718698502e-3)` and
`f32(1.0 / 127.0)` are the same bits (`0x3C01_0204`), now asserted at compile
time in `src/remember/bnb.rs`. Only the association matters.

**The fix costs one multiply per element** — the reciprocal can no longer be
folded into a per-row scale, because folding it *is* the rejected
`w × (SCB × c)` association. Measured with the interleaved protocol from
Experiment 14, `bench_bf16_all_families` `bnb_int8` at 4096 × 11008, min over
**10** interleaved before/after rounds (the box was noisy enough that two rounds
were visibly contaminated — 37.8 ms and 44.9 ms against a ~17 ms floor — so the
min is the only defensible statistic):

| variant | min over 10 rounds |
|---|---:|
| `w × (SCB / 127)` (v0.7.3) | **16.82 ms** |
| `(w × SCB) × INV_127` (v0.7.4) | **17.48 ms** |

**1.04× slower**, i.e. +0.66 ms. Confirmed in the disassembly as exactly what
was intended: all three monomorphisations dump 6 `vpmovsxbd`, 6 `vcvtdq2ps` and
**12 `vmulps`** — two packed multiplies per sign-extending load — with no
`vdivps`/`vdivss` anywhere, so the v0.7.3 per-row *division* is gone rather than
merely relocated. The ratio also flatters the arithmetic: the kernel allocates a
90 MB output inside the timed region, so page-fault cost sits in the denominator
(see Experiment 14's method notes). The trustworthy figure is the absolute
+0.66 ms.

**Why it shipped despite a measured slowdown.** `CLAUDE.md` scopes the "no
measured win, no commit" rule to *perf-claim* commits;
`feedback-capability-before-speed` records that a measured slowdown does not
block a correctness feature. This is a correctness fix: it makes the kernel
bit-exact against the canonical library at every output width, and no `BF16`
output byte changed (all 7 `BnB` fixtures still report 0 mismatches).

**Do not re-propose** hoisting `SCB × INV_127` per row to recover the 4 %. It is
the middle row of the table above, and it is measurably worse than the bug it
would be trying to optimise.

**Verdict:** shipped (v0.7.4, Phase 7.4). The generalisable lesson is the one
this file exists for: **exactness at a narrow output width does not imply
exactness at a wide one.** A `BF16` comparison discards 16 mantissa bits, and on
these fixtures 38–98 % of values carry bits it cannot show (the low end is the
`BnB` `FP4` pair at 38.4 % / 43.9 %, whose 16-entry codebook lands on
`BF16`-representable values more often; `GPTQ`, `AWQ` and `BnB` `NF4` all sit
above 77 %). Every family's `F32` golden in Phase 7.4 exists because of that.

### Postscript — the first CodSpeed data point, and what it corrected

Experiments 14 and 15 were measured locally with the interleaved best-of-N
protocol. The branch was never pushed while the work was in progress, so
CodSpeed saw none of it until the v0.7.4 merge landed on `main`. Its first run
is worth recording, for two reasons.

**It confirmed the `BF16` result on independent hardware.** CodSpeed walltime,
macro runners, `remember_bf16_whole_model`:

| | v0.7.3 | v0.7.4 | change |
|---|---:|---:|---:|
| `threads_1` | 81.353 ms | **78.173 ms** | −3.9 % |
| `threads_4` | 21.207 ms | **20.827 ms** | −1.8 % |

The default path did not regress; it improved slightly. `convert_gguf_to_safetensors`,
which Phase 7.4 never touched, serves as the control and moved **under 0.05 %**
across the same two runs (167.54 → 167.60 ms; f32 315.43 → 315.36 ms). An
untouched series flat to 0.05 % while a touched one moves 3.9 % is what makes
the latter signal rather than drift.

**It corrected a claim this file's own Experiment 13 invites.** That entry
reports `F32` costing 1.79× at the kernel against 2.00× of output bytes, and
concludes "the cost is the doubled write and nothing else". True at one thread.
The new `remember_f32_whole_model` arm shows it does not survive threading:

| | `BF16` | `F32` | `F32`/`BF16` |
|---|---:|---:|---:|
| `threads_1` | 78.17 ms | 165.44 ms | 2.12× |
| `threads_4` | 20.83 ms | 81.60 ms | **3.92×** |

`F32` scales to 4 threads at **2.03×** where `BF16` reaches **3.75×**. Twice the
output bytes saturates memory bandwidth sooner, so the thread budget buys about
half as much, and the `F32` penalty *grows* with thread count instead of holding
at the output-byte ratio. **Do not quote 2× as the expected wall-clock cost of
`F32` above one thread**, and do not quote Phase 7.2's threading ratios for it
either, for the same reason Experiment 13 already warned about at `GGUF`.

Two method notes worth carrying forward:

- The local ad-hoc benches never produced this. Whole-model `remember` at `F32`
  was never measured locally, only the kernels and the `BF16` whole-model path.
  A gap in a bench suite is invisible until something else fills it.
- This is the effect the `walltime` instrument exists for. A CPU-simulation
  instrument counts work, not stalls, so it would have reported `F32` at roughly
  its instruction ratio and missed the bandwidth ceiling entirely — the same
  failure mode `codspeed.yml`'s header warns about for Experiment 10.

---

## Experiment 16 — Phase 7.6 `GGUF` `remember`: what a duplicated sequential path cost

**Date:** 2026-08-20 · **Phase:** 7.6 · **Status:** shipped

### Hypothesis

Not a hypothesis about an optimisation, which is why this entry exists at all.
The 2026-08-20 API-shape audit found that `amn remember model.gguf` and
`amn convert model.gguf --to safetensors` are the *same operation* implemented
twice: `cli.rs::run_remember_gguf` carried 121 lines transcribing what
`convert.rs::read_gguf_as` already did, and the transcription was **sequential**,
took no thread budget (so `--threads` was accepted and discarded), and could not
be called from the library at all. The prediction was therefore mechanical: the
CLI verb should gain exactly the threading the library path already had.

### Method

Both binaries built `--release --features cli,gguf` with
`RUSTFLAGS="-C target-cpu=native"`, from the same working tree (the "before"
binary from `git stash`, so nothing but this change differs). Runs interleaved
before/after to blunt the ~23 % code-layout sensitivity Experiment 14 measured on
this host. Wall clock via PowerShell `Measure-Command`, five runs each, median
reported with the full range.

```powershell
foreach ($i in 1..5) {
  Measure-Command { & $before remember $m --to bf16 -o $out_b }
  Measure-Command { & $after  remember $m --to bf16 -o $out_a }
}
```

### Results

| Fixture | before (median) | after (median) | speedup |
|---|---|---|---|
| `SmolLM2-135M-Instruct-Q6_K.gguf` (132 MiB → 257 MiB `BF16`) | **241 ms** [234, 4754] | **194 ms** [178, 206] | **1.24×** |
| `Qwen2.5-1.5B-Instruct-IQ2_M.gguf` (574 MiB → 2.9 GiB `BF16`) | **5854 ms** [5792, 6358] | **2631 ms** [2559, 2832] | **2.23×** |

Raw runs, in interleaved order:

- 135M before `292 / 241 / 235 / 4754 / 234`, after `206 / 178 / 196 / 193 / 194`
- 1.5B before `5792 / 5908 / 5830 / 5854 / 6358`, after `2631 / 2758 / 2588 / 2559 / 2832`

The 4754 ms outlier in the 135M "before" set is a single host hiccup and is
reported rather than dropped; the median is what the claim rests on, which is
exactly why the protocol specifies a median and not a mean.

**Output is `SHA-256`-identical before and after on both fixtures**, which is the
part that makes the number safe to act on: this is the same computation running
on more threads, not a different one.

### What it says

1. **The gap scales with model size** (1.24× → 2.23×), because the fixed parse
   and write costs amortise. A small fixture *understates* this class of defect,
   which is worth remembering the next time a CI-sized fixture is used to decide
   whether a path is worth threading.
2. **The 2.23× sits inside Experiment 12's band, not outside it.** That
   experiment measured the stage-isolated `read_hub` at 1.90×/1.95× at 4 threads
   and the end-to-end `convert()` at 1.26–1.36× locally, with the output write as
   ~60 % of the wall clock. This measures the *same* end-to-end shape and gets
   more, because the baseline here is fully sequential rather than
   partially-threaded: 2.23× against a 1-thread reader is consistent with a
   reader that scales ~1.9× sitting in front of a write that does not.
3. **The measurement was worth taking even though the direction was obvious.**
   `CLAUDE.md` requires it for a perf-claim commit, and the value delivered was
   the *size* of the claim, not its sign: the audit's earlier best-of-3 default
   build put the same two fixtures at 1.39× and 2.35×, close enough to confirm
   the effect and different enough to show why the protocol pins the build flags.
