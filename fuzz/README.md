# Fuzzing anamnesis

Coverage-guided fuzz harness for the four parser entry points, built on
[`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) + libFuzzer. Each
target feeds **arbitrary attacker bytes** to a parser and asserts the only
acceptable outcomes are `Ok(_)` or a clean `AnamnesisError` — libFuzzer treats
any panic, abort, OOB, or OOM as a crash.

This complements (does not replace) the Phase 6.6 / 6.7 security hardening and
the in-tree regression tests: the caps and guards are the *fix*, the regression
tests *pin* them, and fuzzing *searches* for inputs we didn't think of.

## Targets

Four flavours: **reader/inspect** targets (header + pickle VM, the `HTTP`-range
inspection surface), **path/parse** targets (the full data-extraction path,
materialising the input to a temp file), **limit-enforcement** targets
(Phase 6.8 Step 5) that parse under a `ParseLimits` **derived from the input**,
so the fuzzer co-explores `(malformed file × tightened limits)`, and
**owned-bytes** targets (Phase 6.13 Step 1) that drive the copy-based,
mmap-free full-parse entry points — the path the Python bindings route untrusted
uploads through.

| Target | Entry point | What it exercises |
|---|---|---|
| `fuzz_safetensors` | `parse_safetensors_header_from_reader` | safetensors header parse + cap |
| `fuzz_npz` | `inspect_npz_from_reader` | ZIP central-directory walk + NPY header parser (`NPY_MAX_HEADER_BYTES`) |
| `fuzz_gguf` | `inspect_gguf_from_reader` | metadata KV / typed-array readers + `read_bytes`/`ensure_remaining` |
| `fuzz_gguf_front_matter` | `parse_gguf_front_matter_from_reader` | the **full-detail** GGUF front-matter path (v0.7.1) — same core as `fuzz_gguf`, but the parsed tensor names and metadata values are handed to the caller instead of reduced to counts, so every string/array survives the call |
| `fuzz_pth` | `inspect_pth_from_reader` | **ZIP walk + the pickle VM** (opcodes, `GLOBAL` allowlist, memo/mark stacks, recursion) — the highest-value target |
| `fuzz_zip` | `parse_pth` + `parse_npz` + `inspect_pth_from_reader` | the **vendored ZIP central-directory reader** (Phase 6.12): EOCD scan, `ZIP64` resolution, local-header data-offset, per-entry caps — over both mmap and reader substrates. Index-level differential vs the `zip` crate lives in the in-crate unit tests |
| `fuzz_npz_parse` | `parse_npz` (via temp file) | the **data-extraction path**: `read_array_data` + the `entry.size()` cross-check |
| `fuzz_pth_parse` | `parse_pth` + `tensors()` (via temp file) | the **mmap path + tensor extraction**: `build_entry_index`, the `MAX_PKL_SIZE` mmap guard, stride/offset resolution |
| `fuzz_npz_limits` | `parse_npz_with_limits` (via temp file) | **all four `ParseLimits` axes**: single-alloc, cumulative `Budget` (`checked_add`), item-count, decompression-ratio (`checked_mul`) |
| `fuzz_gguf_limits` | `parse_gguf_with_limits` (via temp file) | GGUF limit branches: single-alloc + `Budget` on variable-length reads + scalar-array charge + tensor/KV count gate |
| `fuzz_pth_limits` | `parse_pth_with_limits` + `tensors()` (via temp file) | `.pth` limit branches: the `data.pkl` cap + the pickle-VM `Budget` charged on each owned payload |
| `fuzz_safetensors_bytes` | `parse_bytes` | the **copy-based** (no-mmap) safetensors full parse over owned bytes (Phase 6.13 Step 1) — the recommended untrusted-input path |
| `fuzz_gguf_bytes` | `parse_gguf_bytes` | the **copy-based** GGUF full parse (header + metadata KV + tensor-info) over owned bytes |
| `fuzz_pth_bytes` | `parse_pth_bytes` | the **copy-based** `.pth` full parse (ZIP walk + pickle VM + tensor extraction) over owned bytes |

## Prerequisites — Linux / macOS / WSL (not Windows-MSVC)

libFuzzer needs an LLVM-backed nightly and a C linker; it does **not** build on
Windows-MSVC. On Windows, run this under **WSL** (Ubuntu) — where this harness
was authored and verified:

```bash
rustup toolchain install nightly
cargo install cargo-fuzz
# a C linker (gcc/cc) must be present; clang is not required
```

## Build & run

```bash
# compile-check every target
cargo +nightly fuzz build

# run a target (Ctrl-C to stop, or bound it)
cargo +nightly fuzz run fuzz_pth -- -max_total_time=60 -rss_limit_mb=2048
```

### Seeding the corpus (strongly recommended for the ZIP-based targets)

Random bytes almost never form a valid ZIP, so `fuzz_npz` / `fuzz_pth` reach
the interesting code (the pickle VM!) far faster when seeded from the real
fixtures already in the repo:

```bash
mkdir -p fuzz/corpus/fuzz_pth
cp tests/fixtures/pth_reference/algzoo_*.pth fuzz/corpus/fuzz_pth/
cargo +nightly fuzz run fuzz_pth fuzz/corpus/fuzz_pth -- -max_total_time=300

mkdir -p fuzz/corpus/fuzz_npz
cp tests/fixtures/npz_reference/*.npz fuzz/corpus/fuzz_npz/

mkdir -p fuzz/corpus/fuzz_safetensors
cp tests/fixtures/safetensors_reference/*.safetensors fuzz/corpus/fuzz_safetensors/
```

(The AlgZoo `.pth` fixtures are small — ideal seeds. The torchvision
`resnet*/vit_*` fixtures are large and slow the loop; don't seed with them.)

A crash writes the minimal reproducer to `fuzz/artifacts/<target>/`; replay it
with `cargo +nightly fuzz run <target> fuzz/artifacts/<target>/<file>` and turn
it into an in-tree regression fixture.

## What is and isn't committed

`Cargo.toml`, `fuzz_targets/`, this README, and `.gitignore` are tracked. The
generated `corpus/`, `artifacts/`, `target/`, and `coverage/` directories are
git-ignored. The whole `fuzz/` tree is excluded from the published crate via
`exclude = ["/fuzz"]` in the root `Cargo.toml`, so ordinary users never build
it and it never ships to crates.io.

## Status

Authored and run under WSL2 Ubuntu (nightly `rustc` + `cargo-fuzz` 0.13,
libFuzzer). Smoke campaigns (Phase 6.7): the four reader targets ~20 s unseeded
(~0.2–0.8 M runs each) plus a 62 s seeded `fuzz_pth` campaign (213 k runs,
coverage 1412 / features 5372, corpus 675 inputs, RSS steady ~489 MB); the two
path targets 30 s seeded each (`fuzz_npz_parse` 102 k runs, `fuzz_pth_parse`
103 k runs) — **zero crashes** across all six. The three `*_limits` targets
(Phase 6.8 Step 5) each ran 60 s seeded under WSL2 Ubuntu (nightly + `cargo-fuzz`
0.13.1, libFuzzer): `fuzz_npz_limits` 264 k runs (cov 1454, RSS 497 MB),
`fuzz_pth_limits` 224 k runs (cov 1819, RSS 460 MB), `fuzz_gguf_limits` 469 k
runs (cov 643, RSS 415 MB) — **~957 k executions, zero crashes**, confirming the
`ParseLimits` enforcement branches (`check_alloc` / `Budget::charge_alloc`
`checked_add` / `check_item_count` / `check_decompression_ratio` `checked_mul`)
reject panic-free under input-derived limits.

**Phase 6.12 (v0.6.7, vendored ZIP reader)** — re-run under WSL2 Ubuntu 24.04
(nightly + `cargo-fuzz` 0.13.1): all 10 targets compile (the new `fuzz_zip`
included). Seeded campaigns: `fuzz_zip` 668 k runs / 181 s, `fuzz_npz` 2.85 M,
`fuzz_npz_limits` 995 k, `fuzz_npz_parse` 503 k, `fuzz_pth` 381 k,
`fuzz_pth_parse` 316 k, `fuzz_pth_limits` 193 k — **≈5.9 M executions**, RSS
steady (~440 MB). The campaign **found one crash**: `fuzz_npz_limits` reached a
panic in the `NPY` dtype-descriptor parser (`parse_descr` sliced `&descr[1..]`,
which panics when the first character is a multi-byte UTF-8 codepoint —
pre-existing since the v0.3.0 NPZ parser, `bb58cd4`). Fixed (`descr.get(1..)` →
clean `Unsupported`) with a regression unit test
(`parse_descr_multibyte_first_char_is_clean_error`); the crash input now parses
cleanly and the full re-sweep above is **zero-crash** post-fix. Not yet wired
into CI; a scheduled Linux fuzz job is a candidate follow-up (it needs nightly +
`cargo-fuzz` install in the runner, so it is intentionally kept out of the
stable-only push/PR matrix).

**Phase 6.13 (v0.6.8, owned-bytes full-parse targets)** — re-run under WSL2
Ubuntu (nightly + `cargo-fuzz` 0.13.1): all 13 targets compile (the three new
`fuzz_*_bytes` included). Unseeded 181 s campaigns on the new copy-based
entry points: `fuzz_safetensors_bytes` 9.09 M runs (cov 1374, RSS 524 MB),
`fuzz_gguf_bytes` 8.47 M, `fuzz_pth_bytes` 58.35 M (cov 160, RSS 430 MB —
the pickle-VM working-set floor holding well under the 2 GB limit) —
**≈75.9 M executions, zero crashes**, pinning the panic-freedom invariant
(Phase 6.13 Step 3) on the recommended untrusted-input path. The always-run
counterpart is `tests/no_panic.rs` (a `catch_unwind` battery in stable CI).
