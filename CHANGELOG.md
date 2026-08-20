# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`NPZ` archives holding Fortran-order arrays now parse** (Phase 7.6,
  item 8), rewritten into C-order on the way out instead of rejected with
  `Unsupported`. `NumPy` writes `fortran_order: True` for any array that is
  `F`-contiguous and not `C`-contiguous, which a transposed view is: a plain
  `np.savez("w.npz", w=x.T)` produces one. The rejection was defensible while
  the audience was Rust consumers of `C`-order `SAE` archives; it is a
  first-contact failure for a `pip install anamnesis-quant` user on a file
  `NumPy` wrote unprompted.

  Materialised rather than flagged, deliberately: everything downstream
  (`npz_to_safetensors`, the `convert` hub, any framework loading the result)
  assumes row-major, so an order flag a caller might ignore would be the same
  silent-orientation trap Phase 6.10 already had to fix once for `AWQ`/`GPTQ` —
  and a wrong orientation is not a crash, it is plausible numbers in the wrong
  places. C-order input pays nothing: no branch inside a loop, no second
  buffer. The transposition uses reversed strides, so rank 2 and rank *n* are
  one code path, rank 0/1 and empty arrays are the identity, and the
  destination buffer is charged to the caller's `ParseLimits` before it is
  allocated.

  `inspect_npz*` no longer rejects these archives either: memory order changes
  no shape, dtype or byte count, so an inspect never had anything to reject,
  and refusing meant a host could not run its inspect-before-parse gate on an
  archive it would now parse happily.

- **The summary `inspect` entry points accept a caller `ParseLimits`**
  (Phase 7.6, item 7), through `InspectOptions::with_limits`. They were
  *"intentionally limit-free"* by design, which left the call the README tells
  a multi-tenant host to make **first** on an untrusted file as the one call it
  could **not** tighten, an inversion of the whole premise of `ParseLimits`:
  that a caller can always narrow a permanent floor. `NPZ` was the sharpest
  case, having no bounded alternative at all (`GGUF` and `.pth` at least offer
  `parse_*_front_matter_from_reader_with_limits`), a walk that is `O(entries)`
  in both I/O and allocation, and one floor, `ZIP_MAX_ENTRIES` = `1 << 20`.
  A hostile archive could therefore force a million-entry central directory, a
  million `NPY` header reads, and a `Vec<NpzTensorInfo>` no caller could cap.
  `README.md`'s untrusted-input recipe called that call "bounded"; it now is,
  and the recipe shows how. Pinned by `inspect_npz_honours_the_caller_budget`.

- **`amn remember <file>.gguf --threads N` now honours `N`** (Phase 7.6,
  item 1). The flag reached the safetensors arm only; the `GGUF` arm took no
  thread parameter, so the budget was parsed, validated, and discarded. It now
  routes through `RememberOptions` like every other path, and the output stays
  byte-identical at any thread count (pinned by
  `remember_is_deterministic_across_thread_counts`).

### Changed

- **BREAKING: `InspectOptions` is no longer `Copy`, and the
  `inspect_with_options` family takes it by reference.** It gained a
  `limits: ParseLimits` field, and `ParseLimits` is deliberately
  `Clone`-not-`Copy` (a `Copy` derive would trip `trivially_copy_pass_by_ref`
  on the 16-byte struct). Following the crate's own rule for a non-`Copy`
  argument that is only read, `ParsedModel::inspect_with_options` and its new
  siblings take `&InspectOptions`. Callers add one `&`.

- **BREAKING: every type the crate returns is now `#[non_exhaustive]`**
  (Phase 7.6, item 4). Twenty public structs gain the attribute: the parsed
  headers (`SafetensorsHeader`, `TensorEntry`), the quantisation configs
  (`GptqConfig`, `AwqConfig`, `BnbConfig`) and companion sets
  (`GptqCompanions`, `AwqCompanions`, `Bnb4Companions`), the `GGUF` types
  (`GgufTensorInfo`, `GgufTensor`, `GgufInspectInfo`, `GgufFrontMatter`), the
  `NPZ` types (`NpzTensor`, `NpzTensorInfo`, `NpzInspectInfo`), the `.pth`
  types (`PthTensor`, `PthTensorInfo`, `PthInspectInfo`, `PthFrontMatter`),
  and `BnbNf4WriteStats`. Every public **enum** has carried the attribute
  since it was introduced, and `InspectInfo` was marked at v0.7.4 with the
  rationale *"the Python bindings freeze this shape in Phase 8, and a struct
  whose fields are all `pub` cannot otherwise gain one without a breaking
  change"* — that reasoning applies to all of its siblings, and this release
  applies it. It has to happen now: v0.8.0 mirrors this surface into a `PyPI`
  package where a signature change is far more expensive than on `crates.io`,
  and adding the attribute afterwards would itself be the breaking change.
  Phase 7.6's next commit depends on it directly, since it grows
  `GgufInspectInfo` by two fields.

  **What breaks:** an external crate can no longer write a struct literal for
  these types, nor destructure one exhaustively. Reading fields, cloning, and
  pattern-matching with `..` are all unaffected, which covers every documented
  usage. A side effect worth naming: the external-construction path the
  v0.7.4 readiness audit's F-1 finding leaned on — building a
  `SafetensorsHeader` by hand and passing it to `InspectInfo::from` to bypass
  the upstream `safetensors` validation — is now closed at the type level as
  well as by the saturating arithmetic that fixed it.

  **Deliberately exempt:** `BnbWriteInput` and `GgufWriteTensor` stay
  literal-constructible because callers must build them to call
  `write_bnb_nf4_safetensors` / `write_gguf`. `ParsedModel`, `ParsedGguf` and
  `ParsedPth` are already unconstructible through private fields.

### Added

- **`parse_npz_bytes`, `parse_npz_bytes_with_limits`, `parse_npz_from_reader`,
  `parse_npz_from_reader_with_limits`** (Phase 7.6, item 3). `NPZ` was the one
  format with no in-memory or streamed parse: safetensors, `.pth` and `GGUF`
  have each had a `parse_*_bytes` and a `parse_*_from_reader` (plus
  `_with_limits` forms of both) since Phase 6.13, and `NPZ` had only the two
  path-based entry points. So Phase 6.13's copy-based untrusted-input contract
  had **no `NPZ` instance**, and a caller holding bytes — an `io.BytesIO`, an
  HTTP response, a blob from a dataset loader — had to write a temporary file
  first. All four entry points share one parse body with the path form, so
  limit enforcement and `NPY` interpretation cannot drift between them.

  `parse_npz_from_reader` takes `Read` alone rather than `Read + Seek`,
  matching `parse_pth_from_reader` and `parse_gguf_from_reader`: the stream is
  read into one bounded owned buffer and the container seeks happen over that,
  so a pipe or an HTTP body works with no seekable adapter.
  (`inspect_npz_from_reader` still requires `Seek`, because it deliberately
  never buffers the archive.) `tests/parse_owned_path.rs` — the suite that
  exists to pin exactly this contract, and that covered three formats because
  the fourth had nothing to pin — now covers all four.

- **A dequantised-size estimate for every format, and one trait to read it**
  (Phase 7.6, item 2). `GgufInspectInfo`, `PthInspectInfo` and
  `NpzInspectInfo` each gain `dequantized_size` and `output_dtype`, joining
  safetensors' `InspectInfo`, and all four now implement the new sealed
  **`InspectSummary`** trait (`tensor_count` / `current_size` /
  `dequantized_size` / `output_dtype`, plus a provided `expansion`).
  `InspectInfo` also gains `tensor_count`, which previously could only be
  obtained by summing the role counters, a sum that silently under-counted
  because `TensorRole::QuantState` has no counter of its own.

  **`GGUF` is why this matters.** It is the one format whose in-memory size
  cannot be inferred from its on-disk size, because the expansion ratio is
  *per kernel*: a `Q2_K` tensor and a `Q6_K` tensor of equal element count
  occupy different bytes on disk and the same bytes after dequantisation. The
  Phase 6.8 inspect-before-parse gate had nothing to check for a `GGUF`, on the
  format where the check matters most. The estimate is asserted to equal the
  exact tensor-data byte count `remember` then writes, at all three widths
  (`gguf_dequantized_size_predicts_what_remember_writes`). On `.pth` and `NPZ`
  nothing is quantised, so the figure equals `total_bytes` at every width and
  the dtype is recorded but vacuous.

- **`inspect_*_with_options` on every entry point** - `inspect_npz_with_options`,
  `inspect_npz_from_reader_with_options`,
  `inspect_gguf_from_reader_with_options`,
  `inspect_pth_from_reader_with_options`, plus `inspect_with_options` on
  `ParsedGguf`, `ParsedPth`, `GgufFrontMatter` and `PthFrontMatter`.

- **`amn inspect --to bf16|f32|f16`** - the CLI could not ask for the width it
  intended, so its size estimate was pinned to the `BF16` assumption. That is
  the identical bug v0.7.4 fixed for `remember`'s summary line and did not
  carry across. On a `Q6_K` `SmolLM2-135M` the same file now reports
  `Dequantized: 257 MB (BF16)` or `513 MB (F32)`, next to the unchanged
  `Total size: 130 MB`.

- **`ParsedGguf::remember` / `remember_with_options` / `remember_to_bytes` /
  `remember_to_bytes_with_options`** (Phase 7.6, item 1) — whole-model
  dequantise-and-write for a `GGUF` input, mirroring `ParsedModel`'s spelling
  method for method. Until now the only whole-model `GGUF` `remember` in the
  crate lived **inside the CLI**: 121 lines in `run_remember_gguf` that
  transcribed what `convert`'s reader already did. The transcription was
  sequential, took no thread budget (so `amn remember model.gguf --threads 8`
  was accepted and silently ignored), and could not be called from the library
  at all, which would have made a Python binding the third copy of the same
  loop. Both paths now run one `hub_from_gguf`.

  **Measured** (medians of 5, `--release`, `target-cpu=native`, output
  `SHA-256`-identical before and after): `SmolLM2-135M-Q6_K` 241 → 194 ms
  (**1.24×**), `Qwen2.5-1.5B-IQ2_M` 5854 → 2631 ms (**2.23×**). The gap widens
  with model size as the fixed parse/write cost amortises. Recorded as
  Experiment 16 in `docs/perf-experiments.md`.

- **`NpzTensor::new` and `PthTensor::new`** — both types are *returned* by
  `parse_npz` / `ParsedPth::tensors` **and** consumed by
  `npz_to_safetensors[_bytes]` / `pth_to_safetensors[_bytes]`, so marking them
  `#[non_exhaustive]` without a constructor would have silently removed a
  capability the public encode API offers. The constructors are the
  construction path that absorbs any field added later without breaking the
  callers that synthesise tensors.
- **Crate-level `# API evolution` section** (`src/lib.rs`) stating the rule
  once — return types are non-exhaustive, encode-side inputs are not — so the
  policy is discoverable rather than inferred from twenty attributes.

## [0.7.5] - 2026-08-18

### Added

- **Reader-generic full `.pth` front matter** (`src/parse/pth.rs`) —
  `parse_pth_front_matter_from_reader<R: Read + Seek>` /
  `parse_pth_front_matter_from_reader_with_limits`, plus the new
  **`PthFrontMatter`** type (`tensors: Vec<PthTensorInfo>`, `big_endian`).
  This is the full-detail counterpart to the summary-only
  `inspect_pth_from_reader` (0.6.x, `PthInspectInfo`): that type reports
  aggregate statistics for a cheap inspect-before-parse policy gate but
  carries no per-tensor list, which turned out to be insufficient for
  `hf-fm` v0.11.4's planned remote `.pth` inspect — that feature needs the
  same per-tensor name/shape/dtype table the mmap-backed
  `parse_pth(path).tensor_info()` exposes, without downloading the
  tensor-data files inside the archive. Direct `.pth` counterpart to
  0.7.1's `GgufFrontMatter`, surfaced by
  `docs/dogfooding-feedbacks/pth-front-matter-for-hf-fm-remote-inspect.md`.
  Implemented as a thin public wrapper over a new shared private core,
  `read_pth_meta` (composing the existing `read_pth_archive_for_inspect` +
  `interpret_pickle_to_meta` steps) — no new parsing logic, so
  `inspect_pth_from_reader` and the new front-matter functions remain
  substrate-equivalent by construction, and `inspect_pth_from_reader`'s
  output is unchanged byte-for-byte. `read_pth_archive_for_inspect` also
  gains a `limits: &ParseLimits` parameter (previously hard-coded to
  `ParseLimits::unbounded()` / `::default()`, which are equal, so this is
  not a behaviour change), so the new `_with_limits` entry point can
  actually bound the ZIP central-directory walk and the `data.pkl` size
  cap. Adds `PthFrontMatter::inspect()` to reduce a full front matter to
  the aggregate `PthInspectInfo` summary, via a reduction kept
  deliberately separate from the existing `build_pth_inspect_info`: unlike
  `GGUF`'s `ParsedGguf`, `ParsedPth` keeps only the lean `TensorMeta` on
  the struct and builds the heavier `PthTensorInfo` only on demand, so
  unifying the two reductions would have added an allocation cost to
  `ParsedPth::inspect()` it does not pay today. 5 new unit tests
  (`front_matter_from_reader_*`, mirroring the existing
  `inspect_from_reader_*` family) plus a
  `substrate_equivalence_front_matter_algzoo_fixtures` integration test
  (`tests/cross_validation_pth.rs`, exercising real populated-tensor
  pickle data across the `Cursor`/`File` substrates), a new
  `fuzz_pth_front_matter` target, extended `tests/no_panic.rs` coverage,
  and a `bench_pth_front_matter` CodSpeed benchmark (`benches/parsing.rs`,
  sharing `bench_pth_inspect`'s fixture) guarding the full-tensor-list path
  against a regression the existing summary-only benchmark wouldn't catch.

### Fixed

- **`ParsedPth::tensor_info()` (and the new front-matter path built on it)
  could report `byte_len: usize::MAX` for a tensor with zero elements**
  (`src/parse/pth.rs`). Found during a consistency pass on the front-matter
  work above. The per-tensor element-count fold used `try_fold` +
  `checked_mul`, which stops at the *first* overflowing shape dimension; a
  shape with a large-but-overflowing leading dimension and a **zero**
  trailing dimension (e.g. `[2^33, 2^33, 0]`) — mathematically zero
  elements, since any zero factor zeroes the product — short-circuited to
  `usize::MAX` before ever reaching the zero, instead of the correct `0`.
  `build_pth_inspect_info` (used by `inspect_pth_from_reader` and
  `ParsedPth::inspect()`) already folded every dimension with
  `saturating_mul` and got this right, so the two "equivalent" entry points
  could disagree by roughly `u64::MAX` on `total_bytes` for the same
  adversarial file — exactly the divergence the new
  `front_matter_inspect_matches_inspect_pth_from_reader` test is meant to
  rule out, on a shape that test didn't happen to exercise.
  `build_pth_tensor_info` now uses the same never-short-circuiting fold.
  Regression test: `front_matter_total_bytes_matches_inspect_on_overflowing_shape`.
- **`InspectInfo` size estimate now saturates instead of overflowing on an
  absurd header-declared shape** (`src/inspect.rs`). The `current_size` /
  `dequantized_size` accumulation used raw `+`/`*`, the lone exception to the
  crate's "checked/saturating on every header-derived value" invariant: a shape
  whose element count approaches `usize::MAX`, multiplied by the output width (up
  to ×4 at `F32`), could exceed `u64::MAX` — a debug-build panic and a silent
  release wrap of the very figure the inspect-before-parse gate reads. Because
  `SafetensorsHeader` has public fields and is not `#[non_exhaustive]`, a caller
  can construct such a header directly, bypassing the upstream `safetensors`
  validation that guards the file-parse fronts. The arithmetic is now
  `saturating_*`, so an overflowing estimate yields `u64::MAX` (fail-closed: the
  policy gate reads it as "too big"). Regression test:
  `oversized_shape_saturates_rather_than_overflowing`.

## [0.7.4] - 2026-08-15

### Added

- **The `remember` dequantisation output type is now caller-chosen**
  (`src/model.rs`, `src/remember/{fp8,gptq,awq,bnb}.rs`). `TargetDtype` gains
  `F32` and `F16`, and the four kernel families `remember` dispatches over
  become generic over the `OutputElement` trait v0.7.3 introduced for `GGUF`.
  The CLI spells it `amn remember <file> --to bf16|f32|f16`; the library takes
  it as the existing `TargetDtype` argument. Every `dequantize_*_to_bf16` entry
  point remains as an `#[inline]` `Bf16Out` wrapper, so **no existing caller
  changes**.

  With this, the output dtype is a caller-chosen parameter on *every* path the
  crate exposes. `convert --out-dtype` also widens from `GGUF`-input-only to
  every dequantising input, because the quantised-safetensors kernels that
  blocked it are the ones this release generalises.

- **`InspectOptions`**, a builder mirroring `RememberOptions` /
  `ConvertOptions`, plus `ParsedModel::inspect_with_options`. The
  dequantised-size estimate that feeds the inspect-before-parse policy gate is
  only meaningful against a specific output width, so it can now be asked for
  the width the caller actually intends. `InspectInfo::output_dtype` records
  which width a figure was computed for.

- **`F32` cross-validation for all four `remember` kernel families**, compared
  against the canonical libraries bit for bit with no tolerance: `FP8` against
  `PyTorch`, `GPTQ` against `GPTQModel`, `AWQ` against `AutoAWQ`, and `BnB`
  against `bitsandbytes`. Every fixture container gained a magic prefix and a
  version, and every generator gained a drift guard that refuses to overwrite a
  committed `BF16` golden that no longer reproduces.

- `remember_f32_whole_model` bench group in `benches/convert.rs`, added
  alongside `remember_bf16_whole_model` rather than renaming it, so the `BF16`
  CodSpeed history that serves as this phase's baseline survives.

- `docs/tutorials/choosing-an-output-dtype.md`, plus four `FAQ` entries on
  choosing a dtype, the size cost, the mixed-dtype passthrough policy, and
  whether the crate is still bit-exact at `F32`.

### Fixed

- **`BnB` `INT8` dequantisation was 1 `ULP` out on 26.9 % of elements at `F32`**
  (`src/remember/bnb.rs`). The kernel hoisted a per-row scale and computed
  `w × (SCB / 127)`, while `bitsandbytes`' `int8_vectorwise_dequant` computes
  `(w × SCB) × (1/127)`. Those are the same real number and round to the same
  `BF16`, which is why five releases of `BF16` cross-validation reported
  0/65536 mismatches and never saw it. Measured at full width against the
  canonical kernel: **17610/65536 (26.9 %)** of elements differed, every one of
  them by exactly 1 `ULP`. The constant was never the problem
  (`f32(7.874015718698502e-3)` and `f32(1.0/127.0)` are the same bits, now
  asserted at compile time); only the multiply order was. anamnesis now uses the
  canonical association. **No `BF16` output byte changes.** Costs 1.04× on that
  kernel (16.82 → 17.48 ms at 4096 × 11008), which is one extra packed multiply
  and is the honest price of exactness at every width.

- **Downstream callers of `dequantize_*_to_bf16` were ~2.9× slower than
  v0.7.3** (`src/remember/{fp8,gptq}.rs`). Turning those entry points into
  `#[inline]` generic wrappers meant an external crate instantiated the generic
  in *its own* crate, where `e4m3_to_f32_bits`, `e4m3_to_scaled_f32`,
  `f32_bits_to_bf16_bits` and `unpack_gptq` were private, non-`#[inline]` and
  therefore opaque: a function call per element. `remember` never suffered it,
  calling the generic from inside the library. Fixed by marking those four
  helpers `#[inline]`.

- **`InspectInfo`'s rendered size line claimed the wrong dtype**
  (`src/inspect.rs`). The estimate became dtype-aware while `Display` still
  printed a literal `(BF16)`, so asking for the `F32` figure rendered a doubled
  number under a `BF16` label. The line now reads the width off
  `output_dtype`, with a regression test.

- **`amn remember --to f32` reported the `BF16` size** (`src/cli.rs`). The
  summary line built its `InspectInfo` with `From<&SafetensorsHeader>`, which is
  hard-wired to the `BF16` default, so every width printed the same figure: a
  file holding 272 B of payload was announced as 144 B. The written file was
  always correct; only the number beside it was wrong. It now sizes the estimate
  at the requested width via `InspectOptions`.

### Changed

- `to_bf16_bytes` (the `bnb`-gated encode-side helper) now rounds to nearest
  even, matching the crate's `f32_bits_to_bf16_bits` convention its own doc
  comment already claimed; it previously truncated.

- The `remember` per-dtype determinism tests now use a fixture sized off
  `MIN_PARALLEL_BYTES`. The previous 32-byte fixture was far below the 4 `MiB`
  parallel threshold, so every thread-count assertion had been exercising the
  sequential path.

- **`scripts/verify-claims.{ps1,sh}` counted the tests it ran, and the count
  was always zero.** Both scripts invoked `cargo test -- --quiet` and then
  counted lines matching `^test .* ok$`, but `--quiet` prints one dot per test
  and never emits those lines, so every suite reported "PASS 0 tests". The
  pass/fail verdict was real (it keyed off the exit status), but a suite that
  compiled and ran *nothing* was indistinguishable from one that verified 22
  kernels. That is a poor property for the script the README points readers at
  to substantiate the correctness claims. Both now parse the `test result:`
  summary line and **treat a zero count as a failure**. The suite descriptions
  also said only `GGUF` was verified at `F32`; every dequantising family is,
  as of this release.

## [0.7.3] - 2026-08-12

### Added

- **The `GGUF` dequantisation output type is now caller-chosen**
  (`src/remember/output.rs`, `src/remember/gguf.rs`). A new sealed
  `OutputElement` trait with three implementations, `Bf16Out` (the unchanged
  default), `F32Out` and `F16Out`, replaces the hard-coded `BF16` narrowing that
  every kernel has performed since the crate's first commit. Two new entry
  points, `dequantize_gguf::<E>` and `dequantize_gguf_blocks::<E>`;
  `dequantize_gguf_to_bf16` and `dequantize_gguf_blocks_to_bf16` remain as
  `#[inline]` `Bf16Out` wrappers, so **no existing caller changes**.

  **Why it matters.** `BF16` keeps 8 significand bits, but a `Q8_0` value is an
  `f16` scale times an `int8` and needs about 18; `Q6_K` needs 24. Measured on
  `SmolLM2-135M-Q4_K_M`, only **3 to 20 %** of dequantised values are exactly
  `BF16`-representable. The usual defence, that quantisation error dwarfs the
  rounding, fails precisely where it matters most: `Q8_0`'s own quantisation
  step is the *same order* as `BF16`'s half-`ULP`, so the crate was adding
  rounding comparable to the error the format exists to avoid. `F32Out` adds no
  narrowing step of its own, so its output **is** the reference's `f32`.

  **All 24 kernel bodies are untouched.** Only their signatures thread the type
  parameter through; the bit manipulation, the formulas and every annotation are
  byte-identical, which the 22 existing cross-validation fixtures confirm by
  still passing unchanged. That economy is specific to `GGUF`, where all 24
  kernels funnel through one pass-2 writer, and is why the `remember` path's
  four families are a separate phase.

  **The streaming sink's block length is now dtype-dependent**: `QK × E::BYTES`,
  so 64 B / 512 B at `BF16` and `F16` but 128 B / 1024 B at `F32`. A sink that
  hard-codes `chunks_exact(2)` is correct only for the 2-byte types. The public
  docs carry the table and a test asserts the observed length per output type,
  so the assumption fails loudly rather than silently misreading `F32` output as
  twice as many `BF16` values.

  `F16` follows **plain IEEE semantics** via `half::f16::from_f32`: overflow to
  infinity, flush to zero below roughly `2⁻²⁴`, round-to-nearest-even between.
  Deliberately not saturating, which would fabricate a value no reference
  produces and put `F16` cross-validation permanently at odds with `NumPy` and
  `PyTorch`. This range is reachable in real data: `MXFP4`'s `E8M0` scale spans
  `2⁻¹²⁸` to `2¹²⁷`. Note that `F16` is **not** uniformly the better 2-byte
  choice, since it buys 3 significand bits and pays a far narrower exponent
  range.

  The trait is **sealed**. Its contract is a byte-level invariant the
  cross-validation depends on, and an outside implementation could break it
  while every test stayed green. Sealing is also the reversible direction:
  un-sealing later is not a breaking change, sealing later would be.

- **`ParsedGguf::dequantize_tensor_as::<E>`** (`src/parse/gguf.rs`), the
  per-tensor counterpart of the whole-file option. `dequantize_tensor` remains
  as the `BF16` spelling. Without it, a caller wanting `F32` for a single tensor
  would have had to re-implement the offset, byte-length and element-count
  validation that method exists to encapsulate, which would have left the
  per-tensor path worse off than the whole-file one.

- **`convert` and the CLI can now choose that output dtype**
  (`src/convert.rs`, `src/cli.rs`, `docs/FAQ.md`).
  `ConvertOptions::output_dtype` with a `with_output_dtype` builder matching
  `with_threads`, and `amn convert --out-dtype bf16|f32|f16`. The flag is named
  `--out-dtype` rather than reusing `--to` because on `convert` `--to` already
  selects the output *format*; on `remember`, `--to` already selects a dtype, so
  that subcommand needs no new flag when Phase 7.4 lands.

  The option takes a `Dtype` rather than introducing a narrower enum, because
  that is already the type a hub tensor carries, so no third dtype vocabulary
  enters the crate. Values outside `{BF16, F32, F16}` are rejected at the
  boundary with a message listing what is accepted.

  **Passthrough policy, now stated rather than assumed.** `remember` and
  `convert` have always emitted mixed-dtype files, with dequantised tensors at
  `BF16` and passthrough tensors keeping their source dtype. `--out-dtype`
  widens **dequantised tensors only**: an `F16` norm stays an `F16` norm and an
  `F32` tensor stays byte-identical. Widening a passthrough tensor would invent
  precision that was never in the file while doubling its size, and a caller who
  wants a uniform-dtype file wants a cast pass, which is a different operation.
  Asserted per tensor in `convert_honours_every_output_dtype_end_to_end`, not
  merely documented.

  **Scope, made explicit in the error rather than silently.** Only the `GGUF`
  reader honours a non-`BF16` request in v0.7.3. A quantised safetensors input
  returns `Unsupported` naming v0.7.4 and the reason (those four families narrow
  inside their hot loops), so the caller never gets a file whose dtype differs
  from the request. `NPZ` and `.pth` dequantise nothing, so the option is
  vacuous there and is accepted rather than refused, since erroring on
  `--out-dtype f32` for an already-`F32` `NPZ` would be hostile.

  Determinism is re-established **per dtype**: output is byte-identical across
  `{1, 2, 4, 8}` threads at each of the three widths, rather than assumed to
  carry over from the `BF16` suite.

  **A derived output filename now names the dtype it actually holds**
  (`derive_output_path_for_dtype`, new; `derive_output_path` kept unchanged as
  the `BF16` spelling, so no caller breaks). `ConvertTarget::suffix()` returns
  `bf16` for the safetensors target, which was correct while `BF16` was the only
  possible answer; without this, `amn convert --out-dtype f32` would have
  written `F32` tensors into `model-bf16.safetensors`. That is worse than an
  unhelpful name because it is an actively wrong one. The `gguf` and `bnb-nf4`
  targets keep their own suffix, since there it names a container or an encoding
  rather than an element type.

- **All 22 `GGUF` kernels are now cross-validated at `F32`, bit-exactly**
  (`tests/cross_validation_gguf.rs`,
  `tests/fixtures/gguf_reference/generate_gguf.py`). Every cross-validation
  before this rounded the `gguf-py` reference to `BF16` before comparing, which
  discarded 16 mantissa bits: **no kernel's `f32` had ever been checked at full
  width.** A kernel could have associated its arithmetic differently from the
  reference (`(d·sc)·q` against `d·(sc·q)`, or a contraction on a `d·q - dmin·m`
  line) and every fixture would still have passed.

  **The result: 22 of 22 pass, exactly, with no tolerance.** This was the step
  the ROADMAP told us to budget for failing, so the null result is worth
  stating plainly rather than passing over. All 22 production kernels already
  associate their arithmetic identically to `gguf-py`. Nothing needed fixing;
  what changed is that it is now *verified* rather than assumed.

  Exact bit equality is the right bar, not an epsilon: anamnesis computes in
  `f32` and so does `gguf-py`, so identical operations in identical order must
  produce identical bits, and any difference is a real divergence rather than
  accumulated noise.

  **The comparison has teeth, and that was demonstrated rather than asserted.**
  Across the 22 fixtures, **76.5 %** of the 1 441 792 reference values (1 102 549
  of them) carry mantissa bits `BF16` cannot represent, so the new assertion
  reads information the old one discarded. Flipping a single mantissa bit in one
  golden makes exactly one of 65 536 elements fail at 1 `ULP` while the `BF16`
  comparison stays green, which is the hidden-16-bits problem shown directly.

  Fixtures move to a **versioned container** (magic `AMNG`, version 2) carrying
  the `BF16` and `F32` goldens side by side; the previous layout had neither
  magic nor version and so could not be extended unambiguously. The Rust reader
  rejects any other version by name instead of misreading offsets.

  Both goldens come from `gguf-py`. The `BF16` one is **not** derived by
  rounding the `F32` one in Rust, which would compare anamnesis's output against
  a golden produced by anamnesis's own rounding, i.e. the circular-fixture class
  that shipped three green bugs in v0.6.4.

  `generate_gguf.py --upgrade` rebuilds the fixtures **from their own raw
  quantised bytes**, so the `F32` goldens are reproducible from a clean checkout
  with `pip install gguf numpy` and no multi-`GiB` model download. Each upgrade
  re-derives the `BF16` golden and refuses to proceed unless it reproduces the
  committed one byte for byte, so a differing `gguf` version fails loudly
  instead of silently rebasing the reference. All 22 verified unchanged.

  The fixtures grew ~6 MB, which costs the published crate nothing: `tests/` is
  now excluded, and the `.crate` stayed at **0.60 MiB** across this change.

- **ThreadSanitizer is now gating, with an instrumented `std`**
  (`.github/workflows/tsan.yml`, `src/bin/tsan_harness.rs`). v0.7.2 shipped
  with the `CONVENTIONS.md` race-detector rule still open — the job existed but
  could not be made green, and its CHANGELOG entry says so. It can now: the
  parallel dispatch reports **zero races** across `{1, 2, 4, 8}` threads plus
  the hardware-resolved default, and every budget is byte-identical.

  Getting there needed three constraints to hold at once, each learned from a
  red build and each individually insufficient:

  1. **A `--bin` target rather than `cargo test`.** `-Zbuild-std` is unusable
     with `cargo test`
     ([cargo#13146](https://github.com/rust-lang/cargo/issues/13146), open since
     2023) because `cargo test` also builds dev-dependencies, several of which
     are crates `std` itself depends on. `cargo run --bin` builds none.
  2. **`-Cunsafe-allow-abi-mismatch=sanitizer`**, because `compiler_builtins`
     is not instrumentable and rustc otherwise refuses the link. It is
     memcpy/memset intrinsics with no synchronisation, so mixing it in is inert
     for a race detector.
  3. **`panic_abort` in the build-std set**, because `[profile.release]` sets
     `panic = "abort"`; without it rustc falls back to the sysroot and drags its
     `core` in ([cargo#15347](https://github.com/rust-lang/cargo/issues/15347)).

  This matters beyond tidiness: an instrumented `std` is a **precondition**, not
  a refinement. Without one, `std::thread::scope`'s own `Arc` teardown
  false-positives ([rust#101206](https://github.com/rust-lang/rust/issues/101206))
  — and scoped threads are the mechanism under test, so the detector fires on
  exactly what it is meant to check.

  `.github/tsan-suppressions.txt` is retained, wired to nothing, as the record
  of the abandoned uninstrumented-`std` approach.

  **A claim that did not survive checking**, recorded because it nearly went
  upstream: an intermediate diagnosis held that `CARGO_TARGET_<TRIPLE>_RUSTFLAGS`
  silently fails to reach `-Zbuild-std` units, and it was on its way to a Cargo
  bug report. It came from a comparison in which two variables changed at once —
  the flag source *and* a newly added `-Cunsafe-allow-abi-mismatch`. Isolating
  them in a single-variable run against the green configuration showed **both
  flag sources work**; the `core` fingerprint change that looked like proof was
  just the ABI flag altering the fingerprint. There is no Cargo bug here.

- **The crate is now on Rust Edition 2024** (`Cargo.toml`, plus twelve
  hand-written sites). `rust-version = "1.88"` already cleared the edition's
  1.85 floor, so the MSRV is unchanged and no consumer needs a newer toolchain.

  Only one site changed meaning rather than spelling. In
  [`src/parallel.rs`](src/parallel.rs), `handle.join()` was a `match` scrutinee
  whose temporary may carry a custom destructor (the `Err` arm holds a panic
  payload as a `Box<dyn Any>`), and Edition 2024 drops such a temporary
  **earlier** than 2021 relative to the iterator in the matched arm
  (`tail_expr_drop_order`). The join result is now bound to a local first, which
  pins the drop point identically under both editions. The reordering is inert
  here in any case, and that was checked rather than assumed: the lint's own
  caveat is to inspect the `impl Drop`s for side effects like releasing a lock
  or sending a message, and there are none on either path.

  The rest is spelling. Edition 2024 stabilises let-chains, so eight nested
  `if let` blocks that previously *could not* be collapsed now can and must
  (`clippy::collapsible_if`), across `src/parse/{pth, safetensors, ollama}.rs`
  and `tests/panic_profile.rs`. And `unsafe_op_in_unsafe_fn` is an error rather
  than a lint, so the AVX2 experiment in `tests/bench_pass2_adhoc.rs` states its
  bounds argument per operation instead of inheriting it from the `unsafe fn`
  signature, which is a strict improvement in what the `// SAFETY:` comments
  actually claim.

  Two knock-on effects worth recording. Cargo's resolver moves to **v3**, which
  is MSRV-aware: `cargo update` now reports "latest Rust 1.88 compatible
  versions" and will no longer propose a dependency that would silently raise
  the floor. It selected nothing new, so the lockfile is byte-identical across
  the flip. Separately, rustfmt's default `style_edition` follows the language
  edition; that reformat is deliberately **not** in this change, so the diff
  here is the migration and nothing else.

### Changed

- **Repository metadata now points at the `mi-for-the-rust-of-us` organization**
  (`Cargo.toml`, `README.md`, `docs/FAQ.md`, `docs/tutorials/`, `ROADMAP.md`,
  `docs/issues/README.md`).
  The repo transferred from `PCfVW/anamnesis` to the org alongside `candle-mi`,
  `hf-fetch-model` and `hypomnesis`, but anamnesis was the one crate whose
  `repository` / `homepage` keys were never updated — so **every published
  version through `0.7.2` links crates.io and docs.rs back to the old personal
  namespace**. GitHub redirects, so nothing was broken, but the eco-system's
  four crates did not agree on where they live. Corrected here alongside the CI
  badge, the `Used by` links, the FAQ issue link, the two tutorials'
  sibling-tool links, and the issues archive's pointer at its `hf-fetch-model`
  original. That last one is corrected because it is a *live* pointer rather
  than a record: the surrounding text instructs the reader to open that file and
  to prefer it over this copy, so it has to resolve to where the file lives now.

  The remaining `PCfVW` references are all correct and are deliberately left
  alone, so that a future `grep` does not read them as a missed spot. Historical
  `CHANGELOG` and per-release roadmap entries, and the archived reply drafts
  under `docs/issues/`, record what was true when written. `CONVENTIONS.md`
  links to `PCfVW/Amphigraphic-Strict`, which is a genuinely personal repository
  and did not transfer. The `LICENSE-*` files name the copyright holder, not a
  repository.

  **This fix only reaches crates.io on the next release** — registry metadata is
  frozen per published version and cannot be amended in place.

- **The published crate no longer carries the test corpus, and is 8× smaller**
  (`Cargo.toml` `exclude`). `tests/` joins `fuzz/` in the exclude list. Measured
  with `cargo publish --dry-run`: the payload drops to 2.2 MiB of almost
  entirely text and gzips to **628 687 bytes (0.60 MiB)**, against the
  **4.8 MiB** crates.io served for `0.7.2`. `tests/fixtures` alone was 6.33 MiB
  of binary goldens, which is 71 % of everything tracked and reachable from
  nobody's build.

  Nothing else was dropped: `README.md`, `CHANGELOG.md`, `ROADMAP.md`,
  `CONVENTIONS.md`, all of `docs/`, and `benches/` still ship. `benches/` stays
  deliberately, because the `[[bench]]` targets are declared explicitly in
  `Cargo.toml` and would dangle without their sources.

  **The tests are not gone.** Every tag now gets a GitHub Release, and GitHub's
  per-tag source tarball carries `tests/` verbatim;
  `scripts/verify-claims.ps1` / `.sh` run the cross-validation suites from
  there. See *Verifying the correctness claims* in the `README`.

  Beyond the download, this removes a constraint that had started pointing the
  wrong way. The 10 MiB registry cap was beginning to bound how wide the
  cross-validation goldens could be, and a test corpus should be sized by the
  correctness claim it has to support, not by a packaging limit. The release
  gate is unaffected: `cargo package`'s verification build never compiled test
  code, and `publish.yml` runs `cargo test --all-features` before `cargo
  publish`.

- **Dependencies refreshed** (`Cargo.lock`). 52 packages moved to their latest
  semver-compatible versions. Every change is transitive: no direct dependency
  altered its requirement string, so this is a lockfile refresh rather than an
  API-surface change.

  Two versions are pinned on purpose and were verified to have stayed put.
  `criterion` (aliased to `codspeed-criterion-compat`) must remain **major 5** to
  match the `cargo-codspeed` binary the CI installs, because the *walltime*
  instrument fails to collect results across a major-version mismatch. And `zip`
  stays at **v2** with v8 available: it is the differential oracle the vendored
  `src/parse/zip.rs` reader is checked against, so moving it across six majors
  means porting the `ZipWriter` fixtures, the `differential_*` tests and the
  `fuzz_zip` harness, with a real risk of quietly weakening the cross-check it
  exists to provide. It is dev-only, so the pin ships to nobody. Both rationales
  now sit in `Cargo.toml` beside the pins, since `cargo update` will keep
  reporting `zip` v8 as available and the next reader deserves to know that is
  expected rather than an oversight.

## [0.7.2] - 2026-08-07

### Added

- **`GGUF`-input conversions are now multi-threaded** (`src/convert.rs`,
  `src/parallel.rs`). `convert()`'s `GGUF` reader dequantised every tensor on a
  single thread — it was the one input path Phase 7 (v0.7.0) left behind, and it
  did not even receive the caller's thread budget. It now runs through the same
  per-tensor dispatch the safetensors path uses, honouring
  `ConvertOptions::with_threads` and preserving the determinism contract:
  **output is byte-identical at any thread count**, verified across
  `{1, 2, 4, 8, 16}` and the hardware-resolved default for the `safetensors`,
  `gguf` and `bnb-nf4` targets. Both branches of the reader benefit — the
  quantised branch because dequantisation is the compute, the scalar branch
  because its `into_owned()` copy is bandwidth-bound — so all-`F32` `GGUF`
  inputs gain too. The ROADMAP had recorded `ParsedGguf: Sync` as the blocker
  for this work; it turned out to be already satisfied, so no `unsafe` and no
  marker impl were needed (and `tests/parallel_contract.rs` now asserts it stays
  that way).

  **Measured** (5950X, `target-cpu=native`, best-of-5, real models;
  `docs/perf-experiments.md` Experiment 12): the reader stage itself is
  **1.90×** faster at the default 4-thread budget on `SmolLM2-135M-Q4_K_M` and
  **1.95×** on `TinyLlama-1.1B-Q5_0`, plateauing ~2.1× at 16 threads.
  End-to-end `convert()` gains **1.19–1.36×**, and the gap is Amdahl, not a
  defect: writing the multi-`GiB` output file is ~60 % of the wall clock and is
  inherently sequential. Both figures are reported because they answer different
  questions — do not quote the reader number for `convert()`.

- **A shared parallel dispatch, `src/parallel.rs`.** v0.7.0's inline
  `thread::scope` block split tensors into contiguous **equal-count** chunks,
  which does not survive contact with `GGUF`: tensor sizes there are heavily
  skewed, so an equal-count split leaves one worker holding most of the bytes.
  The new `map_indexed` helper claims work through an atomic cursor instead, so
  the pool self-balances, and both the safetensors and `GGUF` paths now go
  through one code path with one `// PARALLEL:` annotation. Results are
  index-tagged and sorted, so ordering is independent of steal order as well as
  of thread count; **error selection is deterministic too** — the lowest-indexed
  failure is reported at any budget, matching the sequential loop exactly.

- **`MIN_PARALLEL_BYTES`, a measured size threshold** (4 `MiB` of input).
  `CONVENTIONS.md` requires a named floor below which the sequential path runs;
  v0.7.0 only checked the tensor *count*, so an eight-tensor toy model paid four
  thread spawns to dequantise a few `KiB`. Calibrated, not guessed: a 4-worker
  scoped pool costs **236 µs** to spawn and join on Windows, and the reader
  sustains ~1.05 `GB/s` on one thread, putting break-even at ~0.5 `MiB`; 4 `MiB`
  takes that with an ~8× margin.

- **`--threads N` on `amn remember` and `amn convert`** (`src/cli.rs`). The
  thread budget has been a library knob since v0.7.0 but was unreachable from the
  CLI, which was pinned to the `min(cores, 4)` default with no way to opt out or
  scale up. Output is byte-identical whatever is passed.

- **`ParsedModel::remember_with_progress_and_options`** (`src/model.rs`) — the
  two-knob form of `remember_with_progress`. Callers previously had to choose
  between a progress bar and a thread budget, because the progress variant pinned
  the budget to its default. `on_tensor` still fires only on the calling thread.

- **First test coverage for quantised-`GGUF` conversion** (`src/convert.rs`).
  Every existing `GGUF`-input convert test builds its fixture with `write_gguf`,
  which rejects quantised dtypes (quantised emit is Phase 8.5) — so the
  `dequantize_gguf_to_bf16` branch of the reader was exercised only against
  gitignored multi-`GiB` local models and **never in CI**. A hand-rolled
  quantised-`GGUF` writer closes that gap: a 17-tensor fixture (`Q8_0`, `Q4_K`,
  `F32`, deliberately skewed sizes) whose every tensor is checked against the
  dequant kernel called directly.

- **`benches/convert.rs`, a CodSpeed target for the threaded paths.** `dequant.rs`
  times one kernel and `parsing.rs` times a header parse; neither spawns a thread,
  so Phase 7's headline speedup had no continuous regression guard. Each group is
  measured at **1 and 4 threads**, because the ratio is the signal — a dispatch
  that silently serialises collapses it while leaving every absolute time
  plausible.

- **A ThreadSanitizer job — manual-only, and honest about why**
  (`.github/workflows/tsan.yml`). `CONVENTIONS.md` has asked for a race-detector
  run since v0.7.0, and Windows/MSVC has no TSan, so CI is the only place the
  rule could be satisfied. The job builds, runs the whole unit-test suite, and
  reports — but it is `workflow_dispatch`-only rather than gating, because it
  cannot currently be made green for upstream reasons: `-Zbuild-std` is unusable
  with `cargo test` ([cargo#13146](https://github.com/rust-lang/cargo/issues/13146),
  "duplicate lang item in crate `core`"), and without an instrumented `std`,
  libtest itself reports false races
  ([rust#39608](https://github.com/rust-lang/rust/issues/39608)). The libtest
  result channel was suppressible; the next report landed in
  `std::sys::thread::unix::Thread::new` — the same machinery
  `std::thread::scope` uses — where a suppression would have blinded the
  detector to our own thread creation, so the chain was stopped there rather
  than papered over. The **`CONVENTIONS.md` rule therefore stays open**, and
  re-attempting it is carried into Phase 7.3. What the job still verifies when
  run by hand is documented in the workflow itself.

- **A `gguf-py` performance baseline**
  (`tests/fixtures/gguf_reference/generate_gguf_dequant_timings.py`, committed
  sidecar JSONs). Ahead of the Phase 8 Python bindings, whole-model `GGUF`
  dequantisation is now measured against `gguf` 0.18.0 — the only
  general-purpose `GGUF` dequantiser on PyPI, `NumPy`-vectorised, and the same
  library the two are cross-validated against each other on.
  **17.3–27.9× single-threaded, 33.8–52.9× at the default 4-thread budget.**

  **Not a like-for-like output, and the numbers should not be quoted without
  this:** `gguf-py` returns `float32`, anamnesis returns `BF16` — half the bytes
  and the narrower type. What is verified is that anamnesis's `BF16` is
  bit-identical to `gguf-py`'s `float32` **correctly rounded to `BF16`** (0 ULP,
  all 22 kernels), so the two agree on the values and differ only in delivered
  width; both do their block arithmetic in `f32` internally. Since that width
  difference is itself worth ~2× of memory traffic on a bandwidth-bound
  workload, halving `gguf-py`'s time as a generous correction still leaves
  ~8.7–14× and ~17–26× respectively.

### Changed

- **BREAKING — `RememberOptions::with_threads` is now a builder method**
  (`src/model.rs`). It was an associated *constructor* while the equivalent
  `ConvertOptions::with_threads` was a `self`-taking *builder*: two spellings of
  one concept, about to be frozen into the Python bindings at v0.8.0. Migration
  is one line:

  ```rust
  // before
  RememberOptions::with_threads(4)
  // after
  RememberOptions::new().with_threads(4)
  ```

  `RememberOptions::new()` is new (a `const fn`); `RememberOptions::default()`
  is unchanged and equivalent.

### Fixed

- **`ConvertOptions::threads` documented the wrong scope** (`src/convert.rs`).
  It claimed the budget "applies only to the safetensors input path" and that
  "other readers are single-threaded" — true when written, false as of this
  release. It now names both applicable paths and explains why the `NPZ` and
  `.pth` readers stay sequential by nature rather than by omission: neither
  format is block-quantised, so there is no dequantisation to spread.

- **Determinism assertions no longer dump megabytes on failure**
  (`src/convert.rs`). A plain `assert_eq!` on two multi-`MiB` byte vectors
  renders both in full; a deliberately-injected determinism regression produced a
  46 `MB` test log. The comparisons now report the length and the first differing
  byte offset.

## [0.7.1] - 2026-08-05

### Added

- **Reader-generic full `GGUF` front matter** (`src/parse/gguf.rs`) —
  `parse_gguf_front_matter_from_reader<R: Read + Seek>` /
  `parse_gguf_front_matter_from_reader_with_limits`, plus the newly-public
  **`GgufFrontMatter`** type (`version`, `alignment`, the complete `metadata`
  table, and the complete `tensor_infos` list). This is the full-detail
  counterpart to the summary-only `inspect_gguf_from_reader` (0.4.5,
  `GgufInspectInfo`): that type reports aggregate statistics for a cheap
  inspect-before-parse policy gate but carries no per-tensor list, which
  turned out to be insufficient for `hf-fm` v0.11.2's remote `GGUF` inspect
  — that feature needs the same per-tensor name/shape/dtype/offset table the
  mmap-backed `parse_gguf(path).tensor_info()` exposes, without downloading
  the tensor-data segment. Implemented as a thin public wrapper over the
  existing `read_gguf_structure` core (used internally by `parse_gguf` and
  `inspect_gguf_from_reader` since 0.4.5) — no new parsing logic, so all
  three entry points remain substrate-equivalent by construction. Adds
  `GgufFrontMatter::inspect()` to cheaply reduce a full front matter to the
  aggregate `GgufInspectInfo` summary. 5 new unit tests
  (`front_matter_from_reader_*`, mirroring the existing
  `inspect_from_reader_*` family) plus a new `fuzz_gguf_front_matter` target
  and a `bench_gguf_front_matter` CodSpeed benchmark
  (`benches/parsing.rs`, sharing `bench_gguf_inspect`'s synthetic fixture)
  guarding the full-tensor-list path against a regression the existing
  summary-only benchmark wouldn't catch.

### Fixed

- **Two inaccurate `GGUF` doc comments** (`src/parse/gguf.rs`) — documentation
  only, no behaviour change. `GgufTensorInfo` described `data_offset` as an
  offset "inside the memory-mapped file", but the type has been reachable
  through non-mmap paths since `parse_gguf_bytes` / `parse_gguf_from_reader`
  and is now also returned by `parse_gguf_front_matter_from_reader` over a
  caller-supplied reader; it now reads as an absolute offset from the start
  of the source, whatever that source is. Separately, the internal `align_up`
  helper claimed the parser substitutes the default alignment when
  `general.alignment` is "absent or zero" — in fact a present-but-zero
  `general.alignment` is rejected outright (`GGUF: general.alignment is
  zero`); only an absent key falls back to the 32-byte default.
- **Panic-freedom coverage extended to the new entry points**
  (`tests/no_panic.rs`). The `catch_unwind` battery drives every re-exported
  parse/inspect entry point over a corpus of adversarial inputs, under both
  `ParseLimits::default()` and a hostile tight budget;
  `parse_gguf_front_matter_from_reader` and its `_with_limits` variant now
  join it, so the "no public parse/inspect entry point panics" invariant is
  evidenced across the whole public surface again rather than all-but-two of
  it.
- **Documentation brought back in line with what shipped.** The crate-root
  docs advertised "four reader-generic entry points" and described the
  reader-generic `GGUF` path as returning "just the `GgufInspectInfo`
  summary" — the precise limitation this release removes; both are corrected
  and the runnable example now shows the full-detail path.
  `docs/validation.md`'s per-format reader table lists GGUF's second entry
  point. Separately, three documents (`README.md`, `docs/FAQ.md` ×2) still
  billed `0.7.0` as a runtime-dispatched **SIMD** pass; it shipped as
  multi-threading, because explicit SIMD was prototyped, measured at 1.02×,
  and rejected (`docs/perf-experiments.md`, Experiments 10–11). The `README`
  dependency snippet still pinned `version = "0.6"`, `ROADMAP.md`'s status
  header still read "v0.6.8 released", and `fuzz/README.md`'s target table
  omitted `fuzz_gguf_front_matter`.

## [0.7.0] - 2026-07-26

### Added

- **Multi-threaded per-tensor dequantisation** (Phase 7) behind a new
  **`parallel`** Cargo feature (default-**on**, zero third-party dependency —
  `std::thread::scope` over disjoint per-tensor chunks). `ParsedModel::remember`
  / `remember_to_bytes` now dequantise the quantised tensors across a small pool
  of worker threads. The kernels are embarrassingly parallel per tensor; the
  measured win is **~3–4×** on a real fixture (`docs/perf-experiments.md`
  Experiment 11), bounded by DRAM bandwidth rather than core count.
  - New **`RememberOptions`** (re-exported at the crate root) plus
    `ParsedModel::remember_with_options` / `remember_to_bytes_with_options`
    carry a caller-owned, hardware-bounded thread budget: `None` →
    `min(available_parallelism, 4)` (the measured scaling knee, leaving the
    host's other cores free), `Some(n)` → `n.max(1)`. With the `parallel`
    feature off the path is always sequential. The budget is derived only from
    hardware and the caller — never from any file-declared count.
  - `ConvertOptions` gains a matching `threads` field / `with_threads` builder;
    the safetensors input path (which reuses the model dequant) is parallelised
    for free.
  - **Determinism is guaranteed**: output bytes are byte-identical for any thread
    count (results reassembled in original header order), pinned by a new
    determinism test across `n ∈ {1, 2, 4}`. The existing bit-exact
    cross-validation suite versus PyTorch is unchanged. Existing `remember` /
    `remember_to_bytes` / `remember_with_progress` signatures are unchanged.

## [0.6.9] - 2026-07-22

### Added

- **Convert-matrix completion via a `BF16` hub** (Phase 6.14). `amn convert` /
  the new library `anamnesis::convert` route every `(input × target)` pair through
  one in-memory `BF16` hub (`src/convert.rs`, publicly re-exported), so every
  input reaches every current target (`safetensors`/`bf16`, `gguf`, `bnb-nf4`):
  the `bnb-nf4` target now accepts NPZ / `.pth` / GGUF / quantized-safetensors
  sources, `gguf → gguf` dequantizes in place (preserving the source metadata KV),
  and a quantized input **auto-chains** through `BF16` (the "dequantize first"
  two-hop is gone). Scalar dtypes are preserved end to end. New `--gguf-metadata
  <file.json>` / `--gguf-kv key=value` pass a typed `GGUF` KV table (inference +
  an explicit `{"type","value"}` escape hatch), merged as source → file → flag.
  `ParsedGguf::metadata()` is used to inherit source KV. No new kernels; quantized
  GGUF target columns still need Phase 8.5.

- **`docs/tutorials/convert-between-formats.md`** — a walkthrough of `amn convert`
  with real captured output (FP8 → `bnb-nf4` / `gguf` auto-chain, GGUF-KV stamping,
  `gguf → gguf` KV inheritance), linked from the FAQ convert answer.

- **`docs/validation.md`** and **`docs/cli-reference.md`** — the validation
  evidence (cross-validation tables, per-kernel speeds, conversion benchmarks,
  peak-heap assertions, robustness hardening timeline) and the complete CLI
  reference (subcommands, flags, the convert matrix, output-path rules, the
  `ollama:` URL scheme), relocated out of the README.

### Changed

- **Lower `convert` peak heap** (Phase 6.14 copy-elimination pass, no output
  change). Measured with a `dhat` harness ([`tests/bench_convert_adhoc.rs`](tests/bench_convert_adhoc.rs)):
  a `bnb-nf4` conversion peaks **−39 %** (the encoder no longer re-copies an
  already-`BF16` hub), `NPZ → *` peaks **−49.6 %** (the reader moves tensors out
  of the parse map instead of cloning), and `gguf → gguf` allocates **−20 %**
  cumulative (the inherited source KV — a multi-thousand-entry tokenizer array —
  is borrowed through, not deep-cloned, when there is no caller KV to merge).
  Plus an `O(passthrough × N) → O(N)` dtype lookup in the hub for many-tensor
  models. Full numbers in [`docs/perf-experiments.md`](docs/perf-experiments.md)
  (Experiment 9).

- **README restructured** around an audience-oriented "first window" — a *New to
  anamnesis?* routing block, *Try it*, *Library quick start*, a compact *Formats
  & quantization support* matrix, and a *What's next* roadmap teaser — with the
  long validation and CLI detail moved to the new docs above. The pre-1.0 status
  is now an understated line rather than a warning banner. Docs only — no code or
  API change.

## [0.6.8] - 2026-06-25

### Added

- **`NumPy` / `BF16` data-ownership contract** (Phase 6.13 Step 4), the design
  the Phase 8 PyO3 bindings implement — recorded in `docs/python-interop.md`
  *before* a `pip` API freezes it. **Ownership:** owned-copy by default — a
  returned array always owns its bytes (GGUF / `.pth` via `Cow::into_owned`, npz
  and `remember_to_bytes` already owned), so no array aliases a `Backing` the
  owning `Parsed*` can drop (the pure-Python use-after-free is structurally
  impossible); zero-copy via `PyCapsule` is a documented future opt-in.
  **`BF16`:** return `ml_dtypes.bfloat16` when that optional dep is present, else
  raw `bytes` + a `"bfloat16"` dtype string — never a silent upcast. Pinned by
  `tests/python_ownership_contract.rs` (owned extraction outlives a dropped
  `Parsed*`) and FFI-boundary doc-notes on `GgufTensor` / `PthTensor`. Docs +
  test only — no library behaviour change.
- **Panic/abort-freedom as a tested invariant + the `unwind` build requirement**
  (Phase 6.13 Step 3). No public parse/inspect entry point panics or aborts on
  any input — promoted from an implicit property (a panic would crash the fuzzer)
  to an explicit, stable-CI contract: `tests/no_panic.rs` runs a `catch_unwind`
  battery (synthetic malformed shapes + truncations/bit-flips of the committed
  fixtures) over every entry point in debug, so integer-overflow panics are in
  scope, and three new owned-path `cargo fuzz` targets (`fuzz_safetensors_bytes`
  / `fuzz_gguf_bytes` / `fuzz_pth_bytes`) cover the copy-based untrusted entry.
  Adds a `[profile.python]` (`inherits = "release"`, `panic = "unwind"`) — the
  profile the Phase 8 PyO3 `cdylib` is built with so a panic surfaces as a
  catchable Python `PanicException` instead of an uncatchable abort — guarded by
  `tests/panic_profile.rs` (asserts release = abort, python = unwind). New
  `docs/python-interop.md` records the panic-safety + unwind contract.
- **Copy-based (`no-mmap`) full-parse entry points** for every mmap-backed
  format (Phase 6.13 Step 1): `parse_bytes` / `parse_from_reader` (safetensors),
  `parse_gguf_bytes` / `parse_gguf_from_reader`, and `parse_pth_bytes` /
  `parse_pth_from_reader`, each with a `_with_limits` variant. They read the
  artefact into an owned buffer (bounded by `ParseLimits::max_single_alloc_bytes`)
  and parse with **zero `unsafe` and zero mmap**, so a truncated or
  concurrently-written source yields a clean `Err` instead of an uncatchable
  `SIGBUS` — the **recommended entry point for untrusted input** (e.g. a
  user-uploaded file). The path-based `parse` / `parse_gguf` / `parse_pth` keep
  memory-mapping as the trusted-local-file fast path; behaviour and output are
  unchanged. Internally the three `Parsed*` types now hold their bytes behind a
  shared `Backing` (mmap or owned), so both paths share one type and one
  structure parser. Parity is pinned by `tests/parse_owned_path.rs`.
- **User documentation scaffold**, mirroring the sibling `hf-fetch-model`
  `docs/` system: a [`docs/FAQ.md`](docs/FAQ.md) (install, feature flags,
  supported formats, `inspect` vs `parse`, dequantizing/converting, parsing
  untrusted input) and two [`docs/tutorials/`](docs/tutorials/) walkthroughs —
  [Inspect before you parse (untrusted input)](docs/tutorials/inspect-before-you-parse.md)
  and [Dequantize a GGUF model to BF16](docs/tutorials/dequantize-a-gguf-model.md)
  — both with real captured output. Each doc carries a `STYLE CONVENTIONS`
  block and a freshness marker, and they are discoverable from a new
  **Documentation** table in the README.

### Changed

- **Finer-grained `AnamnesisError` taxonomy** (Phase 6.13 Step 2): two new
  variants on the `#[non_exhaustive]` enum. **`LimitExceeded { limit, message }`**
  is now returned for every budget/cap rejection — every `ParseLimits` axis
  (`max_single_alloc_bytes` / `max_total_bytes` / `max_item_count` /
  `max_decompression_ratio`) and every permanent per-format floor (`MAX_PKL_SIZE`,
  `MAX_PICKLE_WORKING_SET`, `MAX_PICKLE_VM_DEPTH`, `MAX_PICKLE_PAYLOAD`,
  `NPY_MAX_HEADER_BYTES`, `NPZ_MAX_ARRAY_BYTES`, the `GGUF` `MAX_*` family,
  `ZIP_MAX_ENTRIES`, `ZIP_MAX_NAME_LEN`, `MAX_SAFETENSORS_HEADER_BYTES`) — where
  these previously returned `Parse`. **`DisallowedGlobal { module, name }`** is
  now returned when a `.pth` pickle references a `GLOBAL` outside the `torch.*`
  allowlist (previously `Parse`). `Parse` keeps malformed/truncated/overflow
  cases; `Unsupported` and `Io` are unchanged. This lets a host branch on the
  error *kind* (e.g. *413* vs *400* vs a flagged security event) without
  string-matching. The frozen Rust→Python exception map (`ParseError` /
  `UnsupportedError` / `LimitExceededError` / `SecurityError` / `OSError`) is
  documented on `AnamnesisError` and in the README. **Observable** for code that
  matched the old variants on these conditions — acceptable pre-`1.0` and before
  the bindings ship.
- **Release builds now set `panic = "abort"`** (`[profile.release]` in
  `Cargo.toml`). The untrusted-input DoS analysis (parser docs, `CHANGELOG`
  Security notes) has long *assumed* abort-on-panic when arguing severity;
  this makes that the committed, fail-closed behaviour rather than an unstated
  assumption. Cargo auto-rebuilds test/bench targets with `unwind` (libtest
  requires it), so the test suite is unaffected. (The future PyO3 extension
  `cdylib` will override this back to `unwind` so panics surface as catchable
  Python exceptions — see the roadmap's Phase 6.13.)
- **Bumped the `safetensors` dependency from `0.4` to `0.8`** (latest at
  0.8.0, 2026-06-09). anamnesis uses `safetensors` only as the output writer
  (and a test-side reader) — production header parsing is anamnesis's own
  `parse::safetensors` — so the surface is small. Three upstream changes were
  accommodated:
  - **Metadata is now passed by value.** `serialize` / `serialize_to_file`
    take `Option<HashMap<String, String>>` instead of `&Option<…>` (since
    0.5), letting the writer move the map straight into the output header
    rather than cloning it. All call sites updated (`&None` → `None`,
    `&metadata` → `metadata`); the `remember_to_bytes` path now hands its
    metadata clone to the writer by value instead of cloning it a second time.
  - **New `Dtype` variants** (`F4`, `F6_E2M3`, `F6_E3M2`, `F8_E8M0`, the fp8
    `*FNUZ` variants, `C64`). `TryFrom<safetensors::Dtype> for Dtype` now
    rejects them explicitly with `AnamnesisError::Unsupported` — anamnesis has
    no dequant path for sub-8-bit floats / fp8-fnuz / complex64 yet — keeping
    the match exhaustive (the future-proofing wildcard no longer absorbs known
    variants).
  - **`SafeTensors::names()` now returns `Vec<&str>`** (was `Vec<&String>`); a
    test helper dropped its now-redundant `String::as_str` map.

  No public API or behaviour change for supported dtypes; output `.safetensors`
  bytes are unchanged. A Windows-only test adjustment came with the bump:
  safetensors' `serialize_to_file` opens the destination path itself, which
  collides with the open handle a `NamedTempFile` keeps on the same file, so
  the three tests that wrote through a temp **file** now write into a temp
  **dir** + fresh path.

### Fixed

- **`# Errors` doc accuracy across every parse/inspect entry point.** Many entry
  points — the copy-based `parse_*_bytes` / `parse_*_from_reader` wrappers, the
  `read` helper, and the pre-existing path/inspect/header functions (`parse`,
  `parse_safetensors_header[_from_reader]`, `parse_gguf`, `inspect_gguf_from_reader`,
  `inspect_npz[_from_reader]`, `parse_npz`, the `parse_pth` family) — still
  described the pre-Phase-6.13-Step-2 taxonomy: they attributed non-allowlisted
  pickle globals and over-cap declarations to `Parse`, where Step 2 reclassified
  them to `DisallowedGlobal` and `LimitExceeded`. Because the pickle allowlist and
  the permanent per-format caps are **always-on**, both variants are reachable even
  under `ParseLimits::default()`. The docs (and the FAQ) now name the variant each
  condition actually produces, so a caller routing on the error *kind* (e.g. a
  Python binding mapping to HTTP statuses) sees the true contract. Docs only — no
  behaviour change.

## [0.6.7] - 2026-06-15

Phase 6.12 — vendored ZIP container reader. Replaces `zip::ZipArchive` (the
`.pth` / `.npz` central-directory parser) with a lean, read-only, owned
central-directory reader, closing the measured metadata-amplification gap from
the Phase 6.8 "reopened by measurement" track (`ZipArchive::new` materialises
the whole central directory as a fat per-entry record — ~500 B/entry,
unreachable through `zip`'s API and unbounded by `ParseLimits`).

### Changed

- **New vendored ZIP central-directory reader** (`src/parse/zip.rs`,
  `#[cfg(any(feature = "npz", feature = "pth"))]`): a read-only EOCD +
  central-directory + local-header-offset parser with full `ZIP64` support
  (EOCD record + locator + `0x0001` extra field), so `torch.save` checkpoints
  larger than 4 GiB / 65 535 entries keep parsing. It owns only the container
  parsing — `DEFLATE` inflate stays in `flate2` / `miniz_oxide` (the
  upstream-fuzzed codec surface is untouched). The `.pth` mmap path
  (`build_entry_index`, `parse_pth`) now runs on the vendored reader. Hardened
  per `CONVENTIONS.md` "When Parsing Untrusted Input": bounds-checked cursor
  reads, `checked_*` offset arithmetic, a permanent entry-count cap
  (`ZIP_MAX_ENTRIES`, 1 048 576) and per-name cap (`ZIP_MAX_NAME_LEN`, 4 KiB),
  a `data_start + compressed_size <= file_len` cross-check, and a compression
  **allowlist** (`Stored` / `Deflate`). No public API or behaviour change for
  legitimate files. A new `fuzz_zip` target plus an in-crate differential
  oracle (every parse compared index-for-index against the `zip` crate across
  STORED / DEFLATE / `ZIP64` / multi-entry archives, including a 256-archive
  randomized sweep) back the migration.
- **`.npz` and the `.pth` reader path now run on the vendored reader too**
  (Step 2). A `ReaderSource` (`Read + Seek` substrate) and a `BoundedReader`
  (the entry's raw byte window) join `SliceSource`; `inspect_npz`, `parse_npz`,
  and `inspect_pth_from_reader` all parse the container through
  `read_central_directory`. `DEFLATE` `.npz` entries still inflate through
  `flate2` / `miniz_oxide` — the vendored reader hands back the raw compressed
  bytes and the consumer wraps them in the decoder, so the codec surface is
  unchanged. `.npy` entries with a non-`STORED`/`DEFLATE` method now fail fast
  with `AnamnesisError::Unsupported`.
- **`zip` is no longer a runtime dependency** (Step 3) — moved to
  `[dev-dependencies]` (kept for `ZipWriter` test fixtures + the differential
  oracle). The `npz` / `pth` features now pull only `flate2` (the inflate
  codec). The dead `From<zip::result::ZipError>` bridge is removed.
- **`.pth` entry index is now a sorted, trimmed `Vec<(Box<str>, …)>`**
  (binary-searched) instead of a `HashMap<String, …>`, removing the hash
  table's power-of-two bucket slack and `String`'s capacity word. Measured on a
  50 001-entry archive (`tests/peak_heap_zip_metadata.rs`, `dhat`): resident
  container metadata drops from **337 B/entry** (`zip` crate) to **41 B/entry**
  (vendored `parse_pth`) — an **8.07× resident reduction** (3.12× on peak),
  hitting the Phase 6.8 analysis's ~40 B/entry target. (That analysis projected
  ~12×, but the `zip` crate measures 337 B/entry on short entry names, not the
  estimated ~500, so ~8× is the real ceiling.)

### Security

- **Fixed a panic (DoS) in the `NPY` dtype-descriptor parser** found by the
  Phase 6.12 fuzzing campaign ([CWE-248](https://cwe.mitre.org/data/definitions/248.html),
  under the [CWE-400](https://cwe.mitre.org/data/definitions/400.html) umbrella).
  `parse_descr` sliced the descriptor string as `&descr[1..]`; when the `descr`
  field's first character was a multi-byte UTF-8 codepoint, byte index 1 fell
  inside that character and the `str` slice **panicked** (process abort under
  `panic = "abort"`) on a malformed `.npy` / `.npz`. Now uses `descr.get(1..)`,
  returning a clean `AnamnesisError::Unsupported` instead. This bug is
  **pre-existing** — introduced in the v0.3.0 custom NPZ parser (`bb58cd4`,
  2026-03-24) and latent in every release since; the longer, differently-seeded
  `fuzz_npz_limits` campaign reached it where the shorter v0.6.2/v0.6.3 runs had
  not. Verified fixed: the crash input now parses cleanly, and a 5.9 M-execution
  re-sweep across all ZIP/NPZ/PTH fuzz targets runs clean.
- **The vendored container reader now honours the caller's `ParseLimits`**
  ([CWE-770](https://cwe.mitre.org/data/definitions/770.html), under the
  [CWE-400](https://cwe.mitre.org/data/definitions/400.html) umbrella). A
  self-review found that `read_central_directory` enforced only the permanent
  `ZIP_MAX_ENTRIES` (1 M) floor, so `parse_*_with_limits`'s `max_item_count` /
  `max_single_alloc` were not consulted for the *container* metadata — a
  many-tiny-entry archive could drive a bounded but uncharged ~100 MB
  (`Vec<ZipEntry>` + the central-directory read) past a tight caller budget.
  The reader now takes `&ParseLimits` and rejects an over-budget declared entry
  count and central-directory allocation **before** allocating, fail-fast — the
  same invariant every other parser already upheld. On the mmap (`.pth`) path
  the central directory is now also read **zero-copy** (borrowed from the
  mapping via `ZipSource::as_slice`), dropping a transient full-directory copy.
  The inspect paths stay deliberately limit-free (the permanent floor still
  applies). No behaviour change for honest files under the default
  (unbounded) limits.
- **Bounded `DEFLATE` inflation on the `.pth` reader path.** A self-review of
  the new `.pth` reader path (`inspect_pth_from_reader`) found that
  `read_pth_entry_bytes` read a `DEFLATE`-compressed `data.pkl` / `byteorder`
  entry with `read_to_end`, which expands the entire compressed stream
  regardless of the declared `uncompressed_size` — a few-KiB entry could inflate
  to gigabytes (a zip bomb, [CWE-409](https://cwe.mitre.org/data/definitions/409.html)
  / the [CWE-400](https://cwe.mitre.org/data/definitions/400.html) umbrella), and
  the `Vec::with_capacity(uncompressed_size)` could eagerly commit up to the
  100 MiB `MAX_PKL_SIZE` from a tiny file that merely *declares* that size. The
  read is now bounded with `Read::take(uncompressed_size)` (the caller already
  caps that value) and grows the buffer lazily, so neither the inflation nor the
  allocation can exceed the declared, capped size. `STORED` entries and honest
  `DEFLATE` entries are unaffected (real `.pth` `data.pkl` is `STORED`); the
  `.npz` path was already bounded (it reads an exact, cross-checked array length,
  never `read_to_end`).

## [0.6.6] - 2026-06-13

Phase 6.11 — pickle-VM working-set governance. Closes a P0 DoS an independent
security audit surfaced in v0.6.3 (`docs/security-audit-{brief,findings}.md`,
internal, not tracked in the repo).

### Security

- **The `.pth` pickle VM now governs its working set.** Previously the VM
  bounded only the `data.pkl` opcode-stream length (`MAX_PKL_SIZE`, 100 MiB)
  and individual string/bytes payloads; the value stack, memo clones, and
  nesting depth were charged to nothing, so a small crafted pickle could drive
  multi-GiB heap (an `N`-flood, or `BINGET` replay of a large memoised
  subtree) or a recursive-`Drop` stack overflow — a process abort under the
  crate's `panic = "abort"`, reachable from `parse_pth` **and** the cheap
  `inspect_pth_from_reader` pre-filter the README recommends for untrusted
  input. The VM now routes every value-creating opcode through a single
  accounting choke point that:
  - charges each pushed value's heap (enum slot + owned payload) and the deep
    size of every memo clone to a permanent `MAX_PICKLE_WORKING_SET` floor
    (512 MiB) **and** the caller's `ParseLimits::max_total_bytes`, bounding the
    stack-flood and memo-replay amplification vectors (CWE-1325, improperly
    controlled sequential memory allocation); and
  - caps construction nesting depth at a permanent `MAX_PICKLE_VM_DEPTH` (256),
    so an over-deep value never forms and recursive `Drop` (and every recursive
    walk) stays shallow (CWE-674, uncontrolled recursion).

  Those two weaknesses fall under the CWE-770 / CWE-400 umbrella the
  header-allocation caps (v0.6.1–v0.6.3) already address, but the amplification
  and recursion shapes are distinct from a single oversized-field allocation.

  Both floors are always-on (enforced even under `ParseLimits::default()`), in
  `O(1)` per opcode (a parallel depth/size metadata stack — a per-push deep
  walk would itself be an `O(n²)` CPU-DoS). No public API change; honest files
  are unaffected (the floors are ~3 orders of magnitude above any real
  `state_dict`). The `data.pkl` allowlist, opcode set, and tensor extraction
  are unchanged.

### Changed

- **`ParseLimits::max_total_bytes` now also counts the `.pth` pickle VM's
  working set** (every pushed value + memo-clone deep size), not just the
  string/bytes payloads it charged before. A caller who set a pathologically
  tight `max_total_bytes` calibrated to the old payload-only accounting may see
  it bite earlier; realistic budgets are unaffected. This is the intended
  effect of the Security fix above (the old accounting under-counted the real
  VM heap).

### Fixed

- **BnB4 dequant/encode now reject an odd `block_size`.** Two 4-bit values pack
  into one byte, so `bytes_per_block = block_size / 2`; an odd `block_size`
  truncated that division and mis-aligned every block after the first, yielding
  wrong (not out-of-bounds) output. The decode (`dequantize_bnb4_to_bf16`,
  `dequantize_bnb4_double_quant_to_bf16`) and encode (`encode_bnb4`,
  `encode_bnb4_compute_absmax`, `encode_bnb4_double_quant`) entry points now
  return `AnamnesisError::Parse` for an odd `block_size`. (Bundled defense-in-
  depth nit from the same audit.)

## [0.6.5] - 2026-06-10

Phase 6.10 — the second candle-mi dogfooding fix. Full post-mortem:
`docs/dogfooding-feedbacks/awq-gptq-dequant-transpose-orientation.md`
(internal dogfooding report, not tracked in the repo).

### Fixed

- **BREAKING (behavior): AWQ and GPTQ `remember` output is now standard
  `nn.Linear` orientation `[out_features, in_features]`.** The dequant
  kernels produce the GEMM-native `[in, out]` layout (the canonical
  AutoAWQ / GPTQModel kernel orientation, which the 0-ULP cross-validation
  anchors — values were always correct), but `remember` /
  `remember_to_bytes` previously serialized that layout as-is, so every
  2-D projection weight came out transposed and no standard consumer
  (candle, `transformers` as a plain model) could load the output
  (`shape mismatch … expected [512, 2048], got [2048, 512]` on Llama
  `k_proj`). The remember path now transposes each AWQ/GPTQ quantized
  weight at the output-contract boundary — exactly what GPTQModel's own
  `dequantize_model` does (`.T`) when assigning its GEMM-native dequant to
  a plain `nn.Linear`. Passthrough tensors (norms, biases, 2-D
  `embed_tokens` / `lm_head` kept in BF16) are untouched; BnB and FP8
  were already standard-orientation. Anyone consuming the old `[in, out]`
  output must drop their compensating transpose.

### Added

- **Output-orientation contract tests** (`tests/remember_orientation.rs`):
  every quantized scheme (GPTQ, AWQ, BnB NF4, BnB INT8, FP8) is now
  loaded through the public `parse → remember_to_bytes →
  safetensors::deserialize` path on a synthetic **non-square** model,
  asserting both the emitted shape (`[out, in]`) and the element mapping
  (`W_std[o][i] == W_native[i][o]` for the transposed schemes, identity
  for the rest), plus passthrough byte-identity. The GGUF↔safetensors
  shape reversal is pinned with an absolute-orientation assert in the
  convert round-trip test. Closes the gap the dogfooding report
  identified: value-only cross-validation is structurally blind to the
  emitted layout.

## [0.6.4] - 2026-06-10

Phase 6.9 — exact-parity fixes surfaced by dogfooding `v0.6.3` inside candle-mi.
Full post-mortem:
`docs/dogfooding-feedbacks/bnb-nibble-order-and-circular-fixture-validation.md` (internal dogfooding report, not tracked in the repo).

### Fixed

- **BnB NF4/FP4 nibble order (decode + encode).** `dequantize_bnb4_core`
  unpacked the two 4-bit values in each byte low-nibble-first; the
  `bitsandbytes` kernel is **high-nibble-first** (`byte >> 4` → element `2i`,
  `byte & 0x0F` → element `2i + 1`). Every NF4/FP4 dequant produced
  correctly-valued but **element-permuted** weights (adjacent pairs swapped) —
  confirmed end-to-end by candle-mi's forward-parity gate (garbage logits,
  `max|Δlogit| ≈ 19.9` → top-1 ` Paris` after the fix). The `lethe` encoder
  (`encode_bnb4_core`) is mirrored, so the written nibbles now match
  `bitsandbytes`' on-disk layout byte-exactly *and* the decode↔encode round
  trip still closes.
- **BnB double-quant `nested_offset` was dropped (decode + encode + parse).**
  Real `bitsandbytes` double quantization recovers
  `absmax = nested_dequant(absmax_u8) + nested_offset`, where the offset (the
  mean of the original absmax values, ~0.05–0.08 on real checkpoints) is
  stored in the `quant_state` JSON blob. anamnesis ignored it, biasing every
  recovered absmax low — every double-quant BnB model (the `bitsandbytes`
  default; the entire unsloth catalog checked) dequantized with a systematic
  error. `ParsedModel::remember`/`remember_to_bytes` now parse
  `nested_offset` from the blob (mandatory for DQ tensors — a DQ tensor
  without it is rejected as malformed).
- **AWQ GEMM nibble interleave was missing (decode).** AutoAWQ packs the 8
  nibbles of each `I32` in the order `AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]`
  (for **both** `qweight` and `qzeros`); anamnesis unpacked sequentially,
  producing column-permuted output — 44 468 / 65 536 elements wrong on the
  re-anchored Llama fixture. `remember::awq` now applies the canonical
  interleave (`AWQ_ORDER` / `AWQ_REVERSE_ORDER`, mirroring
  `awq/utils/packing_utils.py`).

### Changed

- **BREAKING: `dequantize_bnb4_double_quant_to_bf16` and
  `encode_bnb4_double_quant` take a new `nested_offset: f32` parameter** (the
  `bitsandbytes` `QuantState.offset`). Callers reading real `bitsandbytes`
  checkpoints must pass the value from the `quant_state` blob; `0.0`
  reproduces the only case the old signature handled correctly (synthetic
  states that were never offset-compressed). The high-level
  `remember`/`remember_to_bytes` paths extract it automatically.
- **BREAKING: AWQ dequantization rejects `bits != 4`.** AutoAWQ's GEMM format
  is 4-bit only (its packer raises `NotImplementedError` for other widths), so
  no canonical nibble interleave exists to anchor an 8-bit decode against —
  and no real 8-bit AWQ-GEMM checkpoints exist (the "8-bit AWQ" models on the
  Hub are misnamed 4-bit GEMM or `compressed-tensors`). Rejecting beats
  emitting plausibly-permuted weights.
- **Cross-validation fixtures re-anchored on the canonical libraries' own
  code** (the methodology fix — the previous generators reimplemented the
  dequant formulas in plain PyTorch and shared the Rust code's blind spots,
  validating the BnB and AWQ bugs green at 0 ULP):
  `generate_bnb.py` → `bitsandbytes.functional.dequantize_4bit` /
  `int8_vectorwise_dequant` (CUDA kernel at generation time — measured
  bit-identical to anamnesis' single f32→BF16 rounding, unlike the
  double-rounding 0.49 CPU kernel; committed fixtures stay CPU-checkable);
  `generate_awq.py` → AutoAWQ `unpack_awq` + `reverse_awq_order`;
  `generate_gptq.py` → GPTQModel `TorchLinear.dequantize_weight` +
  `convert_gptq_v1_to_v2_format_module`. GPTQ's convention was **confirmed
  correct** by re-anchoring (the on-disk `+1` zero-point offset matches
  GPTQModel's loader-side v1→v2 conversion). The BnB fixture format gains a
  `nested_offset` header field; AWQ/GPTQ cross-validation tightened from
  `max_ulp = 1` to `0`. GGUF/Ollama/NPZ/PTH/safetensors/FP8 generators were
  audited and already genuinely anchored (real `gguf-py`, the formats' own
  libraries, torch's native fp8 cast).

### Added

- **`ParsedModel::remember_to_bytes(target) -> Result<Vec<u8>>` (Phase 6.8 Step 6).**
  An in-memory twin of `remember`: the identical `BF16` dequant and
  companion-grouping, but returns the serialized `.safetensors` bytes instead of
  writing a file, so an embedder can load the dequantised model without a disk
  round-trip (e.g. candle-mi's quantized loader → `from_buffered_safetensors`).
  Completes the file/bytes pairing the crate's other serializers already have
  (`to_safetensors[_bytes]`, `write_bnb_nf4_safetensors[_bytes]`). The shared
  dequant + view-building is factored so `remember` (→ `serialize_to_file`,
  streamed) and `remember_to_bytes` (→ `serialize`, one `Vec`) differ only in the
  destination. Eager `O(2·n_params)` peak; the streaming, peak-bounded variants
  are planned for Phase 10.

### Changed

- **Inflate-only `zip` dependency (Phase 6.8 Step 4).** Switched the optional
  `zip` dependency from the umbrella `deflate` feature to `["deflate-flate2",
  "flate2"]` plus a direct `flate2 = { default-features = false, features =
  ["rust_backend"] }`, gated on the `npz`/`pth` features. This drops `zopfli`
  (a DEFLATE *compressor* this read-only consumer never runs) and its
  `bumpalo`/`log` transitives from the dependency tree while keeping
  `miniz_oxide` for inflate. Pure dependency hygiene — no API or behaviour
  change; `np.savez_compressed` archives still inflate bit-exactly.

### Security

- **Caller-configurable `ParseLimits` (Phase 6.8 Steps 1–3).** A new
  `ParseLimits` budget lets a caller tighten the parsers' resource use to its
  own environment (an edge board, a per-slot MLaaS worker) below the built-in,
  server-scale constant caps. Four axes: `max_single_alloc` (largest single
  header-declared buffer), `max_total_bytes` (cumulative parse-time heap — the
  running sum of every eager allocation, closing the many-small-items blow-up a
  per-item cap misses), `max_item_count` (declared tensor / array / KV /
  archive-entry count), and `max_decompression_ratio` (the zip-bomb cap — rejects
  a `DEFLATE` `NPZ` entry whose declared uncompressed size is an absurd multiple
  of its compressed size, from archive metadata before any allocation). Each is
  enforced fail-fast, with `AnamnesisError::Parse`, **before** allocation, and is
  tighten-only — where a permanent per-format cap exists (the single-allocation
  caps, the `GGUF` counts) the effective bound is `min(cap, limit)`, never below
  that floor; the aggregate and ratio axes have no built-in cap and are pure
  caller bounds. New entry points `parse_with_limits`, `parse_gguf_with_limits`,
  `parse_npz_with_limits`, `parse_pth_with_limits`,
  `parse_safetensors_header_with_limits`, and
  `parse_safetensors_header_from_reader_with_limits`; the existing limit-free
  functions delegate with `ParseLimits::default()` (unbounded), so default
  behaviour is byte-for-byte unchanged — no breaking change. The recommended
  **inspect-before-parse** policy-gate pattern (cheap `inspect_*_from_reader`
  → check the reported totals against your policy → `parse_*_with_limits`) is
  documented front-of-README, and the `cargo-fuzz` harness gains three
  `*_limits` targets that co-explore malformed files against input-derived
  limits, exercising every enforcement branch.
- **Clamped zip entry-count pre-allocations (consistency hardening).** The
  `NPZ` and `.pth` parsers no longer size a `Vec` / `HashMap::with_capacity`
  hint from the untrusted zip entry count; they clamp it to the shared
  `PREALLOC_SOFT_CAP`, matching the `GGUF` parser, and the containers grow as
  entries are inserted (no behaviour change for legitimate files). This stops
  the parser eagerly reserving for entries it may skip. Note: the *dominant*
  parse-time memory for a many-entry archive is the `zip` crate's own
  central-directory model (measured at ~5.7× the file size for 50 000 tiny
  entries — a fat per-entry record with the filename stored 2–3×), which this
  clamp does **not** address; bounding that requires the vendored container
  reader tracked for a later phase.

## [0.6.2] - 2026-06-04

### Security

- **Pre-Phase-7 security audit hardening (Phase 6.7).** A full audit of every
  parser and transform layer against the [`candle #3533`](https://github.com/huggingface/candle/issues/3533)
  /[`#3556`](https://github.com/huggingface/candle/pull/3556) DoS class, ahead
  of the Python bindings. No high-severity bug; three low-to-medium findings
  closed. No public API change, no behaviour change for legitimate files.
  - **NPZ** `read_array_data` now rejects a shape-derived `data_bytes` that
    exceeds the ZIP entry's declared uncompressed `size()` **before**
    allocating — bounding both over-declared shapes and `DEFLATE` expansion to
    the entry's honest size (the absolute `NPZ_MAX_ARRAY_BYTES` cap remains the
    secondary bound).
  - **GGUF** `read_bytes` validates the declared length against the remaining
    file bytes (`ensure_remaining`) **before** the allocation, so a tiny file
    declaring a string up to `MAX_STRING_LEN` (16 MiB) can no longer drive an
    eager allocation.
  - **Checked shape-products:** a shared `checked_num_elements` helper replaces
    the unchecked `shape.iter().product()` at the two production sites
    (`shape_to_rows_cols`, `is_eligible_for_nf4`), removing a debug-panic /
    release-wrap on adversarial shapes.
  - Added a `cargo fuzz` harness (`fuzz/`, dev-only, excluded from the
    published crate) with a libFuzzer target per parser; authored and run under
    WSL2 with zero crashes across ~1.5 M+ executions.

## [0.6.1] - 2026-05-29

### Security

- **Hardened all parsers against unguarded-allocation DoS (Phase 6.6,
  triggered by the [`candle #3533`](https://github.com/huggingface/candle/issues/3533)
  audit; CWE-770 / CWE-1284 / CWE-400).** Legitimate files are unaffected;
  malformed or malicious inputs now fail fast with an `AnamnesisError::Parse`
  instead of driving multi-GiB allocations. No public API change.
  - **PTH mmap path** now enforces the existing `MAX_PKL_SIZE` (100 MiB) cap
    on `data.pkl` before slicing the mmap and running the pickle VM — it was
    previously bounded only by file size, while the reader path already
    capped. Both paths now route through a shared `enforce_pkl_size_cap`
    helper, and the mmap slice end is computed with `checked_add`.
  - **NPZ `header_len`** (NPY v2/v3 `u32`, reachable up to 4 GiB) is rejected
    against a new `NPY_MAX_HEADER_BYTES` (1 MiB) cap before allocating the
    header buffer.
  - **Defence in depth:** an `NPZ_MAX_ARRAY_BYTES` (8 GiB) cap makes oversized
    declared array shapes fail deterministically below the `usize`-overflow
    boundary, and a `MAX_PICKLE_PAYLOAD` (64 MiB) per-opcode cap bounds the
    `BINUNICODE` / `BINSTRING` / `BINBYTES` clones in both pickle entry points.
  - Regression tests pin each guard (malicious-fixture → `Err`), including an
    `#[ignore]`d >100 MiB end-to-end PoC for the PTH mmap cap.

## [0.6.0] - 2026-05-21

**Phase 6 lands — the format conversion matrix.** v0.6.0 ships a
unified `amn convert` dispatch covering every decode-side conversion
(any input → safetensors BF16), the BnB-NF4 encode path
(safetensors-BF16 → BnB-NF4 safetensors via the four-tensor companion
layout), and the format-symmetric inverse of `parse_gguf`
(`write_gguf`, scalar dtypes today; quantised dtypes reserved for
Phase 7.5 through the same scaffold). Measured **1.11×–6.75× faster
than the closest Python ecosystem default** (numpy / gguf-py /
bitsandbytes CPU) at 4096×4096 on CPU, **2.17×–8.24× faster than
PyTorch-CPU equivalents** for the two non-PyTorch paths. 13
byte-exact integration tests cover every v0.6.0 conversion pair both
directions where reversible, plus a size-matched perf comparison
against six checked-in Python sidecars.

### Added

- **GGUF output writer** (`write_gguf`, `write_gguf_to_writer`,
  `GgufWriteTensor`) — the format-symmetric inverse of `parse_gguf`.
  Phase 6 Step 1: emits scalar dtypes (`F32`, `F16`, `BF16`, `F64`,
  `I8`–`I64`) plus the full metadata KV table; quantised dtypes
  (`Q*`, `IQ*`, `TQ*`, `MXFP4`) are rejected with `Unsupported`
  pending the Phase 7.5 encoders. Files round-trip byte-exactly
  through `parse_gguf`. Behind the `gguf` feature flag.
- **`amn convert` CLI subcommand** — unified entry point that routes
  every v0.6.0-available format pair through a single dispatch.
  Targets: `safetensors` (alias `bf16`), `gguf` (unquantised
  passthrough), `bnb-nf4`. Phase 6 Step 2. New conversion paths
  unlocked: `safetensors-bf16 → bnb-nf4` (encodes 2-D weights to
  BnB-NF4 with the four-tensor companion layout), `safetensors-bf16
  → gguf` (writes a self-describing GGUF), `npz → safetensors` (and
  `npz → gguf`), `pth → gguf`. Quantised GGUF targets (`gguf-q4km`,
  …) land in Phase 7.5 via the same dispatch. Includes the new
  library helpers `npz_to_safetensors` / `npz_to_safetensors_bytes`
  (behind `npz`) and `write_bnb_nf4_safetensors` /
  `write_bnb_nf4_safetensors_bytes` / `is_eligible_for_nf4` /
  `classify_inputs` (behind `bnb`).
- **Cross-format round-trip validation suite**
  (`tests/cross_validation_convert.rs`) — 13 integration tests
  covering every v0.6.0-available format pair, both directions where
  reversible. Phase 6 Step 3. Includes BnB-NF4 byte-exactness against
  the existing Python `bitsandbytes` fixture (`llama_1b_nf4.bin`),
  full `safetensors-BF16 → GGUF → safetensors-BF16` byte-exact loops,
  full `GGUF → safetensors → GGUF` byte-exact loops, BnB-NF4 encode
  idempotency, multi-hop `NPZ → safetensors → GGUF`, mixed-dtype
  safetensors → GGUF, multi-dimensional shape preservation (1-D,
  3-D, 4-D), non-default `general.alignment = 8` round-trip, and
  empty-GGUF metadata-only round-trip. Each test records wall-clock
  µs and compares against a checked-in Python sidecar
  (`tests/fixtures/convert_reference/*.timing.json`, refreshable via
  `generate_convert_timings.py`) when one is available — silently
  skips the comparison line when not, so the suite has no Python
  runtime dependency.
- **Size-matched performance comparison vs Python (CPU)** —
  `t14_perf_vs_python_size_matched` in
  `tests/cross_validation_convert.rs` (gated `#[ignore]`, opt-in via
  `--ignored`) runs each forward conversion on the same 4096×4096
  shape the Python sidecar generator uses (~32 MiB BF16 / 64 MiB F32),
  so elapsed times compare apples-to-apples. Six Python sidecars
  checked in (four ecosystem-default: numpy + safetensors-py,
  torch.load + safetensors.torch, gguf-py, bitsandbytes CPU; plus two
  PyTorch-CPU equivalents for the non-PyTorch paths:
  `npz_to_st_torch`, `st_to_gguf_torch`). Headline ratios at CPU,
  release build, `target-cpu=native`: `npz → safetensors` 6.75× vs
  numpy / 8.24× vs torch; `pth → safetensors` 5.18× vs torch;
  `safetensors-BF16 → GGUF` 1.11× vs gguf-py / 2.17× vs torch+gguf-py;
  `safetensors-BF16 → BnB-NF4` 2.67× vs bitsandbytes CPU. Full table
  in `README.md` under "Format Conversion Pipeline (Phase 6, v0.6.0)".

## [0.5.0] - 2026-05-17

**Lethe lands.** Phase 5 ships the encode-side inverse of `remember`:
the `lethe` namespace with `BnB` encode kernels (`NF4` / `FP4` /
`INT8`, plain + double-quant), the bit-exact round-trip validation
harness every subsequent encode kernel family will reuse, and the
sign-of-zero preservation tweak in the decode path that lets the
round-trip recover the original nibble byte-exactly even on the
bitsandbytes Python `FP4` `quant_map` that collapses `+0.0` and `-0.0`
to the same bits. Cross-validated byte-exact (0 byte diffs) against
the original PyTorch-quantised bytes on **7 fixtures across 4
architecture families** (Llama 3.2 / Qwen3 / Qwen2.5 / Phi-3.5).

### Added

- **`encode_bnb4_double_quant`** — fourth `BnB` encode kernel, the
  inverse of `dequantize_bnb4_double_quant_to_bf16`. Recovers the
  per-block `f32` absmax from the `U8` quantised absmax + nested
  codebook using the same formula the decoder applies, then re-encodes
  `BF16` to packed nibbles via the same nearest-codebook search as
  `encode_bnb4`. Signature strict-
  mirrors the decode side: caller supplies `absmax_data` (`U8`),
  `quant_map_data`, `nested_absmax_data`, `nested_quant_map_data`,
  plus `block_size` and `nested_block_size`. Promoted from a deferred
  Phase 5 polish item to a required Step 1c gate after `hf-fm inspect`
  on candidate cross-architecture models surfaced that every non-Llama
  `bnb-4bit` upload in the wild uses double-quant (the bitsandbytes
  default). The strict mirror is the round-trip API; a future
  `_compute_*` convenience that derives all metadata from a fresh
  `BF16` source is needed by the Phase 6 "any input -> BnB-NF4
  safetensors" conversion path and is deferred there.
- **Two cross-architecture NF4 double-quant fixtures (Qwen2.5 + Phi-3.5)** —
  new `tests/fixtures/bnb_reference/qwen2_5_1_5b_nf4_dq.bin` (from
  `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`) and `phi3_5_mini_nf4_dq.bin`
  (from `unsloth/Phi-3.5-mini-instruct-bnb-4bit`), plus PyTorch
  quantize-timing sidecars. Decode + encode cross-validation tests
  (`cross_validate_{qwen2_5_1_5b,phi3_5_mini}_nf4_dq` and
  `cross_validate_encode_{qwen2_5_1_5b,phi3_5_mini}_nf4_dq`) confirm
  the `encode_bnb4_double_quant` kernel works across three
  architectures (Llama, Qwen2.5, Phi-3.5) at byte-exact round-trip
  versus the original PyTorch `bitsandbytes` bytes.
- **Cross-architecture plain-`FP4` fixture (Qwen3)** — new
  `tests/fixtures/bnb_reference/qwen3_mcqa_fp4.bin` extracted from
  `ema1234/qwen_mcqa_bnb_fp4` (Qwen3 architecture, different HF org
  from the existing `HF-Quantization/Llama-3.2-1B-BNB-FP4` Llama
  fixture). Decode + encode cross-validation tests
  (`cross_validate_qwen3_mcqa_fp4` and
  `cross_validate_encode_qwen3_mcqa_fp4`) confirm the sign-of-zero
  preservation rule (introduced for the Llama `FP4` fixture)
  generalises beyond a single org's quantization pipeline — byte-exact
  round-trip holds on Qwen3 too. Phase 5 step 1b deliverable per
  `ROADMAP.md`.
- **PyTorch quantize-timing sidecars for double-quant fixtures** —
  `generate_bnb.py` now also times a `bitsandbytes`-style
  double-quant quantize pass (with pre-recovered absmax matching the
  Rust `encode_bnb4_double_quant` signature) and writes the result
  alongside the existing plain-quant sidecars. Rust tests print
  side-by-side anamnesis-vs-PyTorch quantize timings for every fixture.
- **`lethe` namespace** (`src/lethe/`) — precision-compression (encoding)
  counterpart to `remember`. Phase 5 ships `BnB` encode plus a generic
  bit-exact round-trip validation harness; subsequent encode kernel
  families (`FP8`, `GGUF` legacy / `K-quants` / `IQ` / `TQ` / `MXFP4`)
  will reuse the same harness in Phase 7.5. Feature-gated behind the
  existing `bnb` feature (no new feature flag).
- **`encode_bnb4`** / **`encode_bnb4_compute_absmax`** — `NF4` / `FP4`
  encode to packed nibbles. `encode_bnb4` mirrors the
  `dequantize_bnb4_to_bf16` signature exactly (caller supplies absmax
  + codebook); the `_compute_absmax` variant derives per-block absmax
  from the source `BF16`. Round-trip is byte-exact on every shipped
  `BnB` fixture.
- **`encode_bnb_int8`** / **`encode_bnb_int8_compute_scb`** —
  `LLM.int8()` per-row absmax + `i8` round-to-nearest with clamp to
  `[-128, 127]`. Round-trip is byte-exact (every `i8` value recovers).
- **`NF4_CODEBOOK`** / **`FP4_CODEBOOK`** — canonical 16-entry
  `bitsandbytes` lookup tables exposed as `pub const`. The `FP4`
  constant stores `-0.0` distinct from `+0.0` at index 8 (recovering
  the sign-of-zero information that `bitsandbytes`' Python on-disk
  `quant_map` collapses to `+0.0`).
- **`lethe::round_trip`** — generic bit-exact round-trip harness:
  `assert_bnb4_decode_encode_round_trip` and
  `assert_bnb_int8_decode_encode_round_trip` for `BnB`; future kernel
  families will add sibling helpers.
- **Sign-of-zero preservation in `dequantize_bnb4_to_bf16`** — when a
  looked-up codebook entry is exactly `+0.0` AND the nibble's high bit
  is set (`nibble & 0x8 != 0`), the emitted `BF16` is `-0.0` instead
  of `+0.0`. This is a deliberate, narrow divergence from
  `bitsandbytes`' Python decode that lets `encode_bnb4` round-trip
  every `BnB` fixture byte-exactly (`FP4` included). Arithmetically
  invisible (both `+0` and `-0` are IEEE 754 zero); the only
  observable difference is the sign bit on `0.2 %` of `FP4` elements.
  No-op for `NF4` and every codebook whose upper-half indices hold
  non-zero entries.
- Cross-validation test (`tests/cross_validation_bnb_encode.rs`) round-trips
  every shipped `BnB` fixture (`NF4`, `FP4`, `INT8`) at 0-ULP bit
  exactness against the original `bitsandbytes`-quantised
  `weight_data`. Optional `PyTorch` quantize-timing sidecar
  (`<fixture>.timing.json`) prints side-by-side runtime comparison
  when present.
- `tests/fixtures/bnb_reference/generate_bnb.py` now also times
  `bitsandbytes`-equivalent quantize passes and emits the
  `<fixture>.timing.json` sidecar for the encode cross-validation
  runtime comparison.

### Changed

- `tests/cross_validation_bnb.rs` `compare_bf16` helper now treats
  `+0` and `-0` `BF16` as IEEE 754 equivalent so the deliberate
  sign-of-zero divergence in `dequantize_bnb4_to_bf16` does not break
  the existing decode bit-exactness contract on `FP4` fixtures.

### Known gaps

- `encode_bnb4_double_quant` (nested-absmax encode) is **not** shipped
  in this drop. Phase 5 step 1 covers three encode kernels (`NF4`,
  `FP4`, `INT8`); double-quant encode adds a fourth kernel
  (nested-absmax derivation + nested-codebook search) and is tracked
  for a follow-up commit. The `llama_1b_nf4_double_quant` fixture
  continues to validate the decode path in
  `cross_validation_bnb.rs`.

## [0.4.6] - 2026-05-14

### Added

- **`inspect_pth_from_reader<R: Read + Seek>`** — reader-generic `.pth`
  metadata inspection. Accepts any `Read + Seek` substrate (in-memory
  `Cursor`, HTTP-range-backed adapter, custom transport) and returns the
  same `PthInspectInfo` (`tensor_count`, `total_bytes`, `dtypes`,
  `big_endian`) as the existing path-based `parse_pth(path).inspect()`,
  without materialising any of the tensor-data files inside the archive
  (`data/0`, `data/1`, …). Only the ZIP central directory and the
  `data.pkl` entry — typically <100 KiB even on torchvision-class 300 MB
  models — are read. A 300 MB torchvision `.pth` inspects with well under
  100 KiB of network transfer through an HTTP-range adapter, instead of
  300 MB. Unblocks `hf-fm` v0.11.3 (remote `.pth` inspect) on top of
  v0.11.0's `HttpRangeReader` adapter, closing the remote-inspect matrix
  across all four tensor formats anamnesis supports (`safetensors`, `NPZ`,
  `GGUF`, `.pth`). Anamnesis itself takes on no network or TLS dependency.
  Phase 4.10; see [`ROADMAP.md`](ROADMAP.md).
- **`parse_pth` and `inspect_pth_from_reader` re-exported from the crate
  root** — both functions are now reachable as `anamnesis::parse_pth` and
  `anamnesis::inspect_pth_from_reader` (the former was already re-exported;
  the latter is new). Joins the existing reader-generic family
  (`parse_safetensors_header_from_reader`, `inspect_npz_from_reader`,
  `inspect_gguf_from_reader`).
- **8 new `.pth` unit tests** covering the reader-generic path:
  `inspect_from_reader_matches_path_empty_dict` (substrate-equivalence on a
  synthetic minimal archive), `inspect_from_reader_honours_byteorder_entry`
  (explicit `byteorder` entry plumbed into `PthInspectInfo`),
  `inspect_from_reader_rejects_legacy_format` (pre-`PyTorch` 1.6 raw
  pickle surfaces as `Unsupported`),
  `inspect_from_reader_rejects_wrong_magic` (non-ZIP / non-legacy bytes),
  `inspect_from_reader_rejects_too_small_file` (fewer than 4 bytes),
  `inspect_from_reader_rejects_missing_data_pkl` (ZIP with no `data.pkl`
  entry), `inspect_from_reader_accepts_older_prefix` (older-style
  `{model_name}/` archive prefix, exercising the suffix-stripping path),
  and `inspect_from_reader_rejects_oversized_byteorder` (the 64-byte
  `byteorder` cap fires before allocating).
- **Shared `build_pth_inspect_info` helper** — both `ParsedPth::inspect()`
  (mmap path) and `inspect_pth_from_reader` (reader-generic path) now
  delegate field computation to the same private function, guaranteeing
  substrate equivalence by construction. Mirrors Phase 4.9's
  `build_inspect_info` for `GGUF`.
- **Shared `interpret_pickle_to_meta` helper** — both entry points run the
  pickle VM via the same private function, so the security allowlist
  (`is_allowed_global`) is identical across the two paths. Any future
  tightening of the pickle interpreter automatically applies to both.
- **Substrate-equivalence integration test on the `AlgZoo` fixtures**
  (`tests/cross_validation_pth.rs::substrate_equivalence_algzoo_fixtures`)
  — for each of the 3 `AlgZoo` fixtures, asserts field-for-field
  equivalence of `PthInspectInfo` across the three substrates
  `parse_pth(path).inspect()`, `inspect_pth_from_reader(File::open(path)?)`,
  and `inspect_pth_from_reader(Cursor::new(fs::read(path)?))`.
- **Cross-language performance comparison** documenting the reader path
  vs `PyTorch`'s `torch.load(weights_only=True)`. On the 3 `AlgZoo`
  fixtures checked into the repo (best-of-5 release-mode median,
  `target-cpu=native`, `PyTorch` `2.10.0+cu130`),
  `inspect_pth_from_reader` is **2.4–3.6× faster than `torch.load`**
  even though `torch.load` has no separate inspect-only primitive — the
  speedup is a lower bound that grows by orders of magnitude on
  torchvision-class models, where the reader path stays bounded by
  `data.pkl` size while `torch.load` scales linearly in total
  tensor-data size. Bench harness:
  `tests/bench_pth_inspect_adhoc.rs` (Rust side, `#[ignore]`-gated) +
  `tests/fixtures/pth_reference/bench_python_inspect.py` (`PyTorch`
  side). Full method + numbers in
  [`docs/perf-experiments.md`](docs/perf-experiments.md) Experiment 6.
- **Broad-sample validation on the full 6 960-file `AlgZoo` corpus**
  (the `algzoo_weights/` set imported for `candle-mi` v0.1.9's
  `stoicheia` module — 22.6 MiB total, median file size 2.5 KiB, 4
  task families). New `bench_pth_inspect_algzoo_sweep` Rust test +
  `ANAMNESIS_ALGZOO_DIR` sweep mode of `bench_python_inspect.py`
  walk a configurable external directory, time every `.pth` file in
  both substrates plus `torch.load`, and report aggregate
  distributions plus per-task-family breakdown. Across all 6 960 files
  (per-file medians, µs):
  - mmap path: median **124.0**, p25 122.3, p75 128.4 (tight, σ ≈ 3 µs).
  - reader path: median **168.7**, p25 165.9, p75 173.1.
  - `torch.load(weights_only=True)`: median **504.3**, p25 500.6, p75 512.7.

  Median speedups across the full corpus: mmap **4.07× faster than
  `torch.load`**, reader **2.99× faster than `torch.load`**, reader
  takes **1.36× the time** of mmap (p25 1.34×, p75 1.39× — the
  reader/mmap gap is structurally fixed at ~45 µs on KiB-scale `.pth`
  files). The widened evidence base confirms the rustdoc's parity claim
  and re-derives Phase 4.10's *Re-attempting* threshold (revisit
  `BufReader<R>` only if a torchvision-class file exceeds 1.5× of the
  mmap path). Full method, per-family breakdown, and the
  `longest_cycle`-family outlier analysis in
  [`docs/perf-experiments.md`](docs/perf-experiments.md) Experiment 6
  *Follow-up*.

### Changed

- **CLI entry-point layout.** The `anamnesis` and `amn` binaries no
  longer share a single `src/bin/main.rs` (which produced a Cargo
  *"file found to be present in multiple build targets"* warning on
  every invocation because Cargo treats each `[[bin]]` as a separate
  crate root). The CLI implementation now lives in
  [`anamnesis::cli`](src/cli.rs) (feature-gated behind `cli`); each
  binary is a 5-line wrapper that delegates to `anamnesis::cli::run()`.
  No user-visible change: `cargo install anamnesis --features cli,...`
  still installs both `anamnesis` and `amn` with identical behaviour.
  Internal benefits: the warning is gone, the CLI code compiles once
  (not twice), and link time is lower.

## [0.4.5] - 2026-05-04

### Added

- **`inspect_gguf_from_reader<R: Read + Seek>`** — reader-generic `GGUF`
  inspection. Accepts any `Read + Seek` substrate (in-memory `Cursor`,
  HTTP-range-backed adapter, custom transport) and returns the same
  `GgufInspectInfo` (version, architecture, tensor count, total bytes,
  dtypes, alignment) as the existing path-based `parse_gguf(path).inspect()`,
  without materialising the data segment. The path-based `parse_gguf`
  remains available for callers that need the full `ParsedGguf` with
  zero-copy tensor views via `memmap2::Mmap`. A 2 GiB quantised `GGUF`'s
  metadata is inspectable in two or three small range requests covering a
  few MiB of front-loaded header — no weight data downloaded. Unblocks
  `hf-fm` v0.11.2 (remote `GGUF` inspect) on top of v0.11.0's
  `HttpRangeReader` adapter. Anamnesis itself takes on no network or TLS
  dependency. Phase 4.9; see [`ROADMAP.md`](ROADMAP.md).
- **`parse_gguf` and `inspect_gguf_from_reader` re-exported from the crate
  root** — both functions are now reachable as `anamnesis::parse_gguf` and
  `anamnesis::inspect_gguf_from_reader` (the former was already re-exported;
  the latter is new). Joins the existing reader-generic family
  (`parse_safetensors_header_from_reader`, `inspect_npz_from_reader`).
- **4 new GGUF unit tests** covering the reader-generic path:
  `inspect_from_reader_matches_path_minimal` (substrate-equivalence on the
  two-tensor F32+Q4_0 fixture), `inspect_from_reader_matches_path_mixed_dtypes`
  (three tensors with F32 and Q4_0 plus a nested-array metadata value
  exercising the seek-back-and-forth substrate), `inspect_from_reader_accepts_header_only_file`
  (24-byte header-only file: tensor-data section absent), and
  `inspect_from_reader_propagates_parse_errors` (truncated file, wrong
  magic, legacy `GGML` magic — all surface through the same code path as
  the file-backed parser).
- **Substrate-equivalence integration test on real GGUF files**
  (`tests/cross_validation_gguf.rs::substrate_equivalence_real_gguf_models`,
  `#[ignore]`-gated because the model directory is local-only). Walks
  every `*.gguf` under `tests/fixtures/gguf_reference/models/` (downloaded
  via `generate_gguf.py`) and asserts that
  `inspect_gguf_from_reader(File::open(path)?)` returns the same
  `GgufInspectInfo` as `parse_gguf(path).inspect()` field-for-field. Run
  on demand with `cargo test --release --features gguf --test
  cross_validation_gguf substrate_equivalence_real_gguf_models --
  --nocapture --ignored`. Locally-confirmed substrate-equivalent on **17
  of 17 files** spanning 4 architectures (`llama`, `qwen2`) × 11 distinct
  dtypes (`F32`, `F16`, `Q2_K`–`Q8_0`, `Q4_K`–`Q6_K`, `IQ1_S`, `IQ1_M`,
  `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ4_XS`) × 87 MiB to 2.7 GiB
  file sizes, including `bartowski/SmolLM2-135M-Instruct` (8 quants),
  `bartowski/Mistral-7B-Instruct-v0.3` (5 quants),
  `bartowski/Qwen2.5-{0.5,1.5}B-Instruct-IQ2_M`, and
  `TheBloke/TinyLlama-1.1B-chat-v1.0` (Q2_K, Q5_0).

### Performance

- **`inspect_gguf_from_reader` ~52× faster on `File` substrate (mmap parity)**.
  The user-supplied reader is now wrapped internally in
  `std::io::BufReader<R>` with a 64 KiB buffer, collapsing the parser's
  many small `read_exact` calls (4–8 B per typed primitive, variable per
  `gguf_string_t`) from one syscall each into one syscall per
  buffer-fill. Per-file medians (best-of-5, release, `target-cpu=native`,
  17 real GGUFs spanning Mistral-7B / Qwen2.5 / SmolLM2 / TinyLlama):
  Mistral-7B-IQ3_XXS 213 ms → 2.8 ms (**75×**), SmolLM2-135M-Q4_K_M
  399 ms → 7.6 ms (**53×**), Qwen2.5-1.5B-IQ2_M 1.23 s → 25.4 ms
  (**48×**), TinyLlama-Q5_0 438 ms → 7.1 ms (**61×**). Aggregate
  reader/mmap ratio collapsed from median 51.7× / mean 56.6× (slower)
  to median 1.0× / mean 1.0× (parity, occasionally slightly **faster**
  than mmap because BufReader does one syscall per 64 KiB while mmap
  incurs one minor page fault per 4 KiB page touched on the front
  matter). API unchanged; `R: Read + Seek` is still the bound.
  Substrate-equivalence on 17/17 real GGUFs preserved. New
  `tests/bench_gguf_inspect_adhoc.rs` ad-hoc benchmark
  (`#[ignore]`-gated) lands the regression-detection harness. See
  [`docs/perf-experiments.md`](docs/perf-experiments.md) Experiment 5
  for the full method, fixture list, per-file numbers, and trade-off
  rationale.

### Changed

- **GGUF parser cursor generalised to `Read + Seek`** — the slice-based
  internal cursor (`Cursor<'a>` over `memmap2::Mmap`) is replaced by a
  `GgufReader<R: Read + Seek>` that runs over any positional substrate.
  The path-based `parse_gguf` becomes a thin wrapper that mmaps the file
  and delegates to a new `parse_gguf_from_reader` core via `std::io::Cursor`.
  The `ParsedGguf` zero-copy tensor-data contract and every adversarial-
  input guard (caps on tensor count, KV count, string length, array length,
  nesting depth, dimension count, element product, per-tensor alignment,
  end-of-data bounds) are preserved verbatim. Behaviour-preserving refactor
  on the path-based entry point.
- **Crate-level rustdoc and README** updated to surface the path-based and
  reader-generic forms of `GGUF` inspection in a single place. New "GGUF
  Inspection" section in the README mirrors the existing "Safetensors
  Header Inspection" and "NPZ/NPY Parsing" entries; the table of contents
  links to it.

## [0.4.4] - 2026-05-02

### Added

- **`parse_safetensors_header_from_reader<R: Read>`** — reader-generic
  safetensors header parsing. Accepts any `Read` substrate (in-memory
  `Cursor`, `HTTP`-range-backed adapter, custom transport) and returns the
  same `SafetensorsHeader` as the existing slice-based
  `parse_safetensors_header`. Reads only the 8-byte little-endian length
  prefix plus the `JSON` header — total transfer ≈ header size (~1 MiB on
  a multi-GB shard) instead of the full file. The trait bound is `R: Read`
  rather than `R: Read + Seek` because the safetensors layout is purely
  prefix-then-`JSON`: two contiguous reads in order, never seek-back —
  keeping the simplest possible HTTP-range adapter (one connection, two
  range fetches) viable. A 100 MiB sanity cap on the declared header
  length bounds the worst-case allocation an adversarial source can
  trigger. Unblocks `hf-fm` v0.11.1, retiring `hf-fm`'s bespoke
  `fetch_header_bytes` parser and removing the only remaining duplicated
  format-knowledge between the two crates. Anamnesis itself takes on no
  network or TLS dependency. Phase 4.8; see [`ROADMAP.md`](ROADMAP.md).
- **`parse_safetensors_header` and `parse_safetensors_header_from_reader`
  re-exported from the crate root** — both functions are now reachable as
  `anamnesis::parse_safetensors_header` and
  `anamnesis::parse_safetensors_header_from_reader`, joining the existing
  path-based `anamnesis::parse`. The slice-based variant was previously
  only reachable via the longer `anamnesis::parse::safetensors::…` path.
- **5 new safetensors unit tests** covering the reader-generic path:
  `parse_from_reader_matches_slice_minimal` (substrate-equivalence on a
  single-tensor `BF16` buffer), `parse_from_reader_matches_slice_fp8_with_scale`
  (FP8 + `_scale_inv` scheme detection), `parse_from_reader_rejects_truncated_prefix`
  (`<8`-byte readers surface as `Io`), `parse_from_reader_rejects_oversized_header_length`
  (declared length above the 100 MiB cap surfaces as `Parse` without
  attempting allocation), and `parse_from_reader_rejects_truncated_json_tail`
  (partial-fetch in the JSON tail surfaces as `Io`).
- **`tests/fixtures/safetensors_reference/` cross-validation harness** —
  4 small `.safetensors` fixtures (`fp8`, `gptq`, `awq`, `bnb_nf4`,
  ~340 B–2 KiB each) plus a sibling `<scheme>.expected.json` reference
  for each, recording exactly what the upstream HuggingFace
  `safetensors` Python library reports about that file's header. The
  references are produced by `generate.py`, which sources the metadata
  two ways (raw 8-byte length prefix + `JSON` parse per spec, then
  cross-checked against `safetensors.safe_open`) before serialising the
  result. Five new integration tests in `tests/cross_validation_safetensors.rs`
  (one per scheme + one on-disk `File` reader cover) assert that **both**
  anamnesis entry points (slice-based `parse_safetensors_header` and
  reader-based `parse_safetensors_header_from_reader`) reproduce the
  Python-sourced reference field-for-field — true cross-validation
  against an external oracle, not anamnesis-against-anamnesis. Total
  fixture footprint ~5 KiB, all `include_bytes!`/`include_str!`-baked
  into the test binary.

### Changed

- **Crate-level rustdoc and README** updated to surface the three forms
  of header-only safetensors parsing (path-based, slice-based,
  reader-generic) in a single place. New "Safetensors Header Inspection"
  section in the README mirrors the existing NPZ entry; the table of
  contents links to it.

## [0.4.3] - 2026-05-01

### Added

- **`inspect_npz_from_reader<R: Read + Seek>`** — reader-generic `NPZ`
  inspection. Accepts any `Read + Seek` substrate (in-memory `Cursor`,
  HTTP-range-backed adapter, custom transport) and returns the same
  `NpzInspectInfo` as the existing path-based `inspect_npz`. The legacy
  `inspect_npz(path)` entry point is now a two-line wrapper that opens a
  `std::fs::File` and delegates here — fully backward-compatible. Unblocks
  remote `NPZ` inspection without materialising the data segment: a
  downstream HTTP-range adapter (e.g., `hf-fm`'s safetensors range-reader
  extended to `NPZ`) can satisfy this function in ~7 small range requests
  totalling well under 100 KiB on a typical Gemma Scope `params.npz` —
  cutting candle-mi's GemmaScope `open()` cold-start from ~30 s on a
  100 Mbps link to <1 s. Anamnesis itself takes on no network or TLS
  dependency. Phase 4.7; see [`ROADMAP.md`](ROADMAP.md).
- **3 new `NPZ` unit tests** covering the reader-generic path:
  `inspect_from_reader_matches_path` (substrate-equivalence on a
  multi-array in-memory archive), `inspect_from_reader_empty_archive`,
  and `inspect_from_reader_rejects_fortran_order` — confirming the
  refactor preserves every guard. Plus a new `cross_validation_npz`
  integration test (`inspect_path_and_reader_agree_on_gemma_scope_fixture`)
  asserting field-for-field parity between `inspect_npz` and
  `inspect_npz_from_reader` on the real Gemma Scope SAE fixture.

### Changed

- **`parse()` now memory-maps the safetensors file** instead of reading
  it into a `Vec<u8>` via `std::fs::read`. The semantic surface is
  unchanged: `ParsedModel::inspect`, `ParsedModel::remember`, and the
  internal `tensor_data` accessor all serve bytes through `&[u8]`
  slices, and `memmap2::Mmap` derefs to `[u8]`. **Measured impact on a
  locally-cached 11.6 GiB single-file safetensors shard
  (`bigcode/starcoder2-3b/model.safetensors`):**
  - `parse()` median: **2881.93 ms → 0.89 ms** (best-of-5 release mode,
    warm FS cache, range 2787-2887 ms before vs 0.86-0.91 ms after).
  - `parse()` + `inspect()` median: **2715.84 ms → 0.94 ms**.
  - Bench command: `cargo test --release --test bench_parse_adhoc bench_parse_safetensors_large -- --nocapture --ignored`.
  - Why: `fs::read` is `O(file_size)` `memcpy` from the OS file cache
    to a freshly-allocated `Vec<u8>`. `mmap` is constant-time virtual
    address setup; pages fault in on access. For inspect-only
    workflows the resident-set growth is bounded by the header (~1 MiB),
    not the file size. For full `remember()` workflows the kernel can
    drop file-backed pages under memory pressure (whereas `Vec<u8>`
    pages can't, they need swap), giving OOM-resilience to large-file
    dequantisation on memory-constrained machines. Identified by the
    v0.4.x algorithmic-weakness audit (finding #2 of 12). Full
    measurement record and analysis in
    [`docs/perf-experiments.md`](docs/perf-experiments.md) entry #4.
- **`memmap2` is now a mandatory dependency** (was optional, gated by
  `pth` and `gguf`). The `pth` and `gguf` features no longer reference
  `memmap2` in their feature lists. The `pth` feature now activates
  only `dep:zip`; `gguf` activates nothing extra.
- **Rustdoc refresh post-mmap** — module-level docs and `ParsedModel`
  / `parse()` / `tensor_data` / `remember()` rustdoc updated to
  reflect the mmap-based parse path. The `remember()` `# Memory`
  section now correctly states peak heap is
  `O(total_dequantised_output_size)` ≈ `2 × n_parameters` bytes (the
  input file is mmap-backed and does not contribute to heap), matching
  the README's framing.
- **`ROADMAP.md`** — inserted Phase 4.7 (Remote-only NPZ inspection,
  Reader-generic API) between Phase 4.5 and Phase 5; status header
  `Next:` pointer updated; the "Remote-only NPZ inspection
  (HTTP-range probe)" out-of-scope bullet from Phase 4.5 rehomed as a
  one-line pointer to its now-scheduled milestone. Phase 4.7's three
  implementation steps marked complete with commit hash citation.
- **`ROADMAP.md`** — added Phase 7.5 (Lethe Encode Completion, v0.7.5)
  and applied a consistency pass: refreshed header status, added
  Phase 7.5 + Phase 10 to the TOC, populated the `lethe/` module box,
  scoped Phase 6 / Phase 7 claims to actual v0.6.0 / v0.7.0 reality
  (BnB-only encode at v0.6.0, full encode matrix at v0.7.5), corrected
  the "v0.4.0 BnB decode kernels" reference in Phase 5 step 1, removed
  the v0.4.2 "awaiting user review" note now that the tag has shipped,
  and added a Phase 9 follow-up note covering Phase 5/7.5 encode-side
  pass-2 loops.
- **Performance-discipline infrastructure** — new `Performance Changes`
  section in [`CLAUDE.md`](CLAUDE.md) requires any perf-claim commit to
  ship a measurement (best-of-5 release-mode median on a real fixture,
  before/after numbers in the commit message). New
  [`docs/perf-experiments.md`](docs/perf-experiments.md) catalogues
  hypotheses tested — the v0.4.3 cycle's revert experiences (NPZ memset
  elimination measured −33 % regression; FP8 per-tensor chunked extend
  measured −23 % regression; v0.4.0 GGUF refactor re-validation
  surfacing a Q4_0 ~8 % win and a Q8_0 ~6 % regression that the
  original CHANGELOG had described as a uniform 10–15 % win). New
  ad-hoc bench harnesses
  [`tests/bench_dequant_adhoc.rs`](tests/bench_dequant_adhoc.rs) and
  [`tests/bench_parse_adhoc.rs`](tests/bench_parse_adhoc.rs)
  (both `#[ignore]`-gated) provide the measurement substrate.

### Fixed

- **`TensorEntry::num_elements` overflow saturation** — replaced the
  unguarded `shape.iter().product()` with `try_fold(checked_mul)`
  saturating to `usize::MAX`, matching the contract `inspect_npz`
  already documents. On a malformed or adversarial header that
  declares a shape whose element count overflows `usize` (e.g.,
  `[u32::MAX, 2]` on a 32-bit target), the previous implementation
  would silently wrap to a small value, after which downstream
  validation could accept a tiny data slice as if it described the
  full tensor. Public-API contract is unchanged: still
  `pub fn num_elements(&self) -> usize`. **3 new unit tests** cover
  the saturation, the exact path on normal shapes, and the empty-shape
  scalar case (product is 1, not 0). Identified by the v0.4.x
  algorithmic-weakness audit (finding #12 of 12).
- **GGUF CLI `remember` `n_elements` overflow guard** — the
  `tensor.shape.iter().product()` in `src/bin/main.rs` now uses
  `try_fold(checked_mul)` returning `AnamnesisError::Parse` on
  overflow, naming the offending tensor and shape. Previously a
  malformed GGUF tensor entry could silently wrap `n_elements` to a
  small value and dequantize a fraction of the data with no error.
  No public-API surface (CLI binary only).

## [0.4.2] - 2026-04-25

### Added

- **`GGUF` `MXFP4` dequant kernel** — the last GGUF block-quant type, closing Phase 4.5. 32-element block, 17 B/block: `e: u8` (E8M0 byte exponent) + `qs[16]` (4-bit packed nibbles). Decodes via a 16-entry signed `i8` codebook (`K_VALUES_MXFP4`, storing 2× the OCP E2M1 magnitudes) and a 1-byte E8M0 exponent decoded by the new `e8m0_to_fp32_half` helper — the doubled codebook plus the half-scale exponent cancel out so the dequantised value matches the raw OCP MX spec. Same low/high split-nibble layout as `Q4_0` / `IQ4_NL`. Bit-exact (0 ULP) against the `gguf` Python reference on a deterministic synthetic fixture (Python `gguf.quants.quantize()` supports MXFP4 too — same synthetic-fixture path as TQ1_0/TQ2_0 from step 5; mainstream MXFP4 GGUFs only ship inside the 11 GB `gpt-oss-20b` upload, too large to justify when synthetic is bit-exact). After this step **anamnesis dequantises every GGUF block type shipping on `HuggingFace` today** — 22 of 22 production kernels, no remaining coverage gap. Sixth and final step of Phase 4.5.
- **`GGUF` `TQ1_0` + `TQ2_0` dequant kernels** — two ternary super-quants invented for BitNet-style 1.58-bit models. `TQ1_0` (~1.6875 bpw, 54 B/block) uses base-3 packing (5 ternaries per byte for `qs`, 4 for `qh`) decoded via the `pow3 = [1, 3, 9, 27, 81, 243]` multiplication trick: after `byte * pow3[n]` (wrapping `u8`), the n-th digit lives in the top bits and is recovered by `(q * 3) >> 8`. `TQ2_0` (~2.0625 bpw, 66 B/block) uses plain 2-bit packing — 4 ternaries per byte. Both produce values in `{-d, 0, +d}` (true ternary). New shared helper `decode_pow3_ternary` alongside the existing `write_signed_grid` / `write_delta_grid` family. Bit-exact (0 ULP) against the `gguf` Python reference on a deterministic synthetic fixture (no model download needed — only ~15 BitNet-derivative uploads exist on HuggingFace, so synthetic input via Python `gguf.quants.quantize()` is the practical path). Fifth step of Phase 4.5.
- **`GGUF` `IQ1_S` + `IQ1_M` dequant kernels** — two 1-bit super-quants, the smallest IQ-family members. `IQ1_S` (~1.56 bpw, 50 B/block, 11-bit grid index via `qs` + `qh`, per-sub-block 3-bit scale + ±`IQ1S_DELTA = 0.125` additive bias selected by the top bit of `qh`); `IQ1_M` (~1.75 bpw, 56 B/block, **no top-level `d` field** — the super-block float scale is reconstructed from a scattered 16-bit pattern across `scales[8]` reinterpreted as `f16` via `half::f16::from_bits`). Both share the 2048-entry `IQ1S_GRID: [u64; 2048]` codebook of signed `i8` 8-element vectors (~16 KB). Inner-loop math is `dl × (grid[j] + delta)` rather than the multiplicative-sign pattern of IQ2/IQ3 — needs the new `write_delta_grid` helper sitting next to `write_signed_grid`. Bit-exact (0 ULP) against `gguf` Python on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (`...-IQ1_S.gguf` and `...-IQ1_M.gguf` — the IQ1 variants don't share files like IQ3 did, so two downloads were needed). Fourth step of Phase 4.5 (GGUF completeness).
- **`GGUF` `IQ3_XXS` + `IQ3_S` dequant kernels** — two 3-bit super-quants. `IQ3_XXS` (~3.06 bpw, 98 B/block, grid-only `scales_and_signs` per sub-block like `IQ2_XXS`); `IQ3_S` (~3.44 bpw, 110 B/block, 9-bit grid index via `qs` + `qh`, inline sign bytes like `IQ2_S`, unusual odd-integer scale formula `d × (1 + 2·nibble)`). Codebook grids (`IQ3XXS_GRID: [u32; 256]`, `IQ3S_GRID: [u32; 512]`, ~3 KB total) ported verbatim from `ggml-common.h`. Both reuse the Phase 4.5 step 2 `write_signed_grid` helper unchanged — the combined-grid/signs packing format is shared across all sign-masked `IQ*` kernels. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (`IQ3_XXS` from the `IQ3_XXS.gguf` variant, `IQ3_S` from the already-local `IQ2_S.gguf` which happens to ship 37 `IQ3_S` tensors). Third step of Phase 4.5 (GGUF completeness).
- **`GGUF` `IQ2_XXS` + `IQ2_XS` + `IQ2_S` dequant kernels** — three 2-bit super-quants from the `IQ*` family, each using a per-sub-block lattice codebook of packed 8-element `u8` vectors with a 7- or 8-bit sign mask flipping individual element signs. `IQ2_XXS` (66 B/block, 8-bit grid index), `IQ2_XS` (74 B/block, 9-bit grid index), and `IQ2_S` (82 B/block, 10-bit grid index with the high 2 bits from a separate `qh` array + inline sign bytes). Codebook grids (`IQ2XXS_GRID: [u64; 256]`, `IQ2XS_GRID: [u64; 512]`, `IQ2S_GRID: [u64; 1024]`) and the `ksigns_iq2xs` / `kmask_iq2xs` sign tables are ported verbatim from `ggml-common.h` into a private `iq_grids` submodule. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/Mistral-7B-Instruct-v0.3-GGUF` (`IQ2_XXS` + `IQ2_XS`) and `bartowski/Qwen2.5-0.5B-Instruct-GGUF` IQ2_M mix (`IQ2_S`). Second step of Phase 4.5 (GGUF completeness).
- **`GGUF` `IQ4_NL` + `IQ4_XS` dequant kernels** — two more GGUF block types dequantised to `BF16`: `IQ4_NL` (32-element non-linear 4-bit, 18-byte blocks) and `IQ4_XS` (256-element non-linear 4-bit super-block, 136-byte blocks — the most widely used member of the `IQ*` family on HuggingFace). Both share the 16-entry `kvalues_iq4nl` codebook. Bit-exact (0 ULP) against the `gguf` Python reference on 65 536-element slices from `bartowski/SmolLM2-135M-Instruct-GGUF`. First step of Phase 4.5 (GGUF completeness).
- **`[package.metadata.docs.rs]`** — docs.rs now builds with `features = ["npz", "pth", "gguf", "awq", "gptq", "bnb"]`, exposing all feature-gated public API items on docs.rs.
- **`docs/formats/gemmascope.md`** — one-page reference for loading GemmaScope (Gemma 2 JumpReLU SAEs). Documents the two-repo split (metadata in `mntss/gemma-scope-transcoders`, weights in `google/gemma-scope-2b-pt-transcoders`), NPZ tensor layout (`W_enc` transpose, `threshold` for JumpReLU, no `W_skip`), and links to the canonical `circuit-tracer` Python loader. No new parser needed — loads via existing NPZ support.

### Changed

- **Peak-heap documentation** — clarified the `# Memory` rustdoc on [`ParsedModel::remember`](src/model.rs) and [`dequantize_gguf_to_bf16`](src/remember/gguf.rs) to accurately describe the orchestrator-level eager buffering: every dequantised tensor's `Vec<u8>` is retained simultaneously until `safetensors::serialize_to_file` returns. Earlier wording suggested individual frees during the loop, which is incorrect. Added matching **Limitations (peak heap)** note to the README's GGUF section with model-size guidance (≤7 B comfortable, 13 B tight, 70 B+ OOMs on 32 GB systems). The streaming-output milestone is now a concrete planned phase ([ROADMAP.md](ROADMAP.md) Phase 10) rather than a Future Directions bullet — the dequantisation kernels already provide streaming entry points; only the orchestrator-level wiring is missing. No code change to dequantisation behaviour.
- **ROADMAP follow-ups from external audit** — added two Phase 9 (CPU SIMD pass) bullets capturing audit-identified opportunities to verify and, if needed, refactor the [`copy_to_contiguous`](src/parse/pth.rs) coordinate-carry loop and the [`AWQ`](src/remember/awq.rs)/[`GPTQ`](src/remember/gptq.rs) four-way `chunks_exact_mut(2).zip(...)` chains. Both are deferred from v0.4.2 — they require `cargo-show-asm` evidence before any refactor, which is exactly Phase 9's scope.
- **GPTQ/AWQ lazy precomputation** — scale and zero-point arrays are now computed per-group on demand instead of precomputing the full `num_groups × out_features` grid upfront. Reduces intermediate memory from `O(num_groups × out_features)` to `O(out_features)` with no throughput regression.
- **BnB double-quant refactor** — extracted shared `dequantize_bnb4_core` accepting `&[f32]` absmax directly. The double-quant path no longer serializes `Vec<f32>` to `Vec<u8>` and back; eliminates one allocation and one copy loop.
- **GPTQ g_idx pre-validation** — `g_idx` entries are now validated against `num_groups` in a single pass before the hot loop, failing fast on corrupted files instead of mid-dequantization.
- **CLI stale-binary guard** — `binary_path()` in CLI integration tests now checks that the binary version matches `Cargo.toml` and panics with a diagnostic message if stale. Pre-commit checks updated to include `cargo build --features cli`.
- **`docs/formats/gemmascope.md`** — added "Where to find files on HuggingFace" section documenting the `google/gemma-scope-{size}-{tune}-{site}` slug convention (sizes 2b/9b/27b/2-270m/2-1b, pt vs it, hook sites res/att/mlp/transcoders) and notable community ports (mwhanna, EleutherAI, weijie210).

### Fixed

- **CLI feature-gate fallback defect** — `detect_format` in [src/bin/main.rs](src/bin/main.rs) now returns a `Result<Format>` and emits an `AnamnesisError::Unsupported` carrying a feature-flag hint when the input matches a format whose Cargo feature is not enabled in this build. Previously a `.pth` / `.npz` / `.gguf` file (or a `.bin` file with the corresponding magic) silently fell through to the safetensors parser when its feature was disabled, producing cryptic downstream errors like `HeaderTooLarge` instead of a useful "rebuild with `--features cli,<flag>`" message. This matters in practice because `cli = ["dep:clap"]` does **not** transitively activate `pth` / `npz` / `gguf` — `cargo install anamnesis --features cli` ships a safetensors-only CLI. The library API (`anamnesis::parse_pth` etc.) was already returning `Unsupported` properly; this fix brings the CLI's UX in line.
- **Rust 1.95 clippy lints** — fixed `clippy::unnecessary_trailing_comma` in `src/inspect.rs` and `clippy::map_unwrap_or` (with the `is_ok_and` suggestion) in `src/bin/main.rs`. Both lints landed/strengthened in Rust 1.95.0 (2026-04-14) and would have broken CI's `stable` matrix job under `#![deny(warnings)]`. No user-visible behavior change.

## [0.4.1] - 2026-04-13

### Added

- **`pth_to_safetensors_bytes`** — in-memory `.pth` → safetensors conversion
  returning `Vec<u8>` instead of writing to disk. Enables downstream crates
  (candle-mi) to load `.pth` files without a temp file round-trip via
  `VarBuilder::from_buffered_safetensors`.
- **`ParsedPth::to_safetensors_bytes`** — convenience method combining
  `tensors()` + `pth_to_safetensors_bytes()`.
- **`ParsedGguf::dequantize_tensor`** — convenience method that slices the
  mmap, infers element count from shape, and delegates to
  `dequantize_gguf_to_bf16`. Saves consumers the three-line boilerplate on
  every tensor iteration.
- **GGUF CLI subcommands** — `amn parse model.gguf`, `amn inspect model.gguf`,
  and `amn remember model.gguf --to bf16 -o out.safetensors` now work when
  built with `--features gguf`. Format detection by `.gguf` extension with
  `GGUF` magic fallback for `.bin` and unknown extensions. Quantized tensors
  are dequantized to `BF16`; non-quantized tensors (`F32`, `F16`, `BF16`,
  integer types) are passed through with their original dtype.

## [0.4.0] - 2026-04-12

### Added

- **`dequantize_gguf_blocks_to_bf16`** — new streaming public API that
  emits one block's worth of `BF16` bytes per call into a caller-supplied
  `FnMut(&[u8]) -> Result<()>` sink closure (64 B per call for legacy
  quants, 512 B for K-quants). Peak heap is O(one scratch + one block
  output) regardless of tensor size — around 1.5 KB — enabling
  dequantisation of 70 B-parameter models on modest-RAM machines by
  streaming directly to disk. The existing `dequantize_gguf_to_bf16`
  `Vec`-returning variant is now a thin convenience wrapper that sinks
  into `Vec::with_capacity`, so both entry points share the same
  validation and the same scalar kernels.
- **`GGUF` block-quant dequantisation to `BF16`** (`dequantize_gguf_to_bf16`,
  feature-gated behind `gguf`) — scalar reference kernels for all 12 block
  types covered by the parser: legacy `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`,
  `Q8_1` (32-element blocks) and K-quants `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`,
  `Q6_K`, `Q8_K` (256-element super-blocks). Formulas ported verbatim from
  `ggml-quants.c`'s `dequantize_row_*` reference functions, with
  block-at-a-time loop fission (packed-bit unpacking into an `[f32; QK]`
  stack scratch buffer, then a branch-free `f32 × scale → BF16` pass via
  the shared `f32_bits_to_bf16_bits` helper). `IQ*`/`TQ*`/`MXFP4` return
  `AnamnesisError::Unsupported` (deferred to a later phase). No new crate
  dependencies — reuses `half` and the existing `gguf` feature. Phase 4
  step 2 toward v0.4.0; bit-for-bit cross-validation against `llama.cpp`
  is Phase 4 step 4.

### Changed

- **GGUF dequant kernels refactored to close over a `FnMut` sink**
  instead of each returning an owned `Vec<u8>`. Per-type pass-1 unpacking
  is now a small closure fed to a generic `run_legacy_kernel` /
  `run_super_kernel` outer-loop helper that handles `chunks_exact(TS)`
  iteration, scratch-buffer management, and the shared pass-2 `BF16`
  writer. `Vec::with_capacity` + `extend_from_slice` replaces the old
  `vec![0u8; n_elements * 2]`, avoiding the zero-init memset (~10–15 %
  of dequant wall time on `Q8_0`/`Q4_0` saved on platforms without lazy
  zero pages). Per-block `.get_mut(range).ok_or_else(...)?` bounds
  checks are replaced with `chunks_exact_mut`-style iteration on the
  inner kernel runners — ~4 M branches removed per 1 M-block tensor.
- **Infallible byte readers**: `read_f16_le(&[u8], usize) -> Result<f32>`
  is replaced with `read_f16_bytes([u8; 2]) -> f32` (and analogously for
  `read_f32_bytes`), eliminating dead `Result` shuffling on every
  hot-loop call. Callers slice fixed-length arrays out of their
  already-validated block slices.
- **Output size overflow guard**: `dequantize_gguf_to_bf16` now checks
  `n_elements.checked_mul(2)` in its shared validation helper, turning
  what would have been a silent `Vec` allocation truncation on 32-bit
  targets with > 2 GiB of `BF16` output into a clean `AnamnesisError::Parse`.
- **`GGUF` file parser** (`parse_gguf`, feature-gated behind `gguf`) — lean
  in-house parser for `GGUF` v2 and v3 files. Reads header, metadata
  key-value pairs (all 13 value types including nested `ARRAY`), and tensor
  info table. Resolves absolute tensor-data offsets from the tensor-info
  table's relative offsets plus the effective `general.alignment` (default
  32 bytes). `ParsedGguf::tensors` returns zero-copy `Cow::Borrowed` slices
  into the memory-mapped file for every dtype with a known `type_size`
  (`F32`, `F16`, `BF16`, `F64`, `I8`–`I64`, `Q4_0`–`Q8_1`, `Q2_K`–`Q8_K`).
  `IQ*`/`TQ*`/`MXFP4` tensors are listed in `tensor_info()` with
  `byte_len = None` and will be sized when dequantisation lands. No new
  third-party crate — reuses the `memmap2` dependency already pulled in by
  the `pth` feature. First commit of Phase 4 toward v0.4.0.
- **`GgufMetadataArray`** — new `#[non_exhaustive]` enum holding natively
  typed arrays (`Vec<u8>`, `Vec<f32>`, `Vec<String>`, …). Replaces the old
  `GgufMetadataValue::Array(Vec<GgufMetadataValue>)` storage to eliminate
  the ~8× enum-discriminant bloat on homogeneous numeric metadata arrays.
- **`ParsedGguf::tensors` returns `impl Iterator<Item = GgufTensor<'_>>`**
  instead of `Result<Vec<GgufTensor<'_>>>` — zero heap allocation per call.
  `GgufTensor::{name, shape}` now borrow from the parsed handle as
  `&'a str` / `&'a [usize]` rather than cloning owned `String`/`Vec`.
  Worst-case allocation dropped from ~130 MB per call on a 1 M-tensor file
  to 0 MB. Callers needing random access should use `.collect::<Vec<_>>()`.
- **`GgufMetadataValue::Array` now holds `Box<GgufMetadataArray>`** (was
  `Vec<GgufMetadataValue>`). `GgufMetadataValue::as_array` now returns
  `Option<&GgufMetadataArray>`. As a side effect, `GgufMetadataValue`
  shrinks from 32 bytes to 24 bytes (25% reduction) across every metadata
  value, not just arrays, because the max-sized variant is now `String`.

### Security / Performance

- **Cap trust-the-header pre-allocation** at `PREALLOC_SOFT_CAP = 256`
  entries for every `Vec::with_capacity` / `HashMap::with_capacity` call
  keyed on a file-declared count (metadata kv count, tensor count,
  per-array length). Previously a ~40-byte adversarial header claiming
  1 M of each could force ~175 MB of eager heap allocation before a single
  entry was read (empirically measured: 114 MB `HashMap` + 61 MB
  `Vec<RawTensorInfo>`); the cap drops this to ~34 KB (5 000× reduction).
  An adversarial `ARRAY` header claiming 16 M `f32` elements forced
  ~488 MB of eager allocation; combined with the typed-array fix, this is
  now capped at ~8 KB (60 000× reduction). Legitimate files grow the
  containers geometrically and are unaffected.
- **`Cursor::read_string` validates UTF-8 on the borrowed mmap slice
  before copying** — an adversarial 16 MiB non-UTF-8 string now costs
  zero heap allocation on the rejection path (was: a full 16 MiB
  `to_vec()` followed by `String::from_utf8`).
- **`ParsedGguf::inspect` dedups distinct dtypes via a `[bool; 32]`
  bitmap** keyed on a dense `GgufType` discriminant, replacing
  `Vec::contains` in the per-tensor loop. Drops the dtype-dedup hot path
  from O(n × d) to O(n) — ~10 ms → ~1 ms on a 1 M-tensor inspect call.
  First-occurrence order of `GgufInspectInfo::dtypes` is preserved.
- **`parse_gguf` builds `tensor_infos` in a single pass** instead of
  first materialising a throwaway `Vec<RawTensorInfo>` and then iterating
  it. The relative tensor-data offset is stored in `data_offset` during
  the read pass and patched to the absolute offset in a short sweep once
  `data_section_start` is known. Peak tensor-info heap on a 1 M-tensor
  file drops by ~60 MB; `RawTensorInfo` and `read_raw_tensor_info` are
  deleted.

### Fixed

- **GGUF dequantization cross-validated against `llama.cpp` reference**
  (Phase 4 step 4). 10 of 12 production kernels bit-exact (0 ULP) against
  the `gguf` Python package's `dequantize` function (the official
  `ggml-org` reference mirroring `ggml-quants.c`). Legacy quants: `Q4_0`,
  `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`. K-quants: `Q2_K`, `Q3_K`, `Q4_K`,
  `Q5_K`, `Q6_K`. Fixtures from 3 real models: bartowski SmolLM2-135M-
  Instruct and TheBloke TinyLlama-1.1B-Chat. `Q8_1` and `Q8_K` are not
  shipped by any real model (internal `llama.cpp` activation quant types)
  and are already covered by unit tests.

- **`parse_gguf` accepted tensors whose relative offset was not a
  multiple of `general.alignment`** (Phase 4 audit I1). The GGUF spec
  mandates that every tensor's offset field is a multiple of the file's
  declared alignment, but the patch sweep only checked the upper bound
  of each tensor's byte range. A malformed file encoding
  `relative_offset = 1` for every tensor would parse successfully and
  hand out unaligned byte slices through `ParsedGguf::tensors`, which
  downstream SIMD dequant kernels would then reinterpret as `f32`/`f16`
  words — unaligned access is undefined behaviour in the `unsafe`
  intrinsics planned for Phase 9. `parse_gguf` now rejects such files
  with `AnamnesisError::Parse` naming the offending tensor.

## [0.3.2] - 2026-04-05

### Fixed

- **`copy_to_contiguous` silent data corruption on mismatched shape/strides** (NI1) —
  added ndim guard rejecting `shape.len() != strides.len()`. Previously, `.zip()`
  silently truncated to the shorter iterator, producing corrupted output. Defence-in-depth
  check also added in `parse_rebuild_args`
- **`copy_to_contiguous` inner loop used `.get()` despite `// INDEX:` annotation** (NI2) —
  switched to direct indexing `storage[range]`, matching CONVENTIONS.md and the
  pre-validation that proves bounds safety. Eliminates dead `.ok_or_else()` branches
- **NPZ `extract_descr` mixed-quote bug** — quote character detection now reads the
  first quote in the value portion, not the entire header tail. Fixes silent
  mis-extraction for mixed-quote headers like `{'descr': "<f4", ...}`
- **`parse_pth` stale return-type doc** (D1) — claimed "Returns `Vec<PthTensor>`" but
  actually returns `Result<ParsedPth>`. Updated to reference `ParsedPth` and
  `ParsedPth::tensors()`
- **`inspect_npz` overflow saturation undocumented** (D2) — added note explaining
  `byte_len` saturates to `usize::MAX` on shape overflow (best-effort metadata),
  unlike `parse_npz` which returns `Err`
- **Misused `// EXPLICIT:` in `build_entry_index`** (D3) — was a char-boundary
  assertion, not a no-op arm or stateful loop. Downgraded to plain comment
- **Per-line `// BORROW:` annotations** — replaced single block-level annotation in
  `execute()` with per-call annotations on all 12 `.to_owned()`/`.to_vec()` sites.
  Added missing annotations in `build_entry_index`
- **`// VECTORIZED:` on `copy_to_contiguous`** — added `scalar fallback` annotation
  documenting why the inner loop cannot auto-vectorize (cross-iteration coords state)
- **`# Memory` section on `ParsedPth::tensors`** — documents zero-copy vs owned
  allocation paths
- **`const fn` on `ParsedPth::len`/`is_empty`** — `Vec::len()`/`is_empty()` are
  `const fn` since Rust 1.39
- **`# Errors` sections** on `parse_rebuild_args`, `build_entry_index`, and
  `copy_to_contiguous` — consistency with other private fallible functions
- **`lib.rs` architecture doc** (D6) — added `pth_to_safetensors()` to bullet list
- **`NpzDtype` `Display` doc** (D7) — documented as canonical uppercase string used
  in inspection output and cross-validation tests
- **`parse/mod.rs` stale docstring** — updated from "wraps `npyz`" to reflect own parser
- **`byteswap_inplace` missing `// VECTORIZED:` annotation** — added per CONVENTIONS.md
- **NPZ annotations** — `// EXPLICIT:` for `=` native-endian prefix, `// EXPLICIT:`
  for `extract_fortran_order` default
- **`bench_npz_adhoc.rs` hardcoded path** — replaced with `dirs::home_dir()` fallback

### Added

- **48 new unit tests** covering code review findings G1–G36, NI1–NI2, NN1–NN4:
  pickle VM opcodes (FRAME, NONE, NEWTRUE/NEWFALSE, BININT, BININT2, BINUNICODE,
  SHORT_BINSTRING, BINSTRING, SHORT_BINBYTES, BINBYTES, EMPTY_LIST, EMPTY_TUPLE,
  TUPLE1, TUPLE3, SETITEMS, APPEND, APPENDS, STACK_GLOBAL, REDUCE, NEWOBJ, BUILD,
  BINPERSID, LONG_BINPUT/LONG_BINGET, MEMOIZE), `long1_to_i64` 8-byte boundary,
  `MEMOIZE` overflow at `u32::MAX`, `MAX_PICKLE_NESTING` enforcement (both
  `unwrap_to_rebuild` and `extract_dict_pairs`), `copy_to_contiguous` (transposed,
  zero-element, overflow, zero-stride broadcast, ndim mismatch, storage boundary),
  missing/compressed `data.pkl` ZIP entries, zero-length ZIP entry,
  NPZ Fortran-order end-to-end rejection, empty NPZ archive (parse + inspect),
  native-endian `=` prefix, big-endian through `parse_npz`, `inspect_npz` overflow

## [0.3.1] - 2026-04-02

### Added

- **PyTorch `.pth` parsing** (`src/parse/pth.rs`) — minimal pickle
  interpreter (~36 opcodes) that parses `PyTorch` ≥ 1.6 `state_dict` ZIP
  archives with a safe, explicit `GLOBAL` allowlist (rejects non-`torch.*`
  callables — equivalent to `weights_only=True` but stricter). Zero-copy
  I/O via `memmap2` with `Cow::Borrowed` tensor data sliced directly from
  the mmap. Handles shared storage, non-contiguous strides, big-endian
  byte order, and both newer (`archive/`) and older (`{model_name}/`)
  `PyTorch` ZIP prefix conventions. Feature-gated behind `pth`. Supports
  `F16`, `BF16`, `F32`, `F64`, `I8`–`I64`, `U8`, `Bool` storage types.
  **11–31× faster** than `torch.load()` on torchvision models (resnet18,
  resnet50, ViT-B/16)
- **`.pth` → safetensors conversion** (`src/remember/pth.rs`) — lossless
  format conversion preserving original dtypes (no dequantization). The
  conversion pipeline writes directly from mmap slices to the output file
  — zero intermediate data copies. Byte-exact roundtrip verified against
  `PyTorch` reference on all 3 test models
- **`.pth` cross-validation** against `PyTorch` on 3 real
  [AlgZoo](https://github.com/alignment-research-center/alg-zoo) models
  (MIT-0 license): `2nd_argmax_2_2` RNN (10 params), `longest_cycle_2_3`
  Transformer (50 params), `one_layer_16_hidden` RNN blog example (432
  params). Byte-exact match on all tensors against `PyTorch` reference
- **CLI `.pth` support** — `amn parse`, `amn inspect`, and `amn remember`
  now accept `.pth`, `.pt`, and `.bin` files when built with
  `--features pth`. Format detection by extension with ZIP magic fallback
  for `.bin` files. `amn remember model.pth` converts to safetensors;
  `amn parse model.pth` shows per-tensor details (name, dtype, shape,
  size)
- **`ParsedPth`** container — owns the mmap, provides zero-copy
  `tensors()`, `inspect()` → `PthInspectInfo`, `tensor_info()` →
  `PthTensorInfo` (metadata only, no data access), and
  `to_safetensors()` convenience method
- **`PthInspectInfo`** — summary struct (tensor count, total bytes,
  dtypes, byte order) with `Display` impl
- **`PthTensorInfo`** — lightweight per-tensor metadata (name, shape,
  dtype, `byte_len`) for display paths that don't need tensor data
- **`PthDtype::to_safetensors_dtype()`** — direct single-hop conversion
  to `safetensors::Dtype`, bypassing the intermediate anamnesis `Dtype`
- **`inspect_npz()`** (`src/parse/npz.rs`) — header-only `NPZ` inspection
  that reads only `NPY` headers (~128 bytes per array), no tensor data.
  Returns `NpzInspectInfo` + `NpzTensorInfo` (name, shape, dtype,
  `byte_len`). For a 300 MB file, uses kilobytes instead of 300 MB
- **CLI `.npz` support** — `amn parse` and `amn inspect` now accept
  `.npz` files when built with `--features npz`, using the header-only
  `inspect_npz` path. `amn remember` for `.npz` returns a clear
  unsupported error (tensors are already full-precision)

### Changed

- Extracted `byteswap_inplace` from `src/parse/npz.rs` to shared
  `src/parse/utils.rs` module (`pub(crate)`) so that multiple format
  parsers (`NPZ`, `.pth`) can reuse it without duplication
- Widened `From<ZipError>` impl in `error.rs` from `npz`-only to
  `any(npz, pth)` feature gate
- Changed `unsafe_code` lint from `forbid` to `deny` to allow
  feature-gated `memmap2` usage in the `pth` module (with `// SAFETY:`
  annotation)

### Fixed

- **`has_zip_magic`** now reads only 4 bytes via `read_exact` instead of
  loading the entire file into heap (prevented 7 GB allocation on large
  `.bin` files)
- **`build_entry_index`** now returns `AnamnesisError::Parse` for corrupt
  ZIP entries whose data range exceeds the file size, instead of silently
  skipping them
- **`extract_dict_pairs`** unreachable `Reduced{OrderedDict}` branch now
  returns `Err` instead of `Ok(&[])`, preventing silent data loss
- **MEMOIZE** opcode uses `checked_add(1)` instead of plain `+= 1`,
  preventing silent `u32` wraparound on adversarial pickles
- **`build_entry_index`** `u64`→`usize` casts replaced with `TryFrom`,
  consistent with codebase conventions (no truncation on 32-bit)
- **`inspect()`** element count uses `saturating_mul` instead of
  `checked_mul().unwrap_or(0)`, avoiding silently wrong `total_bytes`
- **`--to`** argument for `.pth` files is now validated: accepts
  `safetensors` or `bf16`, errors on unsupported values
- **`copy_to_contiguous`** uses the two-level bounds pattern from
  `CONVENTIONS.md` — pre-validates max source offset once before the
  loop, removing 6 per-element `checked_*` calls

### New dependencies

- `memmap2` v0.9 (optional, `pth` feature only) — memory-mapped file I/O

## [0.3.0] - 2026-03-24

### Added

- **NPZ/NPY parsing** (`src/parse/npz.rs`) — `parse_npz(path)` reads `NumPy`
  `.npz` archives into framework-agnostic `NpzTensor` structs. Custom `NPY`
  header parser with bulk `read_exact` data extraction — zero per-element
  deserialization for LE data on LE machines. Feature-gated behind `npz`.
  Supports `F16`/`F32`/`F64`, all integer types, `Bool`, and `BF16` (JAX `V2`
  void dtype). **3,586 MB/s** on 302 MB Gemma Scope file (1.3× raw I/O
  overhead), **17.7× faster** than `npyz`-backed parser
- **NPZ cross-validation** against Gemma Scope 2B SAE weights (`params.npz`,
  5 `F32` arrays). Byte-exact match against `NumPy` reference on all arrays

### Fixed

- **BnB4 output shape recovery** — `NF4`/`FP4` dequantized weights now have their
  original 2D shape (e.g., `[2048, 8192]`) instead of flat `[total_elements]`.
  Shape is recovered from the `quant_state.bitsandbytes__nf4`/`__fp4` companion
  tensor's `JSON` blob, which is stored inside the safetensors file itself (no
  `config.json` needed). Falls back to 1D if the companion is absent
- **`extract_descr` mixed-quote header bug** — quote character detection now reads
  the first character of the value, not the entire header tail. Fixes silent
  mis-extraction for mixed-quote headers (e.g., `{'descr': "<f4", ...}`)
- Added mandatory `// VECTORIZED:`, `// EXPLICIT:` annotations on
  `byteswap_inplace`, `extract_fortran_order`, native-endian `=` treatment,
  and unused minor version byte
- Removed hardcoded absolute path from `bench_npz_adhoc.rs` — now resolves
  from `USERPROFILE`/`HOME` environment variables
- Added missing `[0.2.0]` changelog section for Phase 2 (GPTQ, AWQ, BnB)

## [0.2.0] - 2026-03-24

### Added

- **GPTQ dequantization** (`src/remember/gptq.rs`) — INT4 and INT8 with
  group-wise scale + zero-point, activation-order via `g_idx`. Feature-gated
  behind `gptq`. Bit-exact against PyTorch on 4 real models from 2 quantizers
  (AutoGPTQ, GPTQModel), **6.5–12.2× faster** than CPU PyTorch (AVX2)
- **GPTQ parsing layer** — `TensorRole::ZeroPoint` / `GroupIndex`,
  `QuantScheme::Gptq`, `GptqConfig` (bits + group_size inference from metadata
  or tensor shapes), `find_gptq_companions()`, `gptq` feature gate
- **GPTQ cross-validation** against PyTorch on 4 models: Falcon3-1B INT4/INT8
  (AutoGPTQ), Llama-3.2-1B INT4 (AutoGPTQ), Llama-3.2-1B INT8 (GPTQModel)
- **GPTQ inspect/CLI** — zero-point, group-index counts in `inspect` and
  `parse` output; format-aware size label (FP8/GPTQ/unquantized)
- **AWQ dequantization** (`src/remember/awq.rs`) — INT4 (and INT8 path,
  unit-tested) with per-group scales, no +1 zero-point offset. Feature-gated
  behind `awq`. Bit-exact against PyTorch on 2 real models (AutoAWQ GEMM),
  **4.7–5.7× faster** than CPU PyTorch (AVX2). Loop fission applied from the
  start; full AVX2 `vsubps`/`vmulps` ymm confirmed
- **AWQ parsing layer** — `QuantScheme::Awq`, `AwqConfig`, `AwqCompanions`,
  shape-based detection distinguishing AWQ (packed along cols) from GPTQ
  (packed along rows), `awq` feature gate
- **AWQ cross-validation** against PyTorch on 2 models: Llama-3.2-1B and
  Falcon3-1B (both AutoAWQ GEMM, 4-bit). Note: no real 8-bit AWQ models
  exist in the standard AutoAWQ `.qweight` format — all "8-bit AWQ" models
  on HuggingFace are either dequantized F16, `compressed-tensors` (vLLM), or
  mislabeled 4-bit
- **BitsAndBytes dequantization** (`src/remember/bnb.rs`) — NF4, FP4 (both
  4-bit lookup-table with per-block absmax), double-quant NF4/FP4 (nested
  absmax), and INT8 (`LLM.int8()` with per-row absmax). Feature-gated behind
  `bnb`. Bit-exact against PyTorch on 4 real models, **18–54× faster** for
  NF4/FP4 (AVX2), **1.2× faster** for INT8 (near memory bandwidth limit).
  Loop fission for NF4/FP4; single-pass AVX2 for INT8 (`vpmovsxbd` →
  `vcvtdq2ps` → `vmulps`)
- **BnB parsing layer** — `QuantScheme::Bnb4` / `BnbInt8`,
  `TensorRole::QuantMap` / `NestedScale`, `BnbConfig` (block_size,
  double_quant), `Bnb4Companions`, detection by `.weight.quant_map` (NF4/FP4)
  and `.SCB` (INT8) naming patterns, `bnb` feature gate
- **BnB cross-validation** against PyTorch on 4 models: Llama-3.2-1B NF4,
  Llama-3.2-1B NF4 double-quant, Llama-3.2-1B FP4, Llama-3.2-1B INT8
- **BnB model.rs integration** — `Bnb4` and `BnbInt8` arms in
  `remember_bf16_inner` with companion lookup, double-quant detection, and
  shape handling (flat output for NF4/FP4, preserved 2D for INT8)
- `FromStr` impl for `TargetDtype` — centralizes string-to-enum parsing so new
  variants cannot be silently missed in the CLI
- `ParsedModel::remember_with_progress()` — dequantize with a per-tensor
  callback, enabling progress reporting in CLI contexts
- `indicatif` progress bar during `remember`/`dequantize` when built with the
  `indicatif` feature (`amn remember` shows `[====================] 2.1s`)
- CONVENTIONS.md: two-level bounds checking pattern (reconciles `// INDEX:`
  safety with SIMD rule #2) and loop fission for mixed-domain pipelines

### Fixed

- Added unit tests for `dequantize_per_channel_fp8_to_bf16` covering F32, BF16,
  and F16 scale dtypes, single-row, NaN handling, and validation errors
- Added fine-grained dequantization tests for all three scale dtypes (F32, BF16,
  F16) and multi-block F32 scale path
- Added CLI integration tests (`tests/cli.rs`) — 9 tests covering `parse`,
  `inspect`/`info`, `remember`/`dequantize`, error handling, and `--version`
- Documented single-scheme assumption in `detect_scheme` (all quantized tensors
  in a file use the same scheme; early-return on first scale companion found)
- CLI `parse` subcommand now displays the actual scale dtype (BF16, F16, F32)
  instead of always printing "F32"
- `inspect` Display now shows the actual scale dtype instead of hardcoded "F32"
- `dequantize_per_tensor_fp8_to_bf16` now uses `checked_mul` for output size,
  consistent with the other two dequantize functions (**breaking**: returns
  `Result<Vec<u8>>` instead of `Vec<u8>`)
- Fine-grained dequantization now validates that the scale grid is rectangular
  (rejects `scale_elements % scale_rows != 0` instead of silently truncating)
- `serialize_to_file` I/O errors now surface as `AnamnesisError::Io` instead of
  being misclassified as `AnamnesisError::Parse`
- `derive_output_path` now matches `TargetDtype` exhaustively instead of using a
  wildcard that would silently produce broken paths for future variants
- Simplified `shape_to_rows_cols` 2D arm: direct indexing with `// INDEX:`
  annotation instead of redundant `Option` unwrapping
- **`classify_tensor` AWQ-only builds** — `.qweight`/`.qzeros`/`.scales` were
  gated on the `gptq` feature only; AWQ-only builds silently misclassified all
  quantized tensors as passthrough. Now gated on `any(gptq, awq)` with `.g_idx`
  remaining `gptq`-only
- **`detect_scheme` silent fallthrough** — `return` statements inside the
  GPTQ/AWQ detection block were feature-gated, causing misdetection when only
  one scheme was enabled. Detection is now unconditional; feature-disabled
  errors are handled downstream in `model.rs`
- `derive_output_path` now strips GPTQ, AWQ, and BitsAndBytes suffixes
  (e.g., `-GPTQ-Int4`, `-awq`, `-bnb-4bit`) in addition to FP8 suffixes
- `read_scale_f32` now uses `checked_add` for all byte offset computations,
  consistent with `read_u32_le` in the same codebase
- Extracted duplicated `read_u32_le` and `read_scale_f32` from `gptq.rs` and
  `awq.rs` into shared `remember/quant_utils.rs` module
- Replaced dead-code `checked_mul(1)` in `bnb.rs` with direct parity check
- GPTQ/AWQ outer loop offsets (`i * out_features * 2`) now use `checked_mul`
  for consistency with the codebase's zero-panic discipline
- `parse_g_idx` offset (`i * 4`) now uses `checked_mul`
- Updated GPTQ docstring memory estimate from "~1 MB" to "up to ~8 MB per
  weight tensor" to reflect fine-grained group configurations

## [0.1.0] - 2026-03-24

### Added

- **Safetensors parsing** (`src/parse/safetensors.rs`) — header parsing, tensor
  metadata extraction, dtype classification, tensor role classification (quantized,
  scale, passthrough), quantization scheme detection by scale tensor shape
- **Three FP8 dequantization schemes** (`src/remember/fp8.rs`):
  - **Fine-grained** — 128×128 block scale factors (`dequantize_fp8_to_bf16`)
  - **Per-tensor** — single scalar scale (`dequantize_per_tensor_fp8_to_bf16`)
  - **Per-channel** — one scale per output row (`dequantize_per_channel_fp8_to_bf16`)
- **Three scale dtypes** — `F32`, `BF16`, and `F16` scale tensors all supported
- **Branchless E4M3 → BF16 pipeline** — const subnormal lookup table, bitwise NaN
  select, round-to-nearest-even; auto-vectorized to SSE2 (default) and AVX2
  (`target-cpu=native`), verified with `cargo-show-asm`
- **Inspect module** (`src/inspect.rs`) — format, tensor counts, current/dequantized
  size estimates, Lethe distance
- **Parse-first public API** (`src/model.rs`) — `parse(path)` → `ParsedModel` →
  `.inspect()` / `.remember(path, target)`
- **CLI binary** (`src/bin/main.rs`) — subcommands: `parse`, `inspect`/`info`,
  `remember`/`dequantize`. Installed as both `anamnesis` and `amn`. Feature-gated
  behind `cli`
- **Cross-validation against PyTorch** — 7 fixtures from real models, bit-exact
  match (0 ULP on 65,536 elements each), 2.7–9.7× faster than PyTorch (AVX2)
- **Validated against 7 real models** from 5 quantization tools (LG AI, Qwen,
  Mistral, RedHat, NVIDIA)
