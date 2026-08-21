# anamnesis Coding Conventions (Grit + Grit-AMN Extensions)

This document describes the [Amphigraphic coding](https://github.com/PCfVW/Amphigraphic-Strict) conventions used in anamnesis. It is a superset of
the [Grit — Strict Rust for AI-Assisted Development](https://github.com/PCfVW/Amphigraphic-Strict/tree/master/Grit).

## Trigger Checklist

**Before writing any line of code, check which triggers apply.**

| You are about to... | Check these rules |
|---|---|
| Write a `///` or `//!` comment | [Backtick hygiene](#backtick-hygiene), [field-level docs](#field-level-docs), [intra-doc link safety](#intra-doc-link-safety) |
| Write a `pub fn` or `pub const fn` | [`const fn`](#const-fn), [`#[must_use]`](#must_use-policy), [pass by value](#pass-by-value-vs-reference) |
| Write a `pub fn` returning `Result<T>` | [`# Errors` section](#errors-doc-section) |
| Write a `pub fn` that processes large files | [`# Memory` section](#memory-doc-section) |
| Write a `pub enum` | [`#[non_exhaustive]`](#non_exhaustive-policy) or [`// EXHAUSTIVE:`](#exhaustive-annotation) |
| Write an `as` cast | [`// CAST:`](#cast-annotation) |
| Write `slice[i]` or `slice[a..b]` | [`// INDEX:`](#index-annotation) |
| Write `.as_str()`, `.to_owned()` | [`// BORROW:`](#borrow-annotation) |
| Write an `unsafe` block | [`// SAFETY:`](#safety-annotation), feature-gate check |
| Write `Box<dyn T>` or `&dyn T` | [`// TRAIT_OBJECT:`](#trait_object-annotation) |
| Write a `match` or `if let` | [Control-flow rules](#if-let-vs-match), [`// EXPLICIT:`](#explicit-annotation) if no-op arm |
| Write bit manipulation for (de)quantization | [`// BITWISE:`](#bitwise-annotation) |
| Write a bulk conversion loop | [`// VECTORIZED:`](#vectorized-annotation), [SIMD-friendly loop rules](#when-writing-simd-friendly-loops) |
| Spawn threads or write a parallel loop | [`// PARALLEL:`](#parallel-annotation), [parallelizing work rules](#when-parallelizing-work) |
| Write error strings | [Error message wording](#error-message-wording) |
| Batch operations by key | [HashMap grouping idiom](#hashmap-grouping-idiom) |
| Parse a header, archive, or stream from caller input | [Untrusted input invariants](#when-parsing-untrusted-input) |
| Add `#[allow(clippy::...)]` for a newer lint | [MSRV lint guard](#msrv-lint-guard) |
| Suppress a lint because its advice measured *slower* | [`// MEASURED-REVERT:`](#measured-revert-annotation) |
| Compare two versions of a kernel with `cargo bench` | [Benchmark evidence](#benchmark-evidence-what-a-criterion-baseline-can-and-cannot-show) |

---

## Annotation Grammar

The `// CAST:`, `// INDEX:`, `// BITWISE:`, `// VECTORIZED:`, `// BORROW:`,
`// TRAIT_OBJECT:`, `// SAFETY:`, `// EXPLICIT:`, and `// EXHAUSTIVE:`
comments below pair with `#[allow(clippy::…)]` attributes that suppress the
crate's [lint floor](#lint-floor). The comment explains *why*; the attribute
is what makes the code build under `#![deny(warnings)]`.

### Comment ↔ attribute pairing

| Annotation | Companion attribute(s) |
|---|---|
| `// CAST:` | `#[allow(clippy::as_conversions)]` plus, as the cast requires, `cast_precision_loss`, `cast_possible_truncation`, `cast_possible_wrap` |
| `// INDEX:` | `#[allow(clippy::indexing_slicing)]` |
| `// EXHAUSTIVE:` | `#[allow(clippy::wildcard_enum_match_arm)]` or `#[allow(clippy::exhaustive_enums)]` |
| `// SAFETY:` | `#[allow(unsafe_code)]` (at item or block scope) |
| `// MEASURED-REVERT:` | `#[allow(clippy::<the lint whose advice was measured slower>)]` |
| `// BITWISE:`, `// VECTORIZED:`, `// PARALLEL:`, `// BORROW:`, `// TRAIT_OBJECT:`, `// EXPLICIT:` | none — these are documentation-only |

### Per-site vs block-scope

Single suppressed line: attribute immediately above.

> ```rust
> // CAST: u32 → f32, qz is at most 255, exact in f32
> #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
> let val = qz as f32;
> ```

Tight loop where every iteration shares the same invariant: one block-scope
`#[allow]` plus one comment for the whole loop. If different lines need
different justifications, fall back to per-line.

> ```rust
> #[allow(clippy::indexing_slicing)]
> for (j, qw_chunk) in qw_row.chunks_exact(4).enumerate() {
>     // INDEX: chunks_exact(4) guarantees exactly 4 bytes; j < out_features
>     // bounded by qw_row length validation above
>     let packed = u32::from_le_bytes([qw_chunk[0], qw_chunk[1], qw_chunk[2], qw_chunk[3]]);
> }
> ```

### Lint Floor

The annotations exist because `Cargo.toml`'s `[lints.clippy]` block denies
the underlying lints crate-wide. The relevant settings:

| Lint | Level | How to satisfy |
|---|---|---|
| `unwrap_used`, `expect_used`, `panic` | `deny` | never use these in library code |
| `indexing_slicing` | `deny` | `// INDEX:` + `#[allow(clippy::indexing_slicing)]` |
| `wildcard_enum_match_arm` | `deny` | `// EXHAUSTIVE:` + `#[allow(clippy::wildcard_enum_match_arm)]` |
| `as_conversions` | `warn` | `// CAST:` + `#[allow(clippy::as_conversions, …)]` |
| `must_use_candidate` | `warn` | annotate the function with `#[must_use]` |
| `missing_errors_doc` | `warn` | write a `# Errors` doc section |
| `missing_panics_doc` | `warn` | panics are denied; if `#[allow(clippy::panic)]` is ever needed for a specific reason, write `# Panics` |
| `explicit_iter_loop`, `manual_filter_map`, `manual_find_map`, `needless_range_loop` | `warn` | use iterator methods (`.iter()`, `.filter_map()`, `.find_map()`) instead of indexed loops |
| `pedantic` (priority -1) | `warn` | per-lint allow with explanatory comment |

`#![deny(warnings)]` promotes every `warn` to a hard error — the discipline
is not optional. Keep the table in sync with `Cargo.toml`'s `[lints.clippy]`.

---

## When Writing Doc Comments (`///`, `//!`)

### Backtick Hygiene

All identifiers, types, trait names, field names, crate names, and
file-format names in doc comments must be wrapped in backticks so that
rustdoc renders them as inline code and Clippy's `doc_markdown` lint passes.

Applies to: struct/enum/field names, method names (`fn foo`), types
(`Vec<T>`, `Option<f32>`), crate names (`safetensors`, `half`),
file extensions (`.npy`, `.npz`, `.safetensors`), and acronyms that double
as types (`DType`, `NaN`, `FP8`, `E4M3`, `BF16`, `GPTQ`).

> ✅ `` /// Parses the header of a `.safetensors` file. ``
> ❌ `/// Parses the header of a .safetensors file.`

### Intra-Doc Link Safety

Rustdoc intra-doc links must resolve under all feature-flag combinations
(enforced by `#![deny(warnings)]` → `rustdoc::broken_intra_doc_links`).

Feature-gated items (e.g., NPZ types behind `npz` feature, GPTQ types
behind `gptq` feature) must use plain backtick text, not link syntax:

> ✅ `` /// See `NpzTensor` (requires `npz` feature). ``
> ❌ `` /// See [`NpzTensor`](crate::parse::npz::NpzTensor). ``

### Field-Level Docs

Every field of every `pub` struct must carry a `///` doc comment describing:
1. what the field represents,
2. its unit or valid range where applicable.

> Example:
> ```rust
> pub struct NpzTensor {
>     /// Tensor dimensions (e.g., `[2304, 2048]`).
>     pub shape: Vec<usize>,
>     /// Element data type (e.g., `F32`, `F64`, `BF16`).
>     pub dtype: NpzDtype,
>     /// Raw bytes in row-major order. Length = product(shape) * dtype.byte_size().
>     pub data: Vec<u8>,
> }
> ```

### `# Errors` Doc Section

All public fallible methods (`-> Result<T>`) must include an `# Errors` section.
Each bullet uses the format:

    /// # Errors
    /// Returns [`AnamnesisError::Parse`] if the safetensors header is malformed.
    /// Returns [`AnamnesisError::Io`] if the file cannot be read.

Rules:
- Start each bullet with `Returns` followed by the variant in rustdoc link
  syntax, e.g., `` [`AnamnesisError::Parse`] ``.
- Follow with `if` (condition), `on` (event), or `when` (circumstance).
- One bullet per distinct error path.

### `# Memory` Doc Section

Public methods that process large files (safetensors, NPZ archives) must include
a `# Memory` section documenting:

1. **Peak allocation** — how much memory the method allocates at its peak.
2. **Lifetime** — whether allocations are dropped before the method returns
   or persist in the returned value.

Format:

    /// # Memory
    /// Reads the entire safetensors file into memory (~2× model size during
    /// dequantization: source FP8 + destination BF16). The source buffer is
    /// dropped before returning.

---

## When Writing Function Signatures

### `const fn`

Declare a function `const fn` when **all** of the following hold:
1. The body contains no heap allocation, I/O, or `dyn` dispatch.
2. All called functions are themselves `const fn`.
3. There are no trait-method calls that are not yet `const`.

This applies to constructors, accessors, and pure arithmetic helpers.
When in doubt, annotate and let the compiler reject it — do not omit `const`
preemptively.

> ✅ `pub const fn byte_size(&self) -> usize { ... }`
> ❌ `pub fn byte_size(&self) -> usize { ... }`

### `#[must_use]` Policy

All public functions and methods that return a value and have no side effects
must be annotated `#[must_use]`. This includes constructors, accessors,
and pure queries (`byte_size`, `is_quantized`, `tensor_count`).

The `clippy::must_use_candidate` lint enforces this at `warn` level
(promoted to error by `#![deny(warnings)]`).

### Pass by Value vs Reference

Follow these rules for function parameters:

| Type | Rule |
|---|---|
| `Copy` type ≤ 2 words (`usize`, `f32`, `bool`, small `enum`) | Pass by value |
| `Copy` type > 2 words | Pass by reference |
| Non-`Copy`, not mutated | Pass by `&T` or `&[T]` |
| Non-`Copy`, mutated | Pass by `&mut T` |
| Owned, consumed by callee | Pass by value (move semantics) |
| `&mut T` not actually mutated in body | Change to `&T` |

Clippy enforces both directions: `needless_pass_by_ref_mut` flags `&mut T`
that is never written through; `trivially_copy_pass_by_ref` flags `&T` for
small `Copy` types.

---

## When Writing Public Enums

### `#[non_exhaustive]` Policy

- Public enums that may gain new variants: `#[non_exhaustive]`.
  `Dtype`, `QuantScheme`, `TargetDtype`, and `TensorRole` are all
  non-exhaustive — new formats and dtypes will be added over time.
- Internal dispatch enums matched exhaustively by this crate:
  `#[allow(clippy::exhaustive_enums)] // EXHAUSTIVE: <reason>`.

---

## When Writing Expressions

These annotations are required at the smallest scope that captures the
suppression — per-line for a single occurrence, block-scope for a tight
loop with a shared invariant (see [Annotation Grammar](#annotation-grammar)).
Apply them as you write the line, not in a review pass.

### CAST Annotation

`// CAST: <from> → <to>, <reason>` — required on every `as` cast between numeric types. Prefer `From`/`Into` for
lossless conversions and `TryFrom`/`TryInto` with `?` for fallible ones.
Use `as` only when truncation or wrapping is the deliberate intent, or when
the bit-level reinterpretation is the whole point (as in dequantization).
> Example: `// CAST: u8 → f32, FP8 E4M3 mantissa bits promoted for arithmetic`
> Example: `// CAST: usize → u64, byte offset for safetensors; file size fits in u64`

### INDEX Annotation

`// INDEX: <reason>` — required on every direct slice index (`slice[i]`,
`slice[a..b]`). Direct indexing panics on out-of-bounds; prefer `.get(i)?`
unless the bound is provably valid and indexing is significantly more
readable than the iterator idiom. For SIMD hot loops see
[reconciling bounds checking with vectorization](#reconciling-bounds-checking-with-vectorization).
> Example: `// INDEX: offset is bounded by header.data_offsets checked above`
> Example: `// INDEX: chunks_exact(4) guarantees exactly 4 bytes per chunk`

### BITWISE Annotation

`// BITWISE: <operation>` — required on every raw bit manipulation that implements part of a quantization
or dequantization algorithm. Document what the bits represent and what the
operation achieves — the mapping between storage format and numerical value
must be traceable through the annotations.
> Example: `// BITWISE: extract 3-bit mantissa from E4M3 byte (bits [2:0])`
> Example: `// BITWISE: combine sign, exponent, mantissa into IEEE 754 half-precision`
> Example: `// BITWISE: extract unsigned 4-bit value from packed I32 at bit position shift`

### VECTORIZED Annotation

`// VECTORIZED: <state>` — required on every hot-path conversion loop. Three
states (`confirmed`, `scalar fallback`, `pending`); see
[Verify vectorization](#verify-vectorization) for the full policy and the
evidence each state requires.
> Example: `// VECTORIZED: confirmed AVX2 vsubps + vmulps in cargo-show-asm, x86-64 target-cpu=native, opt-level=3`
> Example: `// VECTORIZED: scalar fallback — loop body contains branch that defeats auto-vectorization`
> Example: `// VECTORIZED: pending cargo-show-asm verification`

### MEASURED-REVERT Annotation

`// MEASURED-REVERT: <lint>, <measured cost>, <bench id>, <significance>` —
required whenever a lint is suppressed **because taking its advice was measured
to be slower**. Pairs with `#[allow(clippy::<lint>)]` at the smallest scope that
covers the loop.

This is not "I disagree with the lint", and it must not be written as though it
were. The annotation records an *experiment*: the suggestion was applied,
benchmarked against a saved baseline, and reverted on the numbers. A reader who
doubts it can re-run the same bench and get the same answer.

Required content, all four parts:

1. the lint name, so a future toolchain bump can find every site;
2. the measured delta **and its direction**;
3. the benchmark identifier and the fixture shape it ran on;
4. the significance, and how many independent runs reproduced it.

> Example (real, from the v0.7.6 evaluation):
> ```rust
> // MEASURED-REVERT: clippy::chunks_exact_to_as_chunks. Migrating this tile
> // loop to `as_chunks` cost +74 % on `dequant_awq_int4` (criterion,
> // 4096 × 11008, target-cpu=native, p = 0.00, reproduced across two runs).
> // Reverted under CLAUDE.md's rule that a hot path does not change without a
> // measured win.
> #[allow(clippy::chunks_exact_to_as_chunks)]
> ```

**When the lint is *inapplicable* rather than costly, say that instead.** That is
a different claim: it needs no measurement, and no future measurement can
invalidate it. Use a plain reason comment naming the compile constraint.

> Example: the `as_chunks` family takes its width as a *const generic argument*,
> so it cannot express one derived from an associated const of a type parameter:
> ```rust
> // GENERIC-CONST: `as_chunks_mut::<{VECTOR_TILE * E::BYTES}>()` is rejected by
> // stable Rust — "generic parameters may not be used in const operations".
> #[allow(clippy::chunks_exact_to_as_chunks)]
> ```

**A crate-wide `#![allow]` is never a `MEASURED-REVERT`.** Suppressing at the
crate root also hides the lint in code written *later*, which no measurement of
today's code can justify. That is a policy decision: it belongs in
`src/lib.rs` with its reasoning stated, it should say plainly that it hides
future occurrences, and it should be revisited rather than inherited.

### BORROW Annotation

`// BORROW: <what is converted>` — required on explicit `.as_str()`, `.as_bytes()`, `.to_owned()` conversions (Grit Rule 2).
> Example: `// BORROW: explicit .as_str() instead of Deref coercion`

### TRAIT_OBJECT Annotation

`// TRAIT_OBJECT: <reason>` — required on every `Box<dyn Trait>` or `&dyn Trait` usage.
> Example: `// TRAIT_OBJECT: heterogeneous format parsers require dynamic dispatch`

---

## When Writing `unsafe`

### SAFETY Annotation

`// SAFETY: <invariants>` — required on every `unsafe` block or function (inline comment, not a doc comment).

anamnesis is `#![deny(unsafe_code)]` at the crate root. `deny` (not `forbid`)
lets feature-gated modules opt in via a local `#[allow(unsafe_code)]` +
`// SAFETY:` comment. Current opt-ins are in the table below; SIMD intrinsics
(`#[target_feature(enable = "avx2")]`) are an anticipated future opt-in,
required by language rules even when the operations are safe in practice.
New opt-ins follow the candle-mi pattern:

| Feature | Accepted `unsafe` scope |
|---------|------------------------|
| *(always-on)* + `pth` + `gguf` | `memmap2::Mmap::map(&file)` for tensor file mmap. Used by `parse()` (safetensors, always-on), `parse_pth`, and `parse_gguf`. Same invariants as the upstream `safetensors` crate's mmap path: read-only artefact assumption — concurrent writes by another process are undefined behaviour. |

Each accepted use must satisfy all of:
1. The `unsafe` is concentrated, never scattered — either a single, dedicated
   module (e.g., a future `src/remember/simd.rs` for SIMD intrinsics) or a
   single, tightly-scoped `unsafe { … }` call inside the parser module that
   needs it (the current shape for `memmap2::Mmap::map(&file)` in the
   safetensors, `pth`, and `gguf` parsers).
2. Every `unsafe` block carries a `// SAFETY:` comment documenting the invariants.
3. **Feature-specific opt-ins** are gated behind `#[cfg(feature = "...")]` so
   users who don't enable the feature compile no `unsafe` code at all (the
   `#[allow(unsafe_code)]` sites are excluded from their build). Always-on
   opt-ins (currently: safetensors mmap) are justified by the always-on-ness
   of the operation — there is no feature for a caller to disable.
4. **Non-`unsafe` parity where it exists.** SIMD-shaped opt-ins keep a scalar
   fallback tested identically. mmap-based inspection paths expose a
   reader-generic alternative (`parse_safetensors_header_from_reader`,
   `inspect_npz_from_reader`, …) parity-tested against the mmap path. As of
   Phase 6.13, **full parsing has non-`unsafe` parity too**: every mmap `parse*`
   entry point has a copy-based, zero-`unsafe`, zero-mmap sibling
   (`parse_bytes` / `parse_from_reader` for safetensors, `parse_gguf_bytes` /
   `parse_gguf_from_reader`, `parse_pth_bytes` / `parse_pth_from_reader`) that
   reads the artefact into an owned `Backing::Owned` buffer and shares the same
   structure parser — parity-tested against the mmap path in
   `tests/parse_owned_path.rs`. The owned path is the recommended entry for
   untrusted input (a faulting mmap raises an uncatchable `SIGBUS`); the mmap
   path remains the trusted-local-file fast path. The `unsafe` `Mmap::map` call
   itself still has no non-`unsafe` substitute and stays on the path-based
   entries only.

Adding a new accepted use requires updating this table and the `cfg_attr` lines
in `lib.rs`.

### MSRV Lint Guard

`lib.rs` carries `#![allow(unknown_lints)]` so that `#[allow(clippy::newer_lint)]`
in test modules does not break the MSRV CI build. Without it, every new clippy
lint suppression is a potential MSRV failure because `#![deny(warnings)]` implies
`#[deny(unknown_lints)]`, and the MSRV toolchain's clippy may not recognise lint
names added in later releases.

No special action is required when adding a new `#[allow(clippy::...)]` — the
crate-level `allow(unknown_lints)` covers it automatically. If the MSRV is bumped,
this guard remains necessary as long as the MSRV is behind the development toolchain.

---

## When Writing Control Flow

### `if let` vs `match`

Use the most specific construct for the pattern at hand:

| Situation | Preferred form |
|---|---|
| Testing a single variant, no binding needed | `matches!(expr, Pat)` |
| Testing a single variant, binding needed | `if let Pat(x) = expr { … }` |
| Two or more variants with different bodies | `match expr { … }` |
| Exhaustive dispatch over an enum | `match expr { … }` (never `if let` chains) |

Two anti-patterns Clippy flags: a `match` with one non-`_` arm and `_ => {}`
(`single_match`, `match_like_matches_macro`); three-or-more `if let … else
if let …` chains that should be a `match`.

### EXPLICIT Annotation

`// EXPLICIT: <reason>` — required when a match arm is intentionally a no-op, or when an imperative
loop is used instead of an iterator chain for a stateful computation.
> Example: `// EXPLICIT: FP8 block accumulation is stateful; .map() would hide the scale update`

### EXHAUSTIVE Annotation

`// EXHAUSTIVE: <reason>` — required on `#[allow(clippy::exhaustive_enums)]` or
`#[allow(clippy::wildcard_enum_match_arm)]` when a wildcard is used on a
foreign `#[non_exhaustive]` enum that we cannot match exhaustively.
> Example: `// EXHAUSTIVE: internal dispatch enum; crate owns and matches all variants`
> Example: `// EXHAUSTIVE: SafeTensorError is a foreign type that may gain variants`

---

## When Writing Error Strings

### Error Message Wording

Error strings passed to `AnamnesisError` variants follow two patterns:

- **External failures** (I/O, serde): `"failed to <verb>: {e}"`
  > Example: `AnamnesisError::Parse { reason: format!("failed to parse safetensors header: {e}") }`
- **Validation failures** (format, range): `"<noun> <problem> (<context>)"`
  > Example: `AnamnesisError::Parse { reason: format!("unexpected dtype {dtype} for tensor {name}") }`
  > Example: `AnamnesisError::Unsupported { format: "GPTQ".into(), detail: "3-bit quantization not yet supported".into() }`

Rules:
- Use lowercase, no trailing period.
- Include the offending value and the valid range or constraint when applicable.
- Wrap external errors with `: {e}`, not `.to_string()`.

---

## When Parsing Untrusted Input

Tensor archives are attacker-controllable: a malicious `.safetensors`,
`.npz`, `.pth`, or `.gguf` can claim arbitrary dimensions, point at
arbitrary offsets, or (pickle) reference arbitrary Python globals. The
following invariants hold at every parser entry point.

### Checked arithmetic on header-derived sizes and offsets

Every multiplication or addition that combines values read from the input
(declared shape, element count, offset, length) must use `checked_*` and
map an overflow to `AnamnesisError::Parse`. `usize` arithmetic on
attacker-controlled inputs is a documented bug class — assume it can
overflow on 32-bit targets even when 64-bit math would not.

> ✅ Single-step pattern (e.g., adding a base offset):
> ```rust
> let abs_end = data_offset
>     .checked_add(end)
>     .ok_or_else(|| AnamnesisError::Parse {
>         reason: "tensor data end offset overflow".into(),
>     })?;
> ```
>
> Chained multiplications use `.and_then`:
> ```rust
> let expected_qw_len = packed_rows
>     .checked_mul(out_features)
>     .and_then(|n| n.checked_mul(4))
>     .ok_or_else(|| AnamnesisError::Parse {
>         reason: "qweight byte length overflow".into(),
>     })?;
> ```

`saturating_*` and `wrapping_*` are not substitutes — silent saturation or
wrapping on a header-derived size is a parser bug, not a recovery strategy.
Use `saturating_sub` only for non-security display computations (e.g., a
byte count that can legitimately be zero).

### Cap declared sizes before allocating

Checked arithmetic stops an *overflow*; it does not stop a *plausible but
enormous* size. A header field that decodes to 3 GiB does not overflow
`usize` on a 64-bit target — `vec![0u8; n]` simply tries to commit 3 GiB.
This is the [`candle #3533`](https://github.com/huggingface/candle/issues/3533)
DoS class (CWE-770 / CWE-1284 / CWE-400): a length / count / size read from
the input and fed to `vec![T; n]`, `Vec::with_capacity(n)`, `for _ in 0..n`,
or a per-element clone without an upper-bound check. Every such site must
compare the declared value against an explicit module-level cap and reject
it with `AnamnesisError::Parse` **before** the allocation.

Rules:
- Name the cap `<FORMAT>_MAX_<THING>` (`NPY_MAX_HEADER_BYTES`, `MAX_PKL_SIZE`,
  `MAX_STRING_LEN`, `MAX_PICKLE_PAYLOAD`, …) and document the realistic-vs-cap
  head-room in its doc comment.
- Choose the cap generously above any legitimate file (real `NPY` headers are
  <1 KiB → 1 MiB cap; real `.pth` `data.pkl` is <100 KiB → 100 MiB cap), so
  the gate is invisible to honest inputs and rejects only absurd ones.
- Type the constant `u64` when the literal would overflow `usize` on a 32-bit
  target (`NPZ_MAX_ARRAY_BYTES: u64 = 1 << 33`) and compare with a lossless
  widening cast (`// CAST: usize → u64, lossless widening`).
- When two entry points share a cap (e.g., an mmap path and a reader path),
  factor the check into one helper (`enforce_pkl_size_cap`) so the bound
  cannot drift between them.

The sibling `safetensors` crate caps its header with `MAX_HEADER_SIZE`; every
anamnesis parser mirrors the pattern (`GGUF` `MAX_*`, the safetensors header
cap, the `NPZ`/`.pth` caps above). Pre-allocation hints derived from a
file-declared count get the additional `PREALLOC_SOFT_CAP` clamp (see
`src/parse/gguf.rs`) so a header that *passes* the hard cap still cannot drive
an eager multi-hundred-MB `with_capacity`.

### Validate a declared size against the container's stated size

A constant cap bounds the *worst case*; it does not catch a value that is
absurd *relative to the input at hand*. When the container already states how
many bytes a region holds — a `ZIP` entry's uncompressed `size()`, the
remaining bytes in a length-prefixed stream, a block-format's
`n_blocks × type_size` — cross-check the size you *derived* from other header
fields (a shape product, an element count) against it, and reject the mismatch
**before** allocating or decompressing. This is a third, independent defence:
checked arithmetic stops overflow, the constant cap stops the absurd-in-absolute
case, and this stops the absurd-relative-to-this-file case (and, for compressed
containers, bounds decompression to the entry's honest declared size).

Exemplars in this crate:
- `GGUF` dequant: `validate_dequant_input` rejects unless
  `data.len() == n_blocks × type_size` (`src/remember/gguf.rs`).
- `GGUF` reader: every variable read is gated by `ensure_remaining(n)` against
  the captured `file_len` **before** the buffer is allocated (`src/parse/gguf.rs`).
- `NPZ`: `read_array_data` rejects a shape-derived `data_bytes` exceeding the
  `ZIP` entry's declared `size()` before the `vec!` (`src/parse/npz.rs`).
- Vendored `ZIP` reader: `data_start` cross-checks each entry's
  `data_start + compressed_size` against the source length, and
  `read_central_directory` validates `cd_offset + cd_size ≤ total_len`, before
  any slice or read (`src/parse/zip.rs`).

When the same bound guards two entry points (an mmap path and a reader path),
factor it into one helper so the two cannot drift — as `enforce_pkl_size_cap`
does for the `.pth` parser.

### Magic-byte plus minimum-size guard

Format detection must read the magic bytes via `.get(..N)` (not direct
indexing) so that a file shorter than `N` bytes produces a parser error,
not a panic. Detect known-but-unsupported variants (legacy formats,
byte-swapped headers) before falling through to a generic "not a valid X"
error — the better diagnostic helps users distinguish a corrupt file from
a wrong-format file.

> ✅ Pattern from the `.pth` and `GGUF` parsers:
> ```rust
> let magic = raw.get(..4).ok_or_else(|| AnamnesisError::Parse {
>     reason: "file too small to be a .pth archive".into(),
> })?;
> if magic.first() == Some(&0x80) && magic.get(1).is_some_and(|&b| b <= 0x05) {
>     return Err(AnamnesisError::Unsupported {
>         format: "pth".into(),
>         detail: "legacy .pth format (pre-PyTorch 1.6) is not supported".into(),
>     });
> }
> if magic != b"PK\x03\x04" {
>     return Err(AnamnesisError::Parse {
>         reason: "file is not a ZIP archive (missing PK\\x03\\x04 magic)".into(),
>     });
> }
> ```

### Allowlist, never denylist

Decoders that interpret opcodes, type tags, or symbolic references
(pickle VM, opcode handlers) accept an explicit allowlist; everything else
returns `AnamnesisError::Unsupported`. Never write a denylist — the attacker
chooses what to send, and the unknown set is unbounded.

Canonical example: the pickle VM in the `.pth` parser allowlists `GLOBAL`
references (`torch._utils.…`, `collections.OrderedDict`, …); all others
return `Unsupported`. New parsers that interpret input symbols follow the
same shape.

### Pre-validate slices once, iterate branch-free inside

The two-level rule from
[Reconciling bounds checking with vectorization](#reconciling-bounds-checking-with-vectorization)
also has a security justification: per-element `.get()? / .ok_or` inside a
loop spreads the bounds proof across N return paths; pre-validating once
concentrates it at one readable site. Applies in every parser hot path, not
just dequant kernels.

---

## When Batching Operations by Key

### HashMap Grouping Idiom

When operations must be batched by a key (e.g., grouping tensors by
quantization scheme), use the `Entry` API:

```rust
let mut by_scheme: HashMap<QuantScheme, Vec<TensorInfo>> = HashMap::new();
for tensor in tensors {
    by_scheme.entry(tensor.scheme()).or_default().push(tensor);
}
```

Rules:
- Name the map `by_<grouping_key>` (e.g., `by_scheme`, `by_dtype`).
- Use `.entry(key).or_default().push()` — never `if let Some` + `else insert`.

---

## When Writing SIMD-Friendly Loops

anamnesis processes billions of elements (one per model parameter). Performance
depends on the compiler auto-vectorizing the inner conversion loops. The
following rules ensure that hot loops remain vectorizable.

### Write loops that the compiler can vectorize

The compiler auto-vectorizes when it can prove: no aliasing, no branches,
no cross-iteration dependencies, fixed stride, and known trip count. Write
loops that satisfy these conditions:

1. **Process contiguous slices, not iterators over structs.**
   Work on flat slices (e.g., `&[u8]` for packed data, `&[f32]` for
   precomputed values, `&mut [u8]` for `BF16` output), not abstracted types.

   > ✅ `for i in 0..block.len() { out[i] = convert(block[i], scale); }`
   > ✅ `block.iter().zip(out.iter_mut()).for_each(|(&b, o)| *o = convert(b, scale));`
   > ❌ `for tensor in tensors { for elem in tensor.elements() { ... } }` ← indirect, defeats vectorization

2. **No branches in the hot path.** Conditional logic (NaN handling, subnormal
   detection) must be expressed as bitwise select or arithmetic, not `if`/`match`.
   This includes bounds checks — see [reconciling bounds checking with
   vectorization](#reconciling-bounds-checking-with-vectorization) below.

   > ✅ `let is_nan_mask = ((exp == 0xF) & (mant != 0)) as u16;` then blend with bitwise ops
   > ❌ `if exp == 0xF && mant != 0 { return BF16_NAN; }` ← branch per element kills SIMD

3. **Hoist invariants out of the loop.** The scale factor is constant for an
   entire block (128 elements in fine-grained FP8) or an entire row (GPTQ
   group). Compute it once before the loop, not inside.

   > ✅ `let scale = load_scale(block_idx);  for i in 0..128 { ... scale ... }`
   > ❌ `for i in 0..128 { let scale = load_scale(block_idx); ... }` ← redundant load, may confuse optimizer

4. **Use exact chunk sizes.** `chunks_exact()` tells the compiler the trip
   count is a multiple of the chunk size, enabling full-width SIMD with no
   remainder loop.

   > ✅ `for chunk in data.chunks_exact(128) { ... }`
   > ❌ `for chunk in data.chunks(128) { ... }` ← last chunk may be short, generates scalar remainder

5. **Avoid `as` casts that widen through intermediate types.** A chain like
   `u8 → u32 → f32 → f32 (multiply) → u16` may force the compiler to emit
   widening/narrowing shuffles that break vectorization. Prefer bitwise
   manipulation that stays in the target width. When the pipeline inevitably
   crosses domains (e.g., GPTQ: byte extraction → float arithmetic → integer
   `BF16` rounding), use loop fission — see below.

6. **Separate input and output slices.** The compiler cannot vectorize if it
   suspects the output aliases the input. Use distinct `&[u8]` input and
   `&mut [u8]` output buffers — never in-place transformation on a single
   `&mut [u8]`.

### Reconciling bounds checking with vectorization

The `// INDEX:` rule (above) says to prefer `.get()` with `?` for bounds
checking. In hot loops, `.get()` generates a branch on every element — the
exact pattern that rule 2 ("no branches in the hot path") forbids.

Satisfy both rules with a **two-level pattern**:

1. **Before the loop:** validate bounds **once** using `.get(range)` with `?`
   on the enclosing slice. This produces a pre-validated sub-slice whose
   length is known at the start of the loop.

2. **Inside the loop:** iterate the pre-validated slice with `chunks_exact`,
   `.iter().zip()`, or direct indexing annotated with `// INDEX:` referencing
   the pre-validation. No `.get()`, no `?`, no branches.

> ✅ Pre-validate, then iterate branch-free:
> ```rust
> let row = data.get(start..start + cols).ok_or_else(|| err())?;  // bounds checked once
> let out = output.get_mut(o_start..o_start + cols * 2).ok_or_else(|| err())?;
> // VECTORIZED: inner loop is branch-free, pre-validated slices
> for (&byte, out_pair) in row.iter().zip(out.chunks_exact_mut(2)) {
>     out_pair.copy_from_slice(&convert(byte, scale).to_le_bytes());
> }
> ```
>
> ❌ Per-element `.get()` inside the loop — defeats vectorization:
> ```rust
> for j in 0..cols {
>     let byte = data.get(start + j).ok_or_else(|| err())?;   // branch per element
>     let scale = scales.get(j).ok_or_else(|| err())?;         // branch per element
>     // compiler cannot vectorize — too many conditional paths
> }
> ```

This pattern appears throughout anamnesis: the FP8 dequantization loops
pre-slice weight and output rows before the inner `chunks_exact` iteration.
Every new dequantization module (GPTQ, AWQ, BnB) must follow the same
two-level structure.

### Loop fission for mixed-domain pipelines

When a single loop mixes data domains (byte extraction, float arithmetic,
integer bit-manipulation for `BF16` rounding), the compiler often cannot
vectorize because it sees cross-domain dependencies in the loop body.

**Split the loop into separate passes**, one per domain. Each pass has a
uniform data flow that the compiler can vectorize independently. Use a
scratch buffer (typically one row, fits in L1 cache) to communicate between
passes.

> ✅ Two passes — byte extraction then float arithmetic:
> ```rust
> // VECTORIZED: scalar fallback — Pass-1 of fission; byte unpacking
> // crosses byte/integer/float domains and is intentionally scalar
> for (chunk, dst) in raw.chunks_exact(4).zip(scratch.iter_mut()) {
>     let packed = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
>     *dst = ((packed >> shift) & mask) as f32;
> }
> // VECTORIZED: confirmed AVX2 vsubps + vmulps in cargo-show-asm,
> // x86-64 target-cpu=native, opt-level=3
> for ((&val, &zero, &scale), out) in scratch.iter()
>     .zip(zeros.iter()).zip(scales.iter()).zip(output.chunks_exact_mut(2)) {
>     let bf16 = f32_bits_to_bf16_bits(((val - zero) * scale).to_bits());
>     out.copy_from_slice(&bf16.to_le_bytes());
> }
> ```
>
> ❌ Single combined loop — mixes byte, float, and integer domains:
> ```rust
> for (chunk, out) in raw.chunks_exact(4).zip(output.chunks_exact_mut(2)) {
>     let packed = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
>     let qw = ((packed >> shift) & mask) as f32;       // byte → int → float
>     let val = (qw - zero) * scale;                     // float
>     let bf16 = f32_bits_to_bf16_bits(val.to_bits());   // float → int → truncate
>     out.copy_from_slice(&bf16.to_le_bytes());           // compiler gives up
> }
> ```

The cost of the extra pass is negligible: the scratch buffer is one row
(typically 1–32 KB, fits in L1 cache). The benefit is full-width SIMD on
the arithmetic pass. In GPTQ dequantization, loop fission turned scalar
`vsubss`/`vmulss` into AVX2 `vsubps`/`vmulps` (8-wide), improving the
function from 1.5–2.8× to **6.5–12.2× faster** than CPU PyTorch.

**When to expect the need for fission:** if the input format packs multiple
logical values per storage word (GPTQ: 8 INT4 per `I32`, AWQ: similar), the
unpacking step will mix byte/integer domains with the float arithmetic.
Plan two passes from the start. Conversely, if the input-to-output mapping
is 1:1 with a small branchless inline kernel (FP8: one byte → one `BF16`),
the compiler typically fuses the pipeline without fission.

### Verify vectorization

After writing a hot loop, **verify** that the compiler actually vectorized it.
Do not assume — auto-vectorization is fragile and can silently regress.

The `// VECTORIZED:` annotation has three states:

- **`// VECTORIZED: confirmed <ISA> <key instruction> in cargo-show-asm,
  <target>, opt-level=3`** — written when *both* of the following hold:
  1. The disassembly contains packed SIMD instructions for the loop body
     (e.g., AVX2: `vmulps`, `vsubps`, `vpsrld`, `vpaddd`, `vpackusdw`;
     NEON: `fmul.4s`, `ushll`). Scalar variants (`vmulss`, `vsubss`)
     indicate the loop did not vectorize. Inspect the assembly via
     `cargo-show-asm` or `RUSTFLAGS="-C target-cpu=native --emit=asm"
     cargo build --release`.
  2. A release-mode measurement (best-of-5 median, `target-cpu=native`,
     real fixture) shows the kernel is at least as fast as the previous
     scalar baseline. See `CLAUDE.md` § Performance Changes for the
     measurement protocol. Vectorization presence ≠ throughput
     improvement — memory-bandwidth-bound kernels can vectorize without
     speeding up.

- **`// VECTORIZED: scalar fallback — <reason>`** — written when the loop
  did not vectorize and won't (e.g., a Pass-1 byte-unpacking step in a
  loop-fission pipeline; a control-flow shape the compiler cannot prove
  branch-free). The `<reason>` must be specific enough that a reader can
  tell whether the situation has changed. Pass-1 of every loop-fission
  kernel carries this annotation; only Pass-2 is expected to vectorize.

- **`// VECTORIZED: pending <what is missing>`** — temporary, written when
  the kernel was just authored or modified and verification has not yet
  been performed. **`pending` annotations must resolve to `confirmed` or
  `scalar fallback` before the next `vX.Y.0` release tag.** A `pending`
  annotation older than one release is a release blocker, not a TODO.

Re-verify after any change to the loop body or its dependencies — the
annotation goes back to `pending` and the verification cycle repeats.

### Benchmark evidence: what a criterion baseline can and cannot show

`cargo bench -- --save-baseline X` then `--baseline X` is the cheapest way to
compare two versions of a kernel. It is a **screening** instrument, not an
adjudicating one, and the difference is not academic.

Criterion's p-value describes sampling noise *within* one run of one binary. It
does not model the effect that dominates on this project's development host:
whole-program code layout. The two are easy to confuse because criterion reports
the second with the confidence of the first.

**Measured, 2026-08-21**, while evaluating a `clippy::chunks_exact_to_as_chunks`
migration: `dequant_bnb_nf4` reported **+10.6 % at p = 0.00** against a saved
baseline while its kernel's source was **byte-identical** (`git diff` on the file
was empty). The only thing that had changed was unrelated code elsewhere in the
crate. `docs/perf-experiments.md` Experiment 14 independently puts this host's
layout sensitivity at up to **23 %**.

Rules that follow, not suggestions:

1. **A criterion baseline comparison cannot establish a regression or a win by
   itself.** Use it to decide what is worth measuring properly.
2. **Below roughly 15 % on this host, treat the result as no signal** — whatever
   the p-value says.
3. **Any claim that reaches a commit message, `CHANGELOG.md`, or
   `docs/perf-experiments.md` needs the interleaved best-of-N protocol in
   [`CLAUDE.md`](CLAUDE.md) § *Performance Changes***: alternating binaries,
   median or min of N, on an idle machine. That protocol controls for layout by
   construction; criterion's baseline does not.
4. **Prefer a low-noise instrument when one exists.** CodSpeed macro runners are
   bare-metal and isolated; a result that survives there outranks a local
   criterion delta. Note the converse trap recorded in Experiment 12: a
   file-writing benchmark can flatten to 1.00× there because CI storage
   dominates, so choose a benchmark whose numerator is the code under test.
5. **Attribute a screened regression before acting on it.** Reverting a whole
   change because a bench moved is how a genuine improvement gets discarded
   beside a layout artefact — and, symmetrically, how a real regression gets
   waved through as "probably noise".

### When auto-vectorization is not enough

If a verified scalar loop is too slow and the compiler refuses to vectorize
despite following the rules above, escalate in this order:

1. **`#[target_feature(enable = "avx2")]` + `is_x86_feature_detected!`** —
   explicit opt-in to wider SIMD on x86-64 while keeping a portable fallback.
   Requires `unsafe` (see `// SAFETY:` rules) and a feature gate.
2. **`pulp` or `portable-simd`** — stable-Rust portable SIMD abstractions.
   Adds a dependency but avoids hand-written intrinsics.
3. **Hand-written intrinsics with `#[cfg(target_arch)]`** — last resort.
   Must include a scalar fallback for non-x86/non-ARM targets.

> **Empirical note (Phase 7, `docs/perf-experiments.md` Experiment 10):** explicit
> SIMD was *measured* to add nothing to anamnesis's dequant kernels — the shared
> pass-2 `f32 → BF16` writer is bandwidth-bound (a bit-exact hand-written AVX2
> `f32x8_to_bf16x8` scored 1.02×), the compiler already auto-vectorizes the
> arithmetic, and the slowest kernels (IQ2/IQ3) are gather-bound. The largest
> measured lever is **multi-threading**, governed by the section below. Reach for
> the escalation ladder above only after a roofline says a loop is compute-bound
> *and* not already vectorized *and* not gather-bound.

---

## When Parallelizing Work

anamnesis dequantizes billions of elements; the kernels are embarrassingly
parallel (each output element depends only on its own input block — no
cross-element state) yet run single-threaded today. Multi-threading is the
largest *measured* performance lever (`docs/perf-experiments.md` Experiment 10).
Concurrency has nastier failure modes than SIMD — data races, non-determinism,
DoS amplification — so the discipline is codified here before the code, exactly
as the SIMD rules were.

### Parallelize only when all of these hold

1. **Provable independence.** Each task writes a **disjoint** output region and
   reads only shared-**immutable** input. No shared mutable state, no lock in the
   hot path, no cross-task reduction. Dequant qualifies: output element `i`
   depends only on input block `⌊i / block_size⌋` and its scale.
   - **One sanctioned exception: a scheduling cursor.** A shared `AtomicUsize`
     that hands each idle worker the next *work item* is permitted, because it
     carries no output data and is touched **once per item, never inside a
     kernel loop**. It exists to fix a real problem a static split cannot:
     tensor sizes are wildly skewed (one `token_embd.weight` can outweigh two
     hundred norm tensors), so an equal-count partition leaves one worker
     holding most of the bytes while the pool idles. The independence rule is
     unchanged for everything the cursor schedules — each task still writes only
     its own output. This is the only shared mutable state permitted; anything
     that accumulates a *result* across tasks is still a reduction and still
     banned. See `src/parallel.rs`.
2. **A measured win.** A release-mode best-of-N median on a real fixture shows a
   scaling win past the thread spawn/join overhead — the same
   "no measured win → no commit" gate every perf claim obeys
   ([`CLAUDE.md`](CLAUDE.md) § Performance Changes). Record it in
   `docs/perf-experiments.md` against the sequential baseline.
3. **Above a size threshold.** Below a named `*_MIN_PARALLEL_*` element/tensor
   count, spawn+join cost dominates and the **sequential path** runs instead.
   Document the threshold's calibration.

### Determinism is mandatory — bit-exact and thread-count-invariant

The crate's entire value is 0-ULP dequant validated against PyTorch. Parallelism
must **never** change an output byte:

- **Partition by disjoint output region** (row/tensor ranges via `split_at_mut` /
  `chunks_mut`); each task writes only its own slice. Output bytes are then
  independent of how the work was split.
- **If tasks are claimed dynamically, tag results with their input index and
  sort before returning.** With a scheduling cursor the *completion* order is
  nondeterministic by design; the *returned* order must not be. Restoring input
  order is what keeps output bytes independent of steal order as well as of
  thread count.
- **Error selection must be deterministic too.** A sequential loop reports the
  **lowest-indexed** failure; a parallel dispatch must report the same one, not
  whichever worker happened to fail first — otherwise a malformed input yields a
  different error message run to run and tests become flaky.
- **Never parallelize a floating-point reduction.** `f32`/`f64` addition is
  non-associative, so a parallel sum reorders rounding and changes the result.
  Dequant has no reductions — keep it that way; if one is ever needed, it runs
  sequentially or uses a fixed, order-preserving tree.
- **A determinism test is required**: assert byte-identical output across thread
  counts `{1, 2, N}` for every parallelized path. Thread count is a performance
  knob, never a correctness variable.

### Mechanism: scoped `std` threads by default; `rayon` only behind a feature

- Default to **`std::thread::scope` + manual disjoint chunking** — **zero
  dependency**, matching the crate's minimal-dependency posture (the vendored ZIP
  reader exists precisely to keep `zip` out of the runtime tree). Scoped threads
  borrowing `split_at_mut` slices are **safe Rust** — parallelism adds **no
  `unsafe`** (unlike SIMD). Never reach for manual `Send`/`Sync` impls or raw
  pointer sharing to parallelize.
- If `rayon`'s ergonomics are wanted later, gate it behind a **`rayon` feature
  flag** so the default build pulls no new dependency, and keep the sequential
  path as the always-available fallback.

### The caller owns the thread budget — never the input file

- **Caller-controllable, and modest by default.** Thread count is an explicit
  parameter, and the sequential path is always reachable. The **default is
  deliberately small — the scaling knee, not the core count**: `min(available_parallelism, 4)`.
  Dequant throughput is memory-bandwidth-bound and plateaus at ~3–4× by ~4 threads
  (`docs/perf-experiments.md` Experiment 11), so 4 captures ~90 % of the win while
  leaving the rest of the host's cores free for the caller's own work. A library
  must never unilaterally saturate the machine: grabbing all cores buys nothing
  past the plateau and actively harms the host process that embeds it. The cap is
  a **default, not a ceiling** — because the limiting resource is DRAM bandwidth,
  a memory-rich host (many-channel DDR5 server) can profitably raise it through
  the caller parameter; hard-coding the core count would be wrong in both
  directions.
- **Bounded by hardware, never by the input.** Thread/task count derives from
  hardware parallelism, **never** from any file-declared quantity (tensor count,
  shape, block count). A malicious archive declaring 10⁹ tensors must not spawn
  10⁹ threads — this extends the [untrusted-input](#when-parsing-untrusted-input)
  DoS invariant to concurrency. Partition into a fixed, hardware-sized set of
  chunks; never spawn per file-entity.
- **Not on parse/inspect paths.** Never spawn threads while interpreting an
  untrusted header. Parallelism belongs on the compute-heavy transform paths
  (`remember`/`forget`/`convert`) operating on already-validated data.

### PARALLEL Annotation

`// PARALLEL: <partition strategy + independence/determinism invariant>` —
required at every site that spawns threads or drives a parallel iterator.
Documents what makes the partition race-free and the output deterministic,
mirroring the `// VECTORIZED:` discipline.

> Example: `// PARALLEL: output rows split via chunks_mut(rows_per_thread) into`
> `disjoint slices; each thread reads shared-immutable qweight/scales, writes only`
> `its own rows — no reduction, byte-identical to the sequential path`

### Verify parallelism

Like vectorization, **verify** — do not assume:

1. The determinism test (byte-identical across thread counts incl. 1) is green.
2. The scaling win is measured (best-of-N, real fixture) and recorded in
   `docs/perf-experiments.md` with the sequential baseline. A criterion
   `--baseline` delta is screening only; see
   [Benchmark evidence](#benchmark-evidence-what-a-criterion-baseline-can-and-cannot-show).
3. Where the platform supports it, the parallelized path is exercised under a
   race detector (ThreadSanitizer on a Linux CI runner — MSVC/Windows has no TSan;
   the disjoint-slice design is race-free by construction, so this is a backstop).
   **Satisfied** by the gating `tsan` job in
   [`.github/workflows/tsan.yml`](.github/workflows/tsan.yml), which runs the
   parallel dispatch under `-Zsanitizer=thread` with an **instrumented `std`**
   (`-Zbuild-std`) and reports zero races across `{1, 2, 4, 8}` threads plus the
   resolved default. *(Landed just after the v0.7.2 tag; that release shipped
   with this item still open, which its CHANGELOG entry records honestly.)*

   An instrumented `std` is not a refinement here but a precondition: without
   it, `std::thread::scope`'s own `Arc` teardown false-positives
   (rust-lang/rust#101206) and scoped threads *are* the mechanism under test.
   Getting there needs three things to hold at once — a `--bin` target rather
   than `cargo test`, blanket `RUSTFLAGS` rather than the target-scoped form,
   and `panic_abort` in the build-std set. Each is load-bearing and each was
   learned from a red build; the workflow header explains all three. If you are
   adding a race-detector run to another project, read it before improvising.
4. **The auto-traits the dispatch depends on are asserted, not assumed.** Sharing
   `&self` across scoped threads needs `Sync`, which holds only because every
   field happens to be `Sync` — adding a `Cell`/`Rc`/raw pointer would silently
   break it, and the error would surface as an inscrutable closure bound far from
   the offending field. Assert it (`tests/parallel_contract.rs`) so the failure
   names the type.
5. **Any test or benchmark meant to prove the parallel path must clear the
   `*_MIN_PARALLEL_*` threshold.** Below it the dispatch is sequential, so an
   undersized fixture produces a green determinism suite and a flat benchmark
   that both measure nothing. Size fixtures off the constant itself where the
   visibility allows, and assert the fixture clears it where it does not.
