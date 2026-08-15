# Python interop — design notes for the PyO3 bindings

<!-- Last updated: 2026-06-25, anamnesis v0.6.8 (Phase 6.13) -->

This is the contract the [Phase 8](../ROADMAP.md#phase-8-python-bindings-pyo3)
PyO3 bindings (`pip install anamnesis-quant`) implement. It is written **before** the
bindings so the core can honour each guarantee and the API shape is frozen rather
than retrofitted. The Rust core is hardened library-side (every guarantee below
benefits Rust consumers too); this file records the Python-facing consequences.

## Panic safety & the `unwind` requirement (Phase 6.13 Step 3)

**Guarantee.** No public parse/inspect entry point panics or aborts on *any*
input. A malformed, truncated, or hostile artefact is always a clean
`Result::Err` (`AnamnesisError`), never an unwinding panic and never a `SIGBUS`
(the copy-based `parse_bytes` / `parse_*_from_reader` paths from Step 1 use no
mmap). This is pinned by `tests/no_panic.rs` (a `catch_unwind` battery over
adversarial inputs across every entry point, run in debug so integer-overflow
panics are in scope) and the coverage-guided `cargo fuzz` harness (including the
owned-path `fuzz_*_bytes` targets).

**Why the binding must build `panic = "unwind"`.** PyO3 wraps each `#[pyfunction]`
in a panic boundary that converts an unwinding Rust panic into a Python
`pyo3_runtime.PanicException` — *but only while panics unwind*. Under
`panic = "abort"` a panic is an immediate, uncatchable process kill: one hostile
upload would take down a multi-tenant worker, voiding the "never a dead worker"
pledge.

anamnesis's shipped library/CLI builds set `panic = "abort"` deliberately — that
is the correct fail-closed posture for a standalone parser (a *reachable* panic,
were one to exist, should kill the process rather than unwind through C FFI or an
inference loop). To reconcile the two, the cdylib is built with a dedicated
profile:

```toml
# Cargo.toml
[profile.release]          # CLI / library
panic = "abort"

[profile.python]           # the PyO3 cdylib — maturin `--profile python`
inherits = "release"       # identical codegen to the shipped wheel
panic = "unwind"           # so PyO3 yields a catchable PanicException
```

The Phase 8 maturin build selects `[profile.python]`; the contract is guarded in
stable CI today by `tests/panic_profile.rs` (asserts release = abort **and**
python = unwind), so it cannot silently regress before the cdylib exists.

**Net for a Python host.** Because (a) the core never panics on untrusted input
and (b) the wheel unwinds, the binding can wrap a hostile upload in
`try/except AnamnesisError` and map it to an HTTP response — a `LimitExceeded`
→ *413*, a `Parse` → *400*, a `DisallowedGlobal` → a flagged security event — and
even an *unexpected* panic (a bug, not an input) surfaces as a catchable
`PanicException` instead of killing the worker. See the error → exception map on
`AnamnesisError` (and the README "Parsing untrusted input" section).

## NumPy / BF16 data-ownership contract (Phase 6.13 Step 4)

Two coupled decisions the Phase 8 `numpy` interop hard-depends on, locked here so
the binding implements rather than re-litigates them — and so a published `pip`
API never freezes an unsafe shape.

### Ownership — *owned copy by default*

**Rule.** A NumPy array the binding hands back must either **own** its bytes or
borrow Rust memory through a lifetime-safe capsule that keeps the owner alive —
**never** a bare view into a `Backing` the owning `Parsed*` can drop. A bare view
is a **use-after-free reachable from pure Python**: `arr = model.tensor("w"); del
model` would free the mmap/`Vec` the array still points at.

**Decision: the first wheel copies.** Every returned array owns its bytes. The
core already supports this on every path — no new API, no lifetime gymnastics in
the binding:

| Format | Accessor | Today | Binding takes ownership via |
|---|---|---|---|
| safetensors | `ParsedModel::remember_to_bytes` | **owned** `Vec<u8>` (dequantised `BF16`) | already owned |
| npz | `NpzTensor::data` | **owned** `Vec<u8>` | already owned |
| GGUF | `ParsedGguf::tensors` → `GgufTensor::data` | `Cow::Borrowed` into the `Backing` | `Cow::into_owned()` |
| `.pth` | `ParsedPth::tensors` → `PthTensor::data` | `Cow::Borrowed` into the `Backing` | `Cow::into_owned()` |

Only GGUF and `.pth` `tensors()` borrow (zero-copy `Cow::Borrowed` slices into the
`Backing`); the binding calls `.into_owned()` before constructing the array, so no
array ever aliases a droppable `Backing`. Combined with Step 1's copy-based
`parse_*_bytes` / `parse_*_from_reader` entry points (owned `Backing`, no mmap),
the untrusted-input path is owned end to end. The Rust-side guarantee — *owned
extraction outlives a dropped `Parsed*`* — is pinned by
`tests/python_ownership_contract.rs`.

**Deferred opt-in: zero-copy.** A later release may offer a zero-copy array whose
NumPy `base` is a `PyCapsule` holding a reference to the owning `Parsed*` (so the
GC cannot drop it while the array lives). It is a real copy-elision win but a
use-after-free footgun if the lifetime wiring is ever wrong, so it is **out of
scope for the first wheel** — opt-in, never the default.

### BF16 — exact, never silently widened

NumPy has no native `bfloat16`. anamnesis's whole purpose is *exact* precision
recovery, so the binding must not quietly upcast.

**Decision.** Return an [`ml_dtypes.bfloat16`](https://github.com/jax-ml/ml_dtypes)
array when that (optional) Python dependency is importable; otherwise return raw
`bytes` + an explicit `"bfloat16"` dtype string the caller can reinterpret. Never
silently widen `BF16` → `float32` (it doubles memory and discards the
exact-bytes property). This mirrors what the core already models: `NpzDtype::BF16`
is a first-class variant, and the NPZ parser already reads the JAX void-`V2`
`bfloat16` convention. All other dtypes (`F16`, `F32`, `I32`, …) map to their
native NumPy types.

**Restated for the caller-chosen output dtype (v0.7.3 / v0.7.4).** The contract
above was frozen when `BF16` was the *only* thing a dequantising call could
return, so it reads as a workaround for a NumPy gap. It is worth separating the
two halves now that it is not the only option:

- **The no-silent-widening rule is unchanged and is not negotiable.** If the
  caller asks for `BF16` and gets `BF16` bytes, the binding hands back exactly
  those bytes. Widening behind the caller's back would double memory and destroy
  the exact-bytes property that the cross-validation suites exist to guarantee.
- **What changed is that widening is now a request rather than a workaround.**
  `remember` (v0.7.4) and `convert` (v0.7.3) both take the output dtype as a
  parameter, so a caller who wants `float32` asks for it *up front* and the
  kernels produce `f32` directly. That is strictly better than widening after the
  fact: there is no narrowing step to undo, and the result is the reference
  implementation's own `f32` rather than a `BF16` value with zeros in the low
  bits.

The practical consequence for the wheel is that `ml_dtypes` moves **off the
common path**. A NumPy user who wants a plain `np.float32` array asks for `f32`
and gets one with no optional dependency at all; `ml_dtypes` is needed only by
callers who specifically want `bfloat16` back. That was the Phase 8 de-risking
the 7.3 / 7.4 pair was meant to buy, and it is now bought.

Two things the binding must therefore expose, and one it must not:

- The output dtype must be a parameter on the dequantising entry points, spelled
  the way the rest of the API spells it (`bf16` / `f32` / `f16`).
- The returned array's dtype must reflect what was actually produced, never a
  fixed assumption. The core already learned this the hard way in v0.7.4: an
  `inspect` size estimate was rendered under a hard-coded `BF16` label after the
  figure itself became dtype-aware.
- It must **not** offer a "give me whatever is cheapest" mode. The caller chooses
  the width, or gets the documented default (`bf16`); an output whose dtype
  depends on what the input happened to be is not a contract.

**Passthrough tensors keep their source dtype**, whatever is requested. A
`remember` result is legitimately a mixed-dtype mapping, and the binding should
surface it as such rather than flattening it, because flattening would either
invent precision or discard it. See the
[FAQ](FAQ.md#does-asking-for-f32-rewrite-every-tensor-as-float32).

See also the *Panic safety* section above and the README "Parsing untrusted
input" error taxonomy — together they are the safety contract the bindings ship
against.
