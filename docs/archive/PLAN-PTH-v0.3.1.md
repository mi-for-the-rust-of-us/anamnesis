# Implementation Plan: `.pth` Parsing — anamnesis v0.3.1

> **HISTORICAL (archived at the v0.7.8 close-out).** The implementation plan for `.pth` support, which shipped in v0.3.1. The code and `ROADMAP.md` Phase 3.5 are the current record.

## Goal

Add PyTorch `.pth` (state_dict) parsing and conversion to safetensors. This enables
loading pre-trained weights from repositories that only ship `.pth` files (e.g.,
AlgZoo's 400+ tiny models on GCS, plus thousands of older HuggingFace models).

**Scope**: Parse `.pth` → extract tensor metadata + raw data → write `.safetensors`.
No dequantization needed (`.pth` tensors are already full-precision F32/F64/BF16/F16).

**License compatibility**: MIT-0 (AlgZoo) is compatible with anamnesis's MIT/Apache-2.0.

---

## Background: The `.pth` Format

Since PyTorch 1.6, `torch.save()` produces a **ZIP archive** (not raw pickle). Structure:

```
model.pth (ZIP archive)
├── archive/
│   ├── data.pkl          ← Pickle stream: reconstructs the state_dict structure
│   ├── byteorder         ← Text file: "little" or "big"
│   └── data/
│       ├── 0             ← Raw bytes for tensor 0
│       ├── 1             ← Raw bytes for tensor 1
│       └── ...           ← One file per tensor
```

### How Tensors Are Stored

The pickle stream contains `torch._utils._rebuild_tensor_v2` calls:

```python
REDUCE(
    GLOBAL("torch._utils", "_rebuild_tensor_v2"),
    TUPLE(
        BINPERSID(("storage", storage_type, "0", "cpu", num_elements)),
        storage_offset,
        size_tuple,         # e.g., (16, 10) for a 16×10 matrix
        stride_tuple,       # e.g., (10, 1) for row-major
        requires_grad,
        EMPTY_DICT          # metadata (usually empty)
    )
)
```

Key facts:
- **`BINPERSID`** references data files by index (`"0"` → `archive/data/0`)
- **`storage_type`** encodes dtype: `torch.FloatStorage` → F32, `torch.HalfStorage` → F16,
  `torch.BFloat16Storage` → BF16, `torch.DoubleStorage` → F64, `torch.LongStorage` → I64,
  `torch.IntStorage` → I32, `torch.ShortStorage` → I16, `torch.CharStorage` → I8,
  `torch.ByteStorage` → U8
- **Strides** indicate memory layout; contiguous tensors have strides matching row-major
- The state_dict is an `OrderedDict` mapping string keys → tensors

### Pickle Opcodes Needed (Minimal Subset)

We do NOT need a full pickle implementation. The following ~36 opcodes cover all
PyTorch state_dict files:

| Opcode | Hex | Purpose |
|--------|-----|---------|
| `PROTO` | `\x80` | Protocol version (2–5) |
| `STOP` | `.` | End of pickle |
| `GLOBAL` | `c` | Import `module.class` |
| `REDUCE` | `R` | Call callable with args |
| `BUILD` | `b` | Set `__dict__` |
| `NEWOBJ` | `\x81` | `cls.__new__(cls, *args)` |
| `NONE` | `N` | Push `None` |
| `NEWTRUE` | `\x88` | Push `True` |
| `NEWFALSE` | `\x89` | Push `False` |
| `EMPTY_DICT` | `}` | Push `{}` |
| `EMPTY_LIST` | `]` | Push `[]` |
| `EMPTY_TUPLE` | `)` | Push `()` |
| `MARK` | `(` | Push mark on stack |
| `TUPLE` | `t` | Pop to mark → tuple |
| `TUPLE1` | `\x85` | Pop 1 → tuple |
| `TUPLE2` | `\x86` | Pop 2 → tuple |
| `TUPLE3` | `\x87` | Pop 3 → tuple |
| `SETITEMS` | `u` | Pop mark → dict items |
| `SETITEM` | `s` | Pop key, value → dict |
| `APPEND` | `a` | Append to list |
| `APPENDS` | `e` | Extend list from mark |
| `BINPERSID` | `Q` | Persistent load (tensor data ref) |
| `SHORT_BINUNICODE` | `\x8c` | String (1-byte len) |
| `BINUNICODE` | `X` | String (4-byte len) |
| `BININT` | `J` | 4-byte signed int |
| `BININT1` | `K` | 1-byte unsigned int |
| `BININT2` | `M` | 2-byte unsigned int |
| `LONG1` | `\x8a` | Arbitrary-precision int (1-byte size) |
| `BINPUT` | `q` | Memo store (1-byte key) |
| `LONG_BINPUT` | `r` | Memo store (4-byte key) |
| `BINGET` | `h` | Memo load (1-byte key) |
| `LONG_BINGET` | `j` | Memo load (4-byte key) |
| `SHORT_BINSTRING` | `U` | Bytes (1-byte len, protocol 2) |
| `BINSTRING` | `T` | Bytes (4-byte len) |
| `FRAME` | `\x95` | Frame marker (protocol 4+) |
| `MEMOIZE` | `\x94` | Auto-memo (protocol 4+) |
| `BINBYTES` | `B` | Bytes (4-byte len, protocol 3+) |
| `SHORT_BINBYTES` | `C` | Bytes (1-byte len, protocol 3+) |
| `STACK_GLOBAL` | `\x93` | GLOBAL from stack (protocol 4+) |

### Security Boundary

Unlike Python's pickle, we **never execute arbitrary code**. The parser:
1. Rejects any `GLOBAL` that is not in an explicit allowlist
2. Rejects `INST`, `OBJ`, `EXT1/2/4` (extension registry), `PERSID` (text persistent IDs)
3. Treats unrecognized opcodes as parse errors

This is equivalent to PyTorch's `weights_only=True` but stricter.

---

## Implementation Plan

### Phase A: Pickle Parser (`src/parse/pth.rs`)

**New file: `src/parse/pth.rs`** (~300–400 lines)

#### A.1 — Data Types

```rust
/// Dtype derived from PyTorch storage class names.
///
/// Maps `torch.FloatStorage` → `F32`, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PthDtype {
    F16,   // torch.HalfStorage
    BF16,  // torch.BFloat16Storage
    F32,   // torch.FloatStorage
    F64,   // torch.DoubleStorage
    U8,    // torch.ByteStorage
    I8,    // torch.CharStorage
    I16,   // torch.ShortStorage
    I32,   // torch.IntStorage
    I64,   // torch.LongStorage
    Bool,  // torch.BoolStorage
}

impl PthDtype {
    pub fn byte_size(self) -> usize { ... }

    /// Convert to anamnesis `Dtype` for safetensors output.
    pub fn to_dtype(self) -> crate::Dtype { ... }
}

/// A single tensor extracted from a `.pth` file.
pub struct PthTensor {
    /// Tensor name (state_dict key, e.g. `"rnn.weight_ih_l0"`).
    pub name: String,
    /// Tensor shape (e.g. `[16, 1]`).
    pub shape: Vec<usize>,
    /// Tensor dtype.
    pub dtype: PthDtype,
    /// Raw tensor data, row-major, native endian.
    pub data: Vec<u8>,
}
```

#### A.2 — Pickle VM (Internal)

A minimal stack machine that interprets pickle opcodes:

```rust
/// Value on the pickle VM stack.
enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    String(String),
    Bytes(Vec<u8>),
    Tuple(Vec<PickleValue>),
    List(Vec<PickleValue>),
    Dict(Vec<(PickleValue, PickleValue)>),
    Global { module: String, name: String },
    PersistentId(Vec<PickleValue>),
    Reduced { callable: Box<PickleValue>, args: Box<PickleValue> },
    Built { obj: Box<PickleValue>, state: Box<PickleValue> },
}
```

The VM has:
- A **value stack** (`Vec<PickleValue>`)
- A **mark stack** (`Vec<usize>`) for `MARK`/`TUPLE`/`SETITEMS`
- A **memo** (`HashMap<u32, PickleValue>`) for `BINPUT`/`BINGET`/`MEMOIZE`

The main loop reads one opcode at a time from a `&[u8]` cursor and dispatches.

**GLOBAL allowlist** (reject anything else with `AnamnesisError::Parse`):

```
torch._utils._rebuild_tensor_v2
torch.FloatStorage
torch.HalfStorage
torch.BFloat16Storage
torch.DoubleStorage
torch.LongStorage
torch.IntStorage
torch.ShortStorage
torch.CharStorage
torch.ByteStorage
torch.BoolStorage
collections.OrderedDict
torch.nn.parameter.Parameter    (may appear in non-state_dict saves)
```

#### A.3 — Tensor Extraction

After the pickle VM produces a top-level `PickleValue` (the state_dict), a
`extract_tensors()` function walks the structure:

1. Find the top-level `Dict` (or `Reduced{OrderedDict, ...}`)
2. Build a **storage cache** (`HashMap<String, Vec<u8>>`) — keyed by the data file
   index string (e.g., `"0"`, `"1"`). Multiple tensors can reference the same
   storage (shared storage), so the cache avoids reading the same ZIP entry twice
   and is required for correct offset slicing
3. For each key-value pair:
   - Key must be `String` → tensor name
   - Value must be `Reduced{_rebuild_tensor_v2, args}` (or `Built` wrapping one)
   - Extract from args: `PersistentId` (data file index + storage type),
     storage offset, shape tuple, stride tuple
4. Load raw bytes from the storage cache, reading from the ZIP archive's
   `data/{index}` entry on first access (lazy-populate the cache)
5. Handle **storage offset** and **strides**:
   - If contiguous (stride matches row-major for shape): slice raw bytes at offset
   - If non-contiguous: copy to contiguous layout (rare for state_dicts but must handle)
6. Handle **byte order**:
   - Read `archive/byteorder` (or default to `"little"`)
   - If big-endian: byte-swap in place using `byteswap_inplace` (see prerequisite below)
7. Return `Vec<PthTensor>`

**Prerequisite — extract `byteswap_inplace` to shared utility:**
`byteswap_inplace` is currently a private `fn` in `src/parse/npz.rs` (line 475).
Before Phase A, extract it to a new `src/parse/utils.rs` module as `pub(crate)`,
and update `npz.rs` to call the shared version. This parallels the existing
`src/remember/quant_utils.rs` pattern. Add to Phase F module wiring.

#### A.4 — Public API

```rust
/// Parse a PyTorch `.pth` state_dict file.
///
/// Returns a map of tensor name → `PthTensor` with raw data in
/// native-endian, row-major layout.
///
/// # Errors
///
/// Returns `AnamnesisError::Parse` if the file is not a valid PyTorch
/// ZIP archive, uses unsupported pickle opcodes, or contains
/// non-allowlisted globals.
///
/// Returns `AnamnesisError::Unsupported` for legacy (pre-1.6) `.pth`
/// files that are raw pickle without ZIP wrapping.
pub fn parse_pth(path: impl AsRef<Path>) -> crate::Result<Vec<PthTensor>>
```

**Note on return type**: `Vec<PthTensor>` (not `HashMap`) to preserve insertion order,
matching PyTorch's `OrderedDict`. Callers can convert to `HashMap` if needed.

#### A.5 — Legacy Format Detection

Pre-PyTorch 1.6 files are raw pickle (not ZIP). Detect by checking the first 4 bytes:
- ZIP magic `PK\x03\x04` → modern format (proceed)
- Pickle `\x80\x02` → legacy format → return `AnamnesisError::Unsupported` with
  message: `"legacy .pth format (pre-PyTorch 1.6) is not supported; re-save with torch.save()"`

This is a deliberate scope cut. Legacy `.pth` files are rare (PyTorch 1.6 was released
in July 2020) and would require a completely different data layout (tensor bytes embedded
in the pickle stream via `BINBYTES`).

---

### Phase B: Safetensors Conversion (`src/remember/pth.rs`)

**New file: `src/remember/pth.rs`** (~80–120 lines)

This is simpler than quantized dequantization — `.pth` tensors are already
full-precision. The "remember" step is just format conversion.

```rust
/// Convert parsed `.pth` tensors to a safetensors file.
///
/// Each tensor is written with its original dtype (no dequantization).
/// Tensor names are preserved as-is from the state_dict keys.
///
/// # Errors
///
/// Returns `AnamnesisError::Io` if the output file cannot be written.
/// Returns `AnamnesisError::Unsupported` if a tensor dtype has no
/// safetensors equivalent.
pub fn pth_to_safetensors(
    tensors: &[PthTensor],
    output: impl AsRef<Path>,
) -> crate::Result<()>
```

Implementation:
1. Build `Vec<(&str, TensorView)>` from `PthTensor` data
2. Map `PthDtype` → `safetensors::Dtype` via `PthDtype::to_dtype()::to_safetensors_dtype()`
3. Call `safetensors::tensor::serialize_to_file(views, &None, output)`

**No dequantization, no dtype conversion.** The output preserves whatever dtype
the `.pth` file contained (typically F32 for AlgZoo, but BF16/F16/F64 are passed through).

---

### Phase C: Integration into `model.rs`

#### C.1 — Separate `ParsedPth` Type (Option A)

**Design decision**: `.pth` parsing returns a different internal representation than
safetensors (no `SafetensorsHeader`). The `.pth` workflow is fundamentally different —
there is no dequantization step, only format conversion. A separate `ParsedPth` type
with `to_safetensors()` is cleaner than making `ParsedModel` an enum.

**`parse()` is NOT extended.** It remains safetensors-only (`-> Result<ParsedModel>`).
The `.pth` path has its own dedicated entry point `parse_pth_model()`. Format dispatch
happens in the CLI (Phase D), not in the library.

```rust
pub struct ParsedPth {
    tensors: Vec<PthTensor>,
}

impl ParsedPth {
    /// Inspect the parsed `.pth` file.
    pub fn inspect(&self) -> PthInspectInfo { ... }

    /// Convert to safetensors format.
    ///
    /// # Errors
    ///
    /// Returns `AnamnesisError::Io` if the output path is not writable.
    pub fn to_safetensors(&self, output: impl AsRef<Path>) -> crate::Result<()> {
        pth_to_safetensors(&self.tensors, output)
    }
}
```

#### C.2 — New Public Entry Point

```rust
#[cfg(feature = "pth")]
pub fn parse_pth_model(path: impl AsRef<Path>) -> crate::Result<ParsedPth>
```

#### C.3 — Inspect Info

```rust
pub struct PthInspectInfo {
    /// Number of tensors in the state_dict.
    pub tensor_count: usize,
    /// Total size of raw tensor data in bytes.
    pub total_bytes: u64,
    /// Distinct dtypes found.
    pub dtypes: Vec<PthDtype>,
    /// Whether the file uses big-endian storage.
    pub big_endian: bool,
}
```

---

### Phase D: CLI Integration (`src/bin/main.rs`)

Extend existing subcommands to accept `.pth` files. Format dispatch lives here,
not in the library. Detect by extension (`.pth`, `.pt`) and fall back to ZIP magic
bytes for `.bin` files:

```rust
fn detect_format(path: &Path) -> Format {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "safetensors" => Format::Safetensors,
        "pth" | "pt" => Format::Pth,
        "bin" => {
            // HuggingFace uses .bin for PyTorch models — check ZIP magic
            if has_zip_magic(path) { Format::Pth } else { Format::Safetensors }
        }
        _ => Format::Safetensors, // default fallback
    }
}
```

This resolves the `.bin` extension ambiguity noted in the Risk Assessment by
using magic-byte detection rather than extension alone.

#### D.1 — `amn parse model.pth`

```
Parsed model.pth (PyTorch state_dict)
  Tensors:      6
  Total size:   1,728 bytes
  Dtypes:       F32
  Byte order:   little-endian

  rnn.weight_ih_l0            F32    [16, 1]       64 B
  rnn.weight_hh_l0            F32    [16, 16]   1,024 B
  linear.weight               F32    [10, 16]     640 B
```

#### D.2 — `amn inspect model.pth`

```
Format:           PyTorch state_dict (.pth)
Tensors:          6
Total size:       1,728 bytes
Safetensors size: 1,856 bytes (header overhead)
Byte order:       little-endian
Dtypes:           F32
```

#### D.3 — `amn remember model.pth --to safetensors`

```
Converting model.pth → model.safetensors
  6 tensors, 1,728 bytes
  Done in 0.2ms
```

**Output path derivation**: `model.pth` → `model.safetensors` (replace extension).

---

### Phase E: Testing

#### E.1 — Fixture Generation (Python Script)

Create `tests/fixtures/generate_pth.py`:

```python
"""Generate .pth test fixtures for anamnesis cross-validation."""
import torch
from collections import OrderedDict

# Fixture 1: Simple state_dict (F32, small)
state = OrderedDict([
    ("weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
    ("bias", torch.tensor([0.5, -0.5])),
])
torch.save(state, "tests/fixtures/simple_f32.pth")

# Fixture 2: Mixed dtypes
state = OrderedDict([
    ("f32_weight", torch.randn(4, 4)),
    ("f16_weight", torch.randn(4, 4).half()),
    ("bf16_weight", torch.randn(4, 4).bfloat16()),
    ("i64_index", torch.tensor([0, 1, 2, 3], dtype=torch.long)),
])
torch.save(state, "tests/fixtures/mixed_dtypes.pth")

# Fixture 3: AlgZoo-like RNN (matches M_2_2 structure)
state = OrderedDict([
    ("rnn.weight_ih_l0", torch.randn(2, 1)),
    ("rnn.weight_hh_l0", torch.randn(2, 2)),
    ("linear.weight", torch.randn(2, 2)),
])
torch.save(state, "tests/fixtures/rnn_like.pth")

# Also save as safetensors for cross-validation
from safetensors.torch import save_file
save_file(dict(state), "tests/fixtures/rnn_like_reference.safetensors")
```

Commit the generated `.pth` fixtures (they're tiny — under 10 KB total).

#### E.2 — Unit Tests (`tests/pth_parsing.rs`)

```rust
#[cfg(feature = "pth")]
mod pth_tests {
    #[test]
    fn parse_simple_f32() { ... }        // 2 tensors, correct names/shapes/dtypes

    #[test]
    fn parse_mixed_dtypes() { ... }      // F32 + F16 + BF16 + I64

    #[test]
    fn parse_rnn_like() { ... }          // AlgZoo structure

    #[test]
    fn roundtrip_to_safetensors() { ... } // .pth → .safetensors, compare against reference

    #[test]
    fn reject_legacy_format() { ... }    // Pre-1.6 .pth → AnamnesisError::Unsupported

    #[test]
    fn reject_malicious_global() { ... } // Pickle with os.system → AnamnesisError::Parse

    #[test]
    fn byte_exact_values() { ... }       // Load .pth, check specific tensor values match Python
}
```

#### E.3 — Cross-Validation Against Python

For the roundtrip test:
1. Load `rnn_like.pth` with `parse_pth()`
2. Convert to safetensors with `pth_to_safetensors()`
3. Load both the converted safetensors AND `rnn_like_reference.safetensors`
4. Compare tensor data byte-for-byte (0 ULP difference expected — no numeric conversion)

#### E.4 — Security Tests

```rust
#[test]
fn reject_exec_payload() {
    // Craft a pickle that calls os.system("rm -rf /")
    // Verify parse_pth returns AnamnesisError::Parse
}

#[test]
fn reject_unknown_global() {
    // Pickle with numpy.core.multiarray.scalar → reject
}
```

---

### Phase F: Module Wiring

#### F.1 — `Cargo.toml`

```toml
[features]
pth = ["dep:zip"]   # Reuse the zip dependency already used by npz
```

Note: `pth` and `npz` both need `zip`. If both are enabled, no duplicate dependency.
If only `pth` is enabled (not `npz`), `zip` still gets pulled in. This is fine.

#### F.2 — `src/parse/mod.rs`

```rust
#[cfg(feature = "pth")]
pub mod pth;

#[cfg(feature = "pth")]
pub use pth::{parse_pth, PthDtype, PthTensor};
```

#### F.3 — `src/remember/mod.rs`

```rust
#[cfg(feature = "pth")]
pub mod pth;

#[cfg(feature = "pth")]
pub use pth::pth_to_safetensors;
```

#### F.4 — `src/lib.rs`

```rust
#[cfg(feature = "pth")]
pub use parse::{parse_pth, PthDtype, PthTensor};

#[cfg(feature = "pth")]
pub use remember::pth_to_safetensors;

// If ParsedPth is added to model.rs:
#[cfg(feature = "pth")]
pub use model::{ParsedPth, PthInspectInfo};
```

#### F.5 — Error Variants

No new error variants needed. Existing variants cover all cases:
- `AnamnesisError::Parse` — malformed pickle, bad opcodes, disallowed globals
- `AnamnesisError::Unsupported` — legacy format, unsupported storage types
- `AnamnesisError::Io` — file read/write failures

**However**, the `From<zip::result::ZipError>` impl in `src/error.rs` is currently
gated on `#[cfg(feature = "npz")]`. Since `pth` also uses the `zip` crate, this
cfg must be widened to `#[cfg(any(feature = "npz", feature = "pth"))]`:

```rust
#[cfg(any(feature = "npz", feature = "pth"))]
impl From<zip::result::ZipError> for AnamnesisError {
    fn from(e: zip::result::ZipError) -> Self {
        Self::Parse {
            reason: format!("failed to read ZIP archive: {e}"),
        }
    }
}
```

---

### Phase G: Documentation & Changelog

#### G.1 — CHANGELOG.md

```markdown
## [0.3.1] - 2026-0X-XX

### Added
- **PyTorch `.pth` parsing** (`src/parse/pth.rs`) — parse PyTorch state_dict
  ZIP archives (PyTorch ≥ 1.6) with a minimal, safe pickle interpreter that
  rejects non-allowlisted globals. Supports F16, BF16, F32, F64, I8–I64, U8,
  Bool storage types. Feature-gated behind `pth`.
- **`.pth` → safetensors conversion** (`src/remember/pth.rs`) — lossless format
  conversion preserving original dtypes. No dequantization needed.
- **CLI support** — `amn parse`, `amn inspect`, and `amn remember` now accept
  `.pth` files when built with `--features pth`.
- **Security boundary** — pickle parser allowlists only `torch._utils`,
  `torch.*Storage`, and `collections.OrderedDict` globals; rejects all others
  including `os.system`, `subprocess`, `builtins`.
- **Cross-validation** — byte-exact roundtrip against Python-generated
  safetensors reference fixtures.
```

#### G.2 — README.md

Add `.pth` to the supported formats table. Add usage example:

```bash
# Parse a PyTorch state_dict
amn parse model.pth

# Convert to safetensors
amn remember model.pth --to safetensors
```

---

## File Summary

| File | Action | Lines (est.) |
|------|--------|-------------|
| `src/parse/utils.rs` | **Create** | ~10 (extract `byteswap_inplace` from `npz.rs`) |
| `src/parse/npz.rs` | Edit | ~2 (use shared `byteswap_inplace`) |
| `src/parse/pth.rs` | **Create** | 300–400 |
| `src/parse/mod.rs` | Edit | +6 |
| `src/remember/pth.rs` | **Create** | 80–120 |
| `src/remember/mod.rs` | Edit | +4 |
| `src/model.rs` | Edit | +40 (ParsedPth, PthInspectInfo) |
| `src/error.rs` | Edit | ~1 (widen `ZipError` cfg to `any(npz, pth)`) |
| `src/lib.rs` | Edit | +6 |
| `src/bin/main.rs` | Edit | +30 (CLI dispatch) |
| `Cargo.toml` | Edit | +1 (feature) |
| `tests/pth_parsing.rs` | **Create** | 150–200 |
| `tests/fixtures/generate_pth.py` | **Create** | 30 |
| `tests/fixtures/*.pth` | **Create** | (binary, <10 KB) |
| `tests/fixtures/*.safetensors` | **Create** | (binary, <10 KB) |
| `CHANGELOG.md` | Edit | +10 |
| `README.md` | Edit | +10 |

**Total new Rust code**: ~530–720 lines + ~150–200 lines of tests.

---

## Implementation Order

1. **`src/parse/utils.rs`** — extract `byteswap_inplace` from `npz.rs`, update `npz.rs` to use it
2. **`src/parse/pth.rs`** — pickle VM + tensor extraction (the core work)
2. **`tests/fixtures/generate_pth.py`** — generate fixtures, commit binaries
3. **`tests/pth_parsing.rs`** — basic parsing tests (parse, reject legacy, reject malicious)
5. **`src/remember/pth.rs`** — safetensors writer (simple)
6. **`tests/pth_parsing.rs`** — roundtrip cross-validation test
7. **`src/model.rs`** — `ParsedPth`, `PthInspectInfo`
8. **`src/bin/main.rs`** — CLI integration (format dispatch with `detect_format`)
9. **Module wiring** — `mod.rs`, `lib.rs`, `Cargo.toml`
10. **Pre-commit checks** — `cargo fmt`, `cargo clippy --features pth -- -W clippy::pedantic`, `cargo test --features pth`
11. **CHANGELOG.md**, **README.md**

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Pickle parser scope | Minimal (~36 opcodes) | Full pickle is thousands of lines; state_dicts use a tiny subset |
| Security model | Explicit GLOBAL allowlist | Equivalent to `weights_only=True` but stricter; no arbitrary code execution |
| Legacy `.pth` support | Reject with clear error | Pre-1.6 format is rare (2020+) and structurally different |
| Return type | `Vec<PthTensor>` | Preserves OrderedDict insertion order |
| Conversion target | Safetensors only | No dtype conversion needed; `.pth` tensors are already full-precision |
| `ParsedPth` vs enum `ParsedModel` | Separate type; `parse()` unchanged | `.pth` has no dequantization step; different workflow from quantized safetensors. Format dispatch lives in the CLI, not the library |
| Feature gate | `pth = ["dep:zip"]` | Reuses `zip` from `npz`; zero new dependencies if `npz` already enabled |
| Non-contiguous tensors | Copy to contiguous | Rare in state_dicts but must handle for correctness |
| Big-endian support | Byte-swap on load | Extract `byteswap_inplace` from NPZ to `parse/utils.rs`; big-endian `.pth` files exist but are rare |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Pickle protocol variation across PyTorch versions | Medium | Test against PyTorch 1.6, 2.0, 2.4+ fixtures; handle protocols 2–5 |
| Non-state_dict `.pth` files (full model saves) | Low | Document scope: state_dict only. Reject `nn.Module` pickles gracefully |
| Shared storage (multiple tensors referencing same data file) | Medium | Track storage index → loaded data; slice per tensor's offset/size |
| Complex strides (transposed views saved as state_dict) | Low | Detect non-contiguous strides; copy to contiguous layout |
| `.bin` extension ambiguity (HF uses `.bin` for PyTorch) | Low | Detect by ZIP magic, not extension alone |

---

## Out of Scope (Future Work)

- **Loading from GCS URLs** — download is hf-fetch-model's concern, not anamnesis
- **Full model saves** (`torch.save(model, path)`) — requires reconstructing `nn.Module` graphs
- **TorchScript archives** (`.pt` with `code/` directory) — different format entirely
- **Quantized `.pth` files** (rare; most quantized models use safetensors or GGUF)
- **Streaming/lazy parsing** — not needed for the target model sizes (< 100 MB)
