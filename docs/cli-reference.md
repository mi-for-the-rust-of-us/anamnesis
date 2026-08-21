# CLI reference

<!-- Last updated: 2026-08-15, anamnesis v0.7.4 -->

Every subcommand, flag, and output shape for the `anamnesis` / `amn` CLI. The
[README](../README.md) has the quick tour; this is the complete reference.

## Install

```sh
cargo install anamnesis --features cli,pth,gguf
```

Installs two binaries: **`anamnesis`** and **`amn`** (a short alias — identical
behaviour). Add features for the formats and capabilities you need:

| Feature | Enables |
|---|---|
| `cli` | the CLI itself (pulls in `clap`) — **required** for the binaries |
| `pth` | PyTorch `.pth` / `.pt` parsing & conversion |
| `gguf` | GGUF parsing, dequantization & writing |
| `npz` | NumPy `.npz` parsing & conversion |
| `gptq` / `awq` / `bnb` | GPTQ / AWQ / BitsAndBytes dequantization (and BnB-NF4 encode for `bnb`) |
| `ollama` | the `ollama:` URL scheme (see below) |
| `indicatif` | a progress bar during `remember` |

`amn --version` and `amn --help` (and `amn <command> --help`) are always
available.

## Commands at a glance

| Command | Description |
|---|---|
| `amn parse <file>` | Parse and summarize a model file (`.safetensors`, `.pth`/`.pt`, `.npz`, `.gguf`, `.bin`) |
| `amn inspect <file>` *(alias `info`)* | Show format, tensor counts, size estimates, dtypes, byte order |
| `amn remember <file>` *(alias `dequantize`)* | Dequantize to safetensors at `bf16` (default), `f32`, or `f16`, or convert `.pth`/`.gguf` → `.safetensors` |
| `amn convert <file> --to <target>` | Convert any input to `safetensors` / `gguf` / `bnb-nf4` through one dispatch |

Format detection is automatic (see [Format detection](#format-detection)).

---

## `amn parse <file>`

Parse the file and print a per-format summary, including a per-tensor table.

| Argument | Description |
|---|---|
| `<file>` | Path to the model file, or an `ollama:` URL |

Example:

```
$ amn parse model.pth
Parsed model.pth (PyTorch state_dict)
  Tensors:    3
  Total size: 1.7 KB
  Dtypes:     F32
  Byte order: little-endian

  rnn.weight_ih_l0               F32 [16, 1]         64 B
  rnn.weight_hh_l0               F32 [16, 16]        1.0 KB
  linear.weight                  F32 [10, 16]        640 B
```

For safetensors, the summary breaks tensors down by role (quantized / scale /
zero-point / `g_idx` / passthrough) and the detected quantization scheme.

## `amn inspect <file>` (alias `info`)

Header-only summary — format, tensor count, total size, dtypes, byte order /
alignment — without materialising tensor data.

| Argument | Description |
|---|---|
| `<file>` | Path to the model file, or an `ollama:` URL |

| Flag | Default | Description |
|---|---|---|
| `--to <value>` | `bf16` | Width the dequantised-size estimate assumes: `bf16`, `f32`, `f16`. Added in **v0.7.6**; before that the estimate was pinned to `bf16`, so a caller who intended `--to f32` on `remember` was reading a figure half the size of what they were about to allocate. Vacuous on `.npz` / `.pth`, whose tensors are already full precision. |

```
$ amn inspect weights.npz
Format:      NPZ archive
Tensors:     5
Total size:  160 B
Dtypes:      F32
```

`amn info` is an exact alias.

## `amn remember <file>` (alias `dequantize`)

Recover precision: dequantize a quantized safetensors, or convert a
`.pth` / `.gguf` to safetensors (dequantizing any quantized GGUF tensors,
passing scalar tensors through).

| Flag | Default | Description |
|---|---|---|
| `--to <value>` | `bf16` | Output dtype for **dequantised** tensors: `bf16`, `f32`, `f16`. `safetensors` is accepted as an alias for `bf16` on `.pth` / `.gguf` inputs (they always produce safetensors). See [Output dtype on `remember`](#output-dtype-on-remember). |
| `--output`, `-o <path>` | *(derived)* | Output path; derived from the input if omitted (see [Output paths](#output-path-derivation)). |
| `--threads <N>` | `min(cores, 4)` | Dequantisation worker threads (see [Threads](#threads)). |

```
$ amn remember model.pth
Converting model.pth → model.safetensors
  3 tensors, 1.7 KB
  Done.
```

`amn dequantize` is an exact alias. An `.npz` input is **accepted** since
**v0.7.6** and copied through into safetensors, producing exactly what
`amn convert weights.npz --to safetensors` produces; before that it was rejected
as "already full precision", which made one verb mean different things on
different formats. As on `.pth`, `--to` is accepted and changes nothing there:
nothing in an `NPZ` is quantised, so there is nothing to narrow or widen.

### Output dtype on `remember`

Widened in **v0.7.4**. Before that `remember` was `bf16`-only, because its four
kernel families (`FP8`, `GPTQ`, `AWQ`, `BnB`) fused the narrowing step into
their inner loops; those loops are now generic over the output width.

```console
$ amn remember model-fp8.safetensors --to f32
```

The flag keeps its name here rather than becoming `--out-dtype`, because on
`remember` there is no output *format* to choose: the answer is always
safetensors. On `convert`, where `--to` already selects the format, the same
choice is spelled [`--out-dtype`](#output-dtype).

The semantics are identical to `convert`'s, and so are the trade-offs:

- **`f32`** removes anamnesis's own narrowing step, so you get the reference
  implementation's own `f32`. Doubles the output bytes and runs slower on a
  bandwidth-bound path.
- **`f16`** buys 3 significand bits over `bf16` and pays twice for them. First a
  far narrower exponent range (overflow to infinity above 65504, flush to zero
  below about `2⁻²⁴`); plain IEEE semantics, never saturation. Second **2x to 3x
  the run time of `bf16` at identical output size** — slower even than `f32`,
  which writes twice the bytes, because the cost is the conversion rather than
  the traffic.
- **It governs dequantised tensors only.** Passthrough tensors keep their source
  dtype, so the output is legitimately mixed-dtype.
- On a `.pth` input nothing is dequantised, so the value is accepted and inert.

The decision, with the measured numbers behind it, is written out in
[Choosing an output dtype](tutorials/choosing-an-output-dtype.md).

## `amn convert <file> --to <target>`

Convert any supported input to a different format through a single dispatch.
Every `(input × target)` pair routes through an in-memory hub (Phase 6.14,
v0.6.9): the input is normalised to the hub — quantised tensors dequantised,
scalar tensors kept in their original dtype — then written to the target.
Quantised inputs **auto-chain** (no hand-staged temp file).

The hub's dequantised width was `BF16` unconditionally until v0.7.3; it is now
whatever [`--out-dtype`](#output-dtype) selects, defaulting to `BF16`.

| Flag | Description |
|---|---|
| `--to <target>` | **Required.** One of `safetensors` (alias `bf16`), `gguf`, `bnb-nf4` (aliases `bnb_nf4` / `nf4`). Case-insensitive. |
| `--output`, `-o <path>` | Output path; derived from the input if omitted. |
| `--gguf-metadata <FILE>` | JSON `GGUF` key/values to stamp on a `gguf` target (see [GGUF metadata](#gguf-metadata-flags)). |
| `--gguf-kv <KEY=VALUE>` | Repeatable one-off `GGUF` metadata (string-valued). |
| `--out-dtype <DTYPE>` | Element type for **dequantised** tensors: `bf16` (default), `f32`, `f16`. Every dequantising input since v0.7.4 (see [Output dtype](#output-dtype)). |
| `--threads <N>` | Dequantisation worker threads; defaults to `min(cores, 4)` (see [Threads](#threads)). |

### Output dtype

Added in **v0.7.3**. Named `--out-dtype` rather than reusing `--to`, because on
`convert` the `--to` flag already selects the output *format*.

```console
$ amn convert model.gguf --to safetensors --out-dtype f32
Converting model.gguf -> model-f32.safetensors
  201 dequantized to F32
```

- **`f32`** removes anamnesis's own narrowing step, so the values you get are the
  `f32` that `gguf-py` itself produces. Expect it to be **slower**, not faster: it
  doubles the output bytes on a bandwidth-bound path (measured 1.54–1.61×
  end to end). That is the honest cost of the precision.
- **`f16`** buys 3 significand bits over `bf16` and pays a far narrower exponent
  range: it overflows to infinity above 65504 and flushes to zero below about
  `2⁻²⁴`, where `bf16` shares `f32`'s range. anamnesis follows plain IEEE
  semantics rather than saturating, so its output matches NumPy and PyTorch.
  It is also **2x to 3x slower than `bf16` at the same output size**, and so
  slower than `f32` despite writing half the bytes: the cost is the conversion,
  not the write. Measured per kernel in
  [Choosing an output dtype](tutorials/choosing-an-output-dtype.md).
- The derived output filename tracks the dtype (`model-f32.safetensors`), so a
  file never claims a width it does not hold.

**It governs dequantised tensors only.** Passthrough tensors (norms, biases,
anything not block-quantised) keep their source dtype, so `--out-dtype f32` is
not "rewrite every tensor as f32". Widening a tensor that was merely copied
would invent precision that was never in the file.

**Every dequantising input, since v0.7.4.** `GGUF` was the only one in v0.7.3,
because the quantised-*safetensors* kernels (`FP8` / `GPTQ` / `AWQ` / `BnB`)
fused the narrowing into their hot loops; they are now generic over the output
width, so a quantised safetensors input honours all three dtypes too. `NPZ` and
`.pth` dequantise nothing, so the flag is accepted and inert there. The same
choice is available on `remember` as
[`--to`](#output-dtype-on-remember).

### Conversion matrix (v0.6.9)

| Input ↓ \ Target → | `safetensors` / `bf16` | `gguf` | `bnb-nf4` |
|---|---|---|---|
| **safetensors** | ✅ dequant to `bf16` / `f32` / `f16`⁴ or lossless passthrough | ✅¹ | ✅² |
| **`.pth`** | ✅ | ✅¹ | ✅² |
| **`.npz`** | ✅ | ✅¹ | ✅² |
| **`.gguf`** | ✅ dequant to `bf16` / `f32` / `f16`⁴ | ✅ dequant-in-place³ | ✅² |

Every current-target cell is wired. ¹ `gguf` target requires the `gguf` feature;
writes an **unquantised** (scalar) GGUF — quantised GGUF emit (`gguf-q4km`, …)
needs the Phase 8.5 encode kernels. A quantised safetensors / GGUF source is
**dequantised automatically** through the hub (the old "dequantise first" error is
gone).
² `bnb-nf4` requires the `bnb` feature; 2-D float tensors (≥64 elements, a
multiple of 64) are encoded to NF4, everything else passes through as `BF16`.
³ `gguf → gguf` recovers precision and re-emits a scalar GGUF, **preserving the
source's metadata KV** (architecture, tokenizer) so the result stays loadable;
`--gguf-metadata` / `--gguf-kv` merge over it.
⁴ `--out-dtype` selects the dequantised element type; see [Output
dtype](#output-dtype). `GGUF` since v0.7.3, quantised safetensors since v0.7.4.
`.pth` and `.npz` dequantise nothing, so the flag is inert rather than refused.

Still out of scope until Phase 8.5: **quantised GGUF target columns**
(`gguf-q4km`, FP8, IQ, TQ, MXFP4). A combination whose Cargo feature is disabled
returns a clear `AnamnesisError::Unsupported` naming the feature to rebuild with.

```
$ amn convert model.npz --to safetensors
Converting model.npz -> model-bf16.safetensors
  Wrote 5 tensors -> model-bf16.safetensors

$ amn convert model-fp8.safetensors --to gguf   # quantised -> auto-chains through BF16
Converting model-fp8.safetensors -> model-gguf.gguf
  1 dequantized to BF16
  Wrote 2 tensors -> model-gguf.gguf
```

### GGUF metadata flags

`--gguf-metadata` / `--gguf-kv` supply the key/value table a `gguf` target
carries. anamnesis writes the KV **verbatim** — it attaches no meaning to keys and
derives nothing from the tensors; producing model-correct KV (architecture
hyper-parameters, the tokenizer arrays) from a source `config.json` /
`tokenizer.json` is a packaging concern for a downstream tool.

Precedence, lowest to highest: **inherited source KV** (a `GGUF` input) →
**`--gguf-metadata` file** → **`--gguf-kv`**.

- **`--gguf-kv key=value`** (repeatable) — always writes a `String`. Split on the
  first `=`, so the value may contain `=`.
- **`--gguf-metadata <FILE>`** — a JSON object. Each value is either a *plain* JSON
  value (type inferred) or an *explicit* `{"type": …, "value": …}` object:

  | JSON | `GGUF` type |
  |---|---|
  | `"llama"` / `true` | `String` / `Bool` |
  | `32` (non-negative, fits `u32`) | `U32` |
  | `-5` / a larger integer | `I64` / `U64` |
  | `1e-5` | `F32` |
  | `["a", "b"]` | `Array<String>` (typed from the first element) |
  | `{"type": "i32", "value": 3}` | `I32` (any of `u8` `i8` `u16` `i16` `u32` `i32` `u64` `i64` `f32` `f64` `bool` `string`) |
  | `{"type": "array", "item_type": "i32", "value": [1, 2]}` | `Array<I32>` |

  The explicit form exists because inference cannot be right for every key —
  `tokenizer.ggml.token_type` is `Array<I32>` by `llama.cpp` convention, but a JSON
  array of non-negative integers infers `Array<U32>`. State the type rather than
  have anamnesis special-case a key name.

```
$ amn convert model.safetensors --to gguf \
    --gguf-metadata model-kv.json \
    --gguf-kv general.name=my-model
```

The flags need the `gguf` feature; on a build without it, supplying either is a
clear `Unsupported` error rather than a silent no-op.

---

## Threads

`--threads N` sets how many worker threads dequantise tensors, on both
`amn remember` and `amn convert` (added in v0.7.2 — before that the budget was a
library-only knob).

**One correction worth knowing if you benchmarked an older build.** On a `GGUF`
input, `amn remember` accepted `--threads` and then *ignored* it from v0.7.2
until **v0.7.6**: that arm ran its own sequential copy of the conversion rather
than the library's threaded one. If you measured no effect there, the flag was
the problem, not your machine. `amn convert` was threaded throughout.

| Value | Effect |
|---|---|
| *(omitted)* | `min(cpu cores, 4)` — the measured scaling knee, leaving the rest of the machine free. |
| `1` | Fully sequential. |
| `N > 1` | Up to `N` workers. Useful on a many-channel-memory host, where the default is conservative. |
| `0` | Clamped to `1`. |

Three things worth knowing:

- **The output is byte-identical whatever you pass.** Thread count is a
  performance knob, never a correctness variable — this is asserted across
  `{1, 2, 4, 8, 16}` and the resolved default in the test suite.
- **The default is deliberately modest.** Dequantisation is memory-bandwidth
  bound, so throughput plateaus at roughly 3–4× by about four threads; grabbing
  every core buys little and harms whatever else the machine is doing. Raise it
  if you have the memory bandwidth to feed it.
- **Small models ignore it.** Below an internal 4 `MiB` floor the sequential path
  runs regardless, because spawning a pool costs more than the work saves.

The flag has no effect in a build made with `--no-default-features` (the
`parallel` Cargo feature off), which is always sequential.

## Output path derivation

When `--output` / `-o` is omitted, the output path is derived from the input:
a known quantization suffix is stripped from the stem, then `-{target}.{ext}` is
appended.

- `remember`: → `<stem>-bf16.safetensors`
- `convert`: → `<stem>-{bf16|f32|f16|gguf|bnb-nf4}.{safetensors|gguf}`. For a
  `safetensors` target the suffix **follows `--out-dtype`**, so the filename
  never claims a width the file does not hold. The `gguf` and `bnb-nf4` suffixes
  name a container or an encoding rather than an element type, so they are
  unaffected by it.

Stripped suffixes (case-sensitive, longest-first) include `-fp8`, `-GPTQ-Int4`,
`-gptq`, `-AWQ`, `-awq`, `-bnb-4bit`, `-bnb-int8`, `-4bit`, `-int8`, … — e.g.
`model-GPTQ-Int4.safetensors` → `model-bf16.safetensors`,
`weights-fp8.safetensors --to gguf` → `weights-gguf.gguf`.

## `ollama:` URL scheme

Build with the `ollama` feature (`cargo install anamnesis --features cli,gguf,ollama`)
and **every** subcommand accepts an `ollama:` URL in place of a file path:

```
$ amn inspect ollama:llama3.2:1b
Format:      GGUF v3
Arch:        llama
Tensors:     147
Total size:  1.22 GB
Dequantized: 2.30 GB (BF16)
Dtypes:      F32, Q8_0
Alignment:   32 bytes
```

- `ollama:<model>:<tag>` — resolves the manifest at
  `~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>` to its
  model-layer GGUF blob under `~/.ollama/models/blobs/sha256-<hash>`.
- `ollama:<model>` — same, with the tag defaulting to `latest`.

Pure path arithmetic plus a single JSON read — no `ollama` CLI shell-out, no Go
interop. Honours the `OLLAMA_MODELS` environment variable for non-default cache
locations. Without the `ollama` feature, an `ollama:` input returns a clear
`Unsupported` error.

## Format detection

Detection is automatic, by extension then magic bytes:

- `.safetensors` → safetensors
- `.pth` / `.pt` → PyTorch pickle (`pth` feature)
- `.npz` → NumPy NPZ (`npz` feature)
- `.gguf` → GGUF (`gguf` feature)
- `.bin` → probed for ZIP magic (`PK\x03\x04` → PyTorch) then GGUF magic, else safetensors
- any other extension → probed for GGUF magic, else safetensors

If the input matches a format whose Cargo feature is **not** enabled, the CLI
returns an `Unsupported` error naming the feature to rebuild with (e.g. a `.gguf`
file in a build without `gguf`) — never a cryptic downstream failure from
misrouting to the safetensors parser.

## Exit codes

`0` on success; `1` on any error (the underlying `AnamnesisError` message is
printed to stderr). For the error *kinds* and how a host can branch on them, see
the [README error taxonomy](../README.md#parsing-untrusted-input).
