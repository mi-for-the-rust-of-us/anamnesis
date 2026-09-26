# anamnesis — Flagship Example (V2)

> **HISTORICAL (archived at the v0.7.8 close-out).** The flagship example the Phase 1 API was designed against. The shipped API and CLI are documented in `README.md` and `docs/cli-reference.md`.

**The problem in one sentence:** Mistral ships Ministral 3B Instruct as FP8 safetensors. No Rust ML framework can load it.

*Note: Tensor counts and sizes in CLI output examples below are illustrative.
Actual values will be confirmed when the test models are downloaded in Phase 1.*

---

## Before anamnesis

```rust
// candle
let vb = VarBuilder::from_mmaped_safetensors(&[path], DType::BF16, &device)?;
// -> Error: unsupported safetensor dtype F8_E4M3
```

This blocks the entire Rust ML ecosystem: candle, candle-mi, burn, tch. The only workaround is a Python script.

## After anamnesis

### Step 1 — Parse (make contact with the weights)

Everything begins with parsing. You cannot remember what you have not first parsed.

```bash
$ amn parse model-fp8.safetensors

277 tensors parsed
  224 quantized   FP8 E4M3, per-tensor scale
   53 passthrough BF16 (norms, embeddings, lm_head)
File: 4.35 GB
```

### Step 2 — Inspect (what did Lethe take?)

```bash
$ amn inspect model-fp8.safetensors

Format:     Per-tensor FP8 (E4M3), single scale factor per tensor
Quantized:  224 tensors (weights)
Passthrough: 53 tensors (norms, embeddings, lm_head)
Size:       4.35 GB (FP8) -> 7.70 GB (BF16)
Lethe took: ~3.35 GB of precision
```

### Step 3 — Remember (dequantize)

```bash
$ amn remember model-fp8.safetensors --to bf16

Parsing...  277 tensors, per-tensor FP8 (E4M3)
Recalling... 224 tensors [====================] 2.1s
Output: model-bf16.safetensors (7.70 GB)
```

### Step 4 — Use anywhere

```rust
// candle — now works
let vb = VarBuilder::from_mmaped_safetensors(&["model-bf16.safetensors"], DType::BF16, &device)?;

// candle-mi — logit lens, circuit tracing, CLT analysis
let model = MIModel::from_pretrained("./model-bf16/")?;
```

---

## Fine-grained FP8 (128x128 block scales)

Not all FP8 is the same. EXAONE-4.0-1.2B-FP8 uses fine-grained quantization
with 128x128 block scale factors — more precision, more metadata:

```bash
$ amn inspect exaone-fp8.safetensors

Format:     Fine-grained FP8 (E4M3), 128x128 blocks
Quantized:  180 tensors (weights) + 180 scale tensors (F32)
Passthrough: 31 tensors (norms, embeddings)
Size:       1.22 GB (FP8) -> 2.40 GB (BF16)
Lethe took: ~1.18 GB of precision

$ amn remember exaone-fp8.safetensors --to bf16

Parsing...  391 tensors, fine-grained FP8 (E4M3), 128x128 blocks
Recalling... 180 tensors [====================] 0.4s
Output: exaone-bf16.safetensors (2.40 GB)
```

---

## NPZ parsing (nothing was forgotten)

SAE weights (Gemma Scope) ship as NPZ archives — a foreign container, not a
quantized format. Parsing alone suffices. There is nothing to remember because
nothing was forgotten:

```bash
$ amn parse params.npz

5 tensors parsed
  W_dec    F32 [16384, 2304]
  W_enc    F32 [2304, 16384]
  b_dec    F32 [2304]
  b_enc    F32 [16384]
  threshold F32 [16384]
File: 302 MB
```

No `remember` step needed — the data is already full-precision. anamnesis
extracted it from the NPZ container; any Rust framework can now consume the
raw arrays.

---

## The same in Rust (library API)

The library API reflects the parse-first architecture: parse once, then act.

```rust
use anamnesis::{parse, TargetDtype};

// Parse — the foundation of every operation
let model = parse("model-fp8.safetensors")?;

// Inspect — built on parse, no re-read
let info = model.inspect();
println!("{}", info.format);       // "Per-tensor FP8 (E4M3)"
println!("{}", info.quantized);    // 224
println!("{}", info.passthrough);  // 53

// Remember — built on parse, no re-read
model.remember("model-bf16.safetensors", TargetDtype::BF16)?;
// Output: standard BF16 safetensors, loadable by any Rust ML framework.
```

### NPZ in Rust

```rust
use anamnesis::parse_npz;

// Parse — nothing to remember, the data is intact
let tensors = parse_npz("params.npz")?;
let w_enc = &tensors["W_enc"];
assert_eq!(w_enc.shape, &[2304, 16384]);
// w_enc.data is raw F32 bytes, ready for any framework
```

---

## The opposite direction (future: lethe module)

```rust
use anamnesis::lethe::{QuantScheme};

let model = anamnesis::parse("model-bf16.safetensors")?;

// The weights forget some precision
model.forget("model-fp8.safetensors", QuantScheme::FineGrainedFp8)?;
```

```bash
$ amn forget model-bf16.safetensors --scheme fp8-fine-grained

Parsing...  277 tensors, BF16
Forgetting... 224 tensors [====================] 1.8s
Output: model-fp8.safetensors (4.35 GB)
Lethe distance: 0.0012 mean absolute error
```

---

## CLI verbs and synonyms

Every verb has a technical synonym. Use whichever vocabulary you prefer:

| Verb | Synonym | Operation | Metaphor |
|---|---|---|---|
| `amn parse` | — | Decode + validate format | Make contact with the weights |
| `amn inspect` | `amn info` | Show format, sizes, what Lethe took | What was forgotten? |
| `amn remember` | `amn dequantize` | FP8/GPTQ/AWQ -> BF16/F16/F32 | Anamnesis — recollection of the forms |
| `amn forget` | `amn quantize` | BF16/F32 -> FP8/INT8/INT4 | Lethe — drinking from the river |

## Integration with hf-fetch-model

```bash
# Download + remember in one step
$ hf-fm mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors --dequantize bf16
```

hf-fetch-model calls anamnesis under the hood. The user never touches FP8 files.
