# anamnesis — Flagship Example

> **HISTORICAL (archived at the v0.7.8 close-out).** The first flagship-example draft, superseded by `amn-flagship-v2.md` before Phase 1.

**The problem in one sentence:** Mistral ships Ministral 3 3B Instruct as FP8 safetensors. No Rust ML framework can load it.

---

## Before anamnesis

```rust
// candle
let vb = VarBuilder::from_mmaped_safetensors(&[path], DType::BF16, &device)?;
// → Error: unsupported safetensor dtype F8_E4M3
```

This blocks the entire Rust ML ecosystem: candle, candle-mi, burn, tch. The only workaround is a Python script.

## After anamnesis

### Step 1 — Inspect (what did quantization forget?)

```bash
$ amn inspect model-fp8.safetensors

Format:     Fine-grained FP8 (E4M3), 128×128 blocks
Tensors:    224 quantized  ·  53 passthrough (norms, embeddings)
Size:       3.4 GB (FP8) → 6.8 GB (BF16)
```

### Step 2 — Remember (dequantize)

```bash
$ amn remember model-fp8.safetensors --to bf16

Anamnesis complete: 224 tensors recalled in 2.3s
Output: model-bf16.safetensors (6.8 GB)
```

### Step 3 — Use anywhere

```rust
// candle — now works
let vb = VarBuilder::from_mmaped_safetensors(&["model-bf16.safetensors"], DType::BF16, &device)?;

// candle-mi — logit lens, circuit tracing, CLT analysis
let model = MIModel::from_pretrained("./model-bf16/")?;
```

## The same in Rust (library API)

```rust
use anamnesis::{inspect, dequantize_file, TargetDtype};

let info = inspect("model-fp8.safetensors")?;
assert_eq!(info.format.to_string(), "Fine-grained FP8 (E4M3), 128×128 blocks");

dequantize_file("model-fp8.safetensors", "model-bf16.safetensors", TargetDtype::BF16)?;
// Standard BF16 safetensors — loadable by any Rust ML framework.
```

## The opposite direction (future: `lethe` module)

```rust
use anamnesis::lethe::{quantize_file, QuantScheme};

// BF16 → FP8: the weights forget some precision
quantize_file("model-bf16.safetensors", "model-fp8.safetensors", QuantScheme::FineGrainedFp8)?;
```

## CLI verbs

| Verb | Operation | Metaphor |
|---|---|---|
| `amn inspect` | Show format, tensor count, size estimate | What did Lethe take? |
| `amn remember` | Dequantize (FP8/GPTQ/AWQ → BF16/F16/F32) | Anamnesis — recollection of the forms |
| `amn forget` | Quantize (BF16/F32 → FP8/INT8/INT4) | Lethe — drinking from the river |

## Integration with hf-fetch-model

```bash
# Download + remember in one step
$ hf-fm mistralai/Ministral-3-3B-Instruct-2512 --preset safetensors --dequantize bf16
```

hf-fetch-model calls anamnesis under the hood. The user never touches FP8 files.
