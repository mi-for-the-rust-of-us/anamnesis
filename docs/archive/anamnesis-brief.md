# anamnesis

> **HISTORICAL (archived at the v0.7.8 close-out).** The project brief from 2026-03-19, before the first release. `README.md` and `ROADMAP.md` describe the project as it is.

*Make model weights usable in Rust — parse any tensor format, recover any precision.*

**crates.io:** available — 0 results for "anamnesis" as of 2026-03-19.

---

## Motivation

Model weights ship in formats that Rust ML frameworks cannot consume directly: quantized safetensors (FP8, GPTQ, AWQ, NF4), NumPy archives (NPZ/NPY for SAE weights), and more to come. The Python ecosystem handles these transparently. The Rust ecosystem has **nothing**. `anamnesis` fills this gap: a framework-agnostic, pure-Rust crate that parses model weights from their stored format and, when needed, recovers precision lost to quantization. No PyTorch, no Python, no GPU required.

## The Name

In Plato's epistemology, **ἀνάμνησις** (*anamnesis*) means *recollection* — the idea that learning is not acquiring new knowledge but *remembering* what the soul already knew.

But recollection does not happen all at once. Before you can remember, you must first *make contact* with the thing — apprehend its form, decode its structure, validate that it is what it claims to be. This is **parsing**: the foundational act of anamnesis. You cannot remember what you have not first parsed.

Sometimes parsing alone is enough. An NPZ file stores weights in a foreign container, but the values are intact — nothing was forgotten. Parsing recovers them fully: anamnesis complete.

Other times, parsing reveals that something was lost. An FP8 safetensors file has been through **λήθη** (*lethe*) — the river of forgetting in the Greek underworld. Souls drank from it and forgot their previous life. Weights passed through quantization and forgot their precision. Parsing decodes the structure and validates what remains; then **remembering** — dequantization — recovers what it can from the compressed shadow. Some information crossed the river and is gone forever.

The full arc: **parse** (apprehend) → **remember** (recover) or **forget** (surrender). Parsing is always the first step. Remembering and forgetting build on what parsing finds.

## Morphological Family

| Form | Word | Meaning |
|---|---|---|
| Crate name | **anamnesis** | the art of recollection — making weights usable again |
| Foundation | **parse** | apprehend the form — decode, validate, make contact |
| Recovery | **remember** | dequantize — recover precision from a compressed shadow |
| Opposite concept | **lethe** / **forget** | quantize — surrender precision to compression |
| Adjective | **anamnestic** | "an anamnestic conversion" |
| CLI shortcut | **amn** | `amn remember model-fp8.safetensors --to bf16` |
| Error metric | **lethe distance** | the gap between original and recovered values |

## Architecture

Parsing is the foundation. Every operation begins with it. You cannot remember what you have not first parsed.

```
anamnesis (library crate)
│
├── parse/              ← decode + validate any tensor format
│   ├── safetensors        safetensors (including quantized metadata)
│   ├── npz                NPZ/NPY archives (feature-gated)
│   └── (future)           GGUF, etc.
│
├── remember/           ← built on parse: precision recovery (dequantize)
│   ├── fp8                fine-grained + per-tensor FP8 (E4M3, E5M2)
│   ├── gptq               GPTQ dequantization
│   ├── awq                AWQ dequantization
│   └── bnb                BitsAndBytes (NF4, INT8) dequantization
│
├── lethe/              ← built on parse: precision reduction (quantize)
│
├── inspect             ← built on parse: report without transforming
│
└── bin/
    └── main.rs         ← CLI binary (feature-gated behind "cli")
                           installed as both `anamnesis` and `amn`
                           subcommands: parse, inspect/info, remember/dequantize, forget/quantize
```

- **Parse only** (NPZ): parsing suffices — the values are intact, nothing to remember.
- **Parse + remember** (FP8 safetensors): parsing decodes the structure; remembering recovers precision.
- **Parse + forget** (BF16 → FP8): parsing reads the source; forgetting compresses it.
- **Parse + inspect**: parsing reads the metadata; inspect reports what Lethe took.

## Usage

The library API reflects the parse-first architecture: parse once, then act.

### Remember (FP8 — recover what Lethe took)

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

### Parse (NPZ — nothing was forgotten)

```rust
use anamnesis::parse_npz;

// Parse — nothing to remember, the data is intact
let tensors = parse_npz("sae_weights.npz")?;
let encoder = &tensors["W_enc"];
// encoder.shape, encoder.dtype, encoder.data — ready for any framework
```

### Forget (future: `lethe` module)

```rust
use anamnesis::lethe::QuantScheme;

let model = anamnesis::parse("model-bf16.safetensors")?;

// The weights forget some precision
model.forget("model-fp8.safetensors", QuantScheme::FineGrainedFp8)?;
```

## Ecosystem Fit

```
safetensors / npz       ← file formats
    ↓
anamnesis (library)     ← parse, remember, lethe
anamnesis / amn (CLI)   ← amn parse, amn inspect, amn remember, amn forget
    ↓
hf-fetch-model          ← download + transform pipeline
                           (hf-fm --dequantize bf16, download_and_parse_npz)
candle / candle-mi      ← load + run + interpret
burn / tch / ...        ← any Rust ML consumer
```

## Scope

**Phase 1:** FP8 dequantization (fine-grained + per-tensor). The parsing layer for safetensors is built here as the foundation. SIMD-friendly scalar loops verified with `cargo-show-asm`. CLI binary (`anamnesis` / `amn`). Dependencies: `safetensors`, `half`, `float8`, `clap` (feature-gated behind `cli`).

**Phase 2:** GPTQ, AWQ, BitsAndBytes dequantization. Extends the `remember` layer. Feature-gated.

**Phase 3:** NPZ/NPY parsing — extends the `parse` layer to NumPy archives for SAE weights (Gemma Scope, etc.). Builds on `npyz` crate (adds bf16 interpretation layer). Feature-gated behind `npz`. Here, parsing alone suffices — no remembering needed. Download integration stays in hf-fetch-model as a thin wrapper (`download()` + `anamnesis::parse_npz()`).

**Phase 4:** Quantization (`lethe` module) — BF16/F32 → FP8/INT8/INT4.

See `ROADMAP.md` for full details, `amn-flagship-v2.md` for the flagship example, and `CONVENTIONS.md` for coding conventions.
