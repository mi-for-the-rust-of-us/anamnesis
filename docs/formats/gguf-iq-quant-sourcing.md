# GGUF `IQ*` / `TQ*` / `MXFP4` / `NVFP4` / `Q1_0` / `Q2_0`: fixture-sourcing reference

One-page reference for finding HuggingFace GGUF files that ship each of the `IQ*`, `TQ*`, and `MXFP4` block types, for use in anamnesis cross-validation fixtures. Distilled from a sourcing investigation on 2026-04-22 after Phase 4.5 steps 1–2 landed.

Phase 4.5 step 7 (`cross-validation extension`) requires a 65 536-element fixture per block type, extracted from a real model tensor where possible and synthesised via the Python `gguf` package where not. **As of step 6 (MXFP4) the last remaining coverage gap was closed** — anamnesis dequantised every GGUF block type shipping on HuggingFace at the time (22 of 22 production kernels). This document tracks where each type comes from and is now a historical reference rather than a live to-do list.

**Phase 7.10 reopened it, briefly.** Upstream `ggml` added `NVFP4` (40), `Q1_0` (41) and `Q2_0` (42) in 2026, and a user's `NVFP4` file was refused outright ([issue #15](https://github.com/mi-for-the-rust-of-us/anamnesis/issues/15)). All three now have kernels and fixtures, so 25 of 25 production kernels are cross-validated. `gguf-py` cannot quantise any of the three and cannot dequantise `Q1_0` or `Q2_0`, so their goldens come from ggml's own C through `tests/fixtures/gguf_reference/ggml_ref/`, and the generator now checks every fixture against that C. A weekly CI job (`ggml-drift.yml`) compares upstream's `enum ggml_type` with the parser's, so the next addition is found by CI rather than by a user.

## Contents

- [Block-layout summary](#block-layout-summary)
- [Model sourcing matrix](#model-sourcing-matrix)
- [Gotchas discovered along the way](#gotchas-discovered-along-the-way)
- [Remote-header probe recipe](#remote-header-probe-recipe)
- [Python `gguf` quantize coverage](#python-gguf-quantize-coverage)
- [Canonical sources](#canonical-sources)

---

## Block-layout summary

All sizes from `ggml-common.h` at commit cut 2026-04-22, except the three Phase 7.10 rows, which are from llama.cpp [`37b53fd`](https://github.com/ggml-org/llama.cpp/blob/37b53fd4545847188fdad29e38ba57875efc8228/ggml/src/ggml-common.h). `QK_K = 256`.

| `ggml_type` | Disc | Block | `type_size` | Notes |
|---|---:|---:|---:|---|
| `IQ4_NL` | 20 | 32 | 18 | shipped (`Phase 4.5 step 1`) |
| `IQ4_XS` | 23 | 256 | 136 | shipped (`Phase 4.5 step 1`) |
| `IQ2_XXS` | 16 | 256 | 66 | shipped (`Phase 4.5 step 2`) |
| `IQ2_XS` | 17 | 256 | 74 | shipped (`Phase 4.5 step 2`) |
| `IQ2_S` | 22 | 256 | 82 | shipped (`Phase 4.5 step 2`) |
| `IQ3_XXS` | 18 | 256 | 98 | shipped (`Phase 4.5 step 3`) |
| `IQ3_S` | 21 | 256 | 110 | shipped (`Phase 4.5 step 3`) |
| `IQ1_S` | 19 | 256 | 50 | shipped (`Phase 4.5 step 4`) |
| `IQ1_M` | 29 | 256 | 56 | shipped (`Phase 4.5 step 4`) |
| `TQ1_0` | 34 | 256 | 54 | shipped (`Phase 4.5 step 5`) |
| `TQ2_0` | 35 | 256 | 66 | shipped (`Phase 4.5 step 5`) |
| `MXFP4` | 39 | 32 | 17 | shipped (`Phase 4.5 step 6`) |
| `NVFP4` | 40 | 64 | 36 | shipped (`Phase 7.10`, issue #15); four `UE4M3` scales per block, one per 16 elements |
| `Q1_0` | 41 | 128 | 18 | shipped (`Phase 7.10`, issue #15); `f16` scale + 1 bit per element |
| `Q2_0` | 42 | 64 | 18 | shipped (`Phase 7.10`, issue #15); `f16` scale + 2 bits per element |

Every type's byte size is verifiable in [`src/parse/gguf.rs::type_size()`](../../src/parse/gguf.rs). After step 6 every variant returns `Some(_)` — there is no longer any deferred type.

---

## Model sourcing matrix

Download sizes and tensor counts from remote-header probes performed 2026-04-22. Tensor thresholds are `≥ 65 536 elements` (the slice size Phase 4 fixtures use).

### Shipped kernels (reference — fixtures already committed)

| Kernel | Fixture | HF source | Download |
|---|---|---|---|
| `Q4_0` / `Q4_1` / `Q8_0` | `smollm2_q4_0.bin` / `_q4_1.bin` / `_q8_0.bin` | `bartowski/SmolLM2-135M-Instruct-GGUF` | already local |
| `Q5_0` / `Q2_K` | `tinyllama_q5_0.bin` / `_q2_k.bin` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | already local |
| `Q5_1` / `Q3_K`–`Q6_K` | `smollm2_*.bin` | `bartowski/SmolLM2-135M-Instruct-GGUF` (various `Q*_K_*.gguf`) | already local |
| `IQ4_NL` | `smollm2_iq4_nl.bin` | `SmolLM2-135M-Instruct-Q2_K.gguf` (mix contains IQ4_NL) | already local |
| `IQ4_XS` | `smollm2_iq4_xs.bin` | `SmolLM2-135M-Instruct-IQ4_XS.gguf` | already local |
| `IQ2_XXS` | `mistral_7b_iq2_xxs.bin` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ2_XXS.gguf` | **1.86 GB** (one-off) |
| `IQ2_XS` | `mistral_7b_iq2_xs.bin` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ2_XS.gguf` | **2.05 GB** (one-off) |
| `IQ2_S` | `qwen25_iq2_s.bin` | `bartowski/Qwen2.5-0.5B-Instruct-GGUF` / `...-IQ2_M.gguf` | already local (313 MB) |
| `IQ3_XXS` | `mistral_7b_iq3_xxs.bin` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ3_XXS.gguf` | **2.64 GB** (one-off) |
| `IQ3_S` | `mistral_7b_iq3_s.bin` | same `...-IQ3_XXS.gguf` file (ships 33 `IQ3_S` secondary tensors) | already local (via `IQ3_XXS`) |
| `IQ1_S` | `mistral_7b_iq1_s.bin` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ1_S.gguf` | **1.50 GB** (one-off) |
| `IQ1_M` | `mistral_7b_iq1_m.bin` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` / `...-IQ1_M.gguf` (no shared file with IQ1_S) | **1.64 GB** (one-off) |
| `TQ1_0` | `synthetic_tq1_0.bin` | **synthetic** via `gguf.quants.quantize()` (seed=42, scale=0.1) | already (no download) |
| `TQ2_0` | `synthetic_tq2_0.bin` | **synthetic** via `gguf.quants.quantize()` (same seed) | already (no download) |
| `MXFP4` | `synthetic_mxfp4.bin` | **synthetic** via `gguf.quants.quantize()` (same seed) | already (no download) |
| `NVFP4` | `zeta21_nvfp4.bin` | `distaste447/zeta-2.1-NVFP-GGUF` / `zeta-2.1-NVFP4.gguf` at `3cb91594`, `output.weight` (golden from `gguf-py`, checked against ggml's C) | **5.18 GiB** (one-off) |
| `NVFP4` (scale folded) | `zeta21_nvfp4_folded.bin` + `.scale` | the same slice times its tensor's `output.scale`, as `remember` writes it | same file |
| `NVFP4` | `synthetic_nvfp4.bin` | **synthetic**, quantised and dequantised by ggml's C (same seed); `gguf-py`'s dequantise agrees | already (no download) |
| `Q1_0` | `synthetic_q1_0.bin` | **synthetic** via ggml's C only (same seed) | already (no download) |
| `Q2_0` | `synthetic_q2_0.bin` | **synthetic** via ggml's C only (same seed) | already (no download) |

### Pending kernels

None. Every recognised `GgufType` block-quant variant has a dedicated kernel and a committed cross-validation fixture: Phase 4.5 closed the original set and Phase 7.10 the three types `ggml` added in 2026. The next upstream addition will reopen this section, and `ggml-drift.yml` is what will notice it.

### Alternative real-model sources (for future cross-checking)

| Kernel | Alternative | Size | Status |
|---|---|---:|---|
| `TQ2_0` | `gianni-cor/bitnet_b1_58-large-TQ2_0/bitnet_b1_58-large-TQ2_0.gguf` | 207 MiB | real-model cross-check option — unverified |
| `TQ1_0` | `BoscoTheDog/Llama3-8B-1.58-100B-tokens-TQ1_0_gguf_chunked` | chunked, ~1.5 GB | real-model cross-check option — chunked upload, unverified |
| `MXFP4` | `ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf` | 11.28 GiB | **72 `MXFP4` tensors confirmed** — too large to justify if synthetic path works |

The synthetic path is sufficient for bit-exact cross-validation (the ggml reference dequant is deterministic). Real-model sources are only needed if we ever want to validate against a byte-identical real-world byte stream — not a current requirement.

---

## Gotchas discovered along the way

Things that would have wasted hours if hit blind.

### 1. `Mistral-7B-Instruct-v0.3-IQ2_S.gguf` does **not** ship `IQ2_S` tensors

Despite the filename, remote-header probe reveals it actually ships **`IQ2_XS` (156) + `IQ3_S` (37) + `Q4_K` (32) + `Q5_K` (1) + `F32` (65)** — **zero `IQ2_S` tensors**. Confirmed 2026-04-22.

The filename advertises the **intended** base quant of the mix recipe, not the only quant actually used. For step 2 we sourced `IQ2_S` from `Qwen2.5-0.5B-Instruct-IQ2_M.gguf` instead (which does ship 21 real `IQ2_S` tensors).

Silver lining: the same file is a free source of `IQ3_S` tensors for step 3, so the 2.16 GB download already on disk is reusable.

### 2. Small models' `IQ2_M.gguf` mixes dropped `IQ2_XXS` and `IQ2_XS` entirely

For all of `bartowski/SmolLM2-135M-Instruct-GGUF`, `bartowski/Qwen2.5-0.5B-Instruct-GGUF`, `bartowski/Qwen2.5-1.5B-Instruct-GGUF`, `bartowski/Phi-3.5-mini-instruct-GGUF` — their `IQ2_M.gguf` mix contains `IQ2_S` + `IQ3_S` + `Q4_K` / `Q5_K` only, **no `IQ2_XXS` or `IQ2_XS`** anywhere. The `IQ2_M` base quant was revised in newer `llama.cpp` to skip the less-common 2-bit variants for small models.

Verified on 2026-04-22 on all four files. Mistral-7B-v0.3 is the smallest repo shipping **pure** `IQ2_XXS.gguf` and `IQ2_XS.gguf` as separate files.

### 3. `microsoft/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf` uses `ggml_type 36`, not `TQ1_0`/`TQ2_0`

Type 36 is in the reserved/removed range (34 = `TQ1_0`, 35 = `TQ2_0`, 39 = `MXFP4`; 36–38 reserved). Microsoft's BitNet toolchain ships its own non-standard `I2_S` quant that anamnesis **cannot** dequantise without explicit support for type 36. Out of scope for Phase 4.5.

The community uploads (`gianni-cor/bitnet_b1_58-large-TQ2_0`, `BoscoTheDog/Llama3-8B-1.58-100B-tokens-TQ1_0_gguf_chunked`) use the standard `TQ1_0` / `TQ2_0` types and are the correct real-model sources if ever needed.

### 4. Python `gguf.quants.quantize()` coverage is uneven

Confirmed 2026-04-22 on `gguf==0.17+`:

| Type | `quantize()` | `dequantize()` | Notes |
|---|:-:|:-:|---|
| `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` | ✅ | ✅ | legacy block quants |
| `Q2_K`–`Q8_K` | ✅ | ✅ | K-quants |
| `IQ4_NL`, `IQ4_XS` | ✅ | ✅ | `ggml` hand-coded |
| `IQ2_XXS`, `IQ2_XS`, `IQ2_S` | ❌ `NotImplementedError` | ✅ | **real-model source required** |
| `IQ3_XXS`, `IQ3_S`, `IQ1_S`, `IQ1_M` | ❌ `NotImplementedError` | ✅ | **real-model source required** |
| `TQ1_0`, `TQ2_0`, `MXFP4` | ✅ | ✅ | **all three backed by Phase 4.5 step 5 + 6 synthetic fixtures** |
| `NVFP4` | ❌ | ✅ | synthetic fixture quantised by ggml's C; `gguf-py`'s dequantise must agree (checked at `gguf-py` 0.19.0, llama.cpp `37b53fd`) |
| `Q1_0`, `Q2_0` | ❌ | ❌ | **ggml's C is the only reference** (`ggml_ref/`) |

This asymmetry drove the "synthetic vs real-model" split in the sourcing matrix above. With step 6 landed every variant has a committed fixture.

---

## Remote-header probe recipe

`hf-fm inspect` is `.safetensors`-only as of `hf-fetch-model v0.9.7`. For GGUF, we built a one-off Python probe that fetches only the first ~6 MB of a remote GGUF via HTTP `Range` request and parses the tensor-info table manually. This enumerates tensor types and counts without pulling the (multi-GB) tensor data.

Pattern:
```python
import struct, urllib.request
from collections import Counter

HEADER_BYTES = 6 * 1024 * 1024  # bump to 24 MB for files with huge metadata KV sections (e.g. gpt-oss, BitNet)

def probe(repo, filename):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{HEADER_BYTES-1}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        buf = r.read()
    assert buf[:4] == b"GGUF"
    # parse magic + version + n_tensors + n_kv, skip each KV pair by type,
    # then walk the tensor info table counting ggml_type discriminants.
    # See /tmp/probe_gguf_manual.py in the 2026-04-22 session for the full code.
```

The script worked for all Mistral-7B IQ variants, the IQ3_M/IQ3_XS small-model variants, and Qwen2.5-*-IQ3_XS at 6 MB; BitNet and gpt-oss needed 24 MB due to long metadata KV sections.

**Future:** extending `hf-fm inspect` to accept GGUF files (using the same Range-request trick) would make this a one-line CLI call. The format is well-defined and the magic is unambiguous. Flagged as a natural follow-up for the hf-fetch-model crate.

---

## Python `gguf` quantize coverage

Script that confirmed the coverage matrix above, kept inline for reproducibility:

```python
import numpy as np
from gguf import quantize, dequantize, GGMLQuantizationType, GGML_QUANT_SIZES

np.random.seed(42)
x = np.random.randn(65536).astype(np.float32) * 0.1
for name, tt in [
    ('MXFP4', GGMLQuantizationType.MXFP4),
    ('TQ1_0', GGMLQuantizationType.TQ1_0),
    ('TQ2_0', GGMLQuantizationType.TQ2_0),
    ('IQ3_XXS', GGMLQuantizationType.IQ3_XXS),
    ('IQ3_S',   GGMLQuantizationType.IQ3_S),
    ('IQ1_S',   GGMLQuantizationType.IQ1_S),
    ('IQ1_M',   GGMLQuantizationType.IQ1_M),
]:
    bs, ts = GGML_QUANT_SIZES[tt]
    try:
        raw = quantize(x, tt)
        y = dequantize(raw, tt)
        print(f'{name}: OK, block={bs}, type_size={ts}, raw={raw.nbytes} bytes')
    except NotImplementedError:
        print(f'{name}: dequant only (real-model source required)')
```

---

## Canonical sources

- **Block layouts**: `ggml-org/llama.cpp/ggml/src/ggml-common.h` — `block_iq*`, `block_tq*`, `block_mxfp4` structs with `_Static_assert` byte counts.
- **Dequant reference**: `ggml-org/llama.cpp/ggml/src/ggml-quants.c` — `dequantize_row_iq*`, `dequantize_row_tq*`, `dequantize_row_mxfp4` scalar functions (the Python `gguf` package's `dequantize()` mirrors these).
- **Python reference**: `gguf` PyPI package, `gguf/quants.py` — the `Q*Class.quantize_blocks` / `dequantize_blocks` methods.
- **Probe script** (session-local, 2026-04-22): `/tmp/probe_gguf_manual.py`, `/tmp/probe_all_remaining.py`.
- **ggml's C itself** (Phase 7.10): `tests/fixtures/gguf_reference/ggml_ref/` builds `ggml-base` at llama.cpp `37b53fd` and calls each type's `from_float_ref` / `to_float` (`quantize_row_*_ref` / `dequantize_row_*`). It is the reference for the types `gguf-py` lacks, and a second, independent check on every other golden.

Update this document whenever a `GGUF` type is added or a new source surfaces.
