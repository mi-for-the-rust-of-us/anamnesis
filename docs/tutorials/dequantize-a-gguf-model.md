# Dequantize a GGUF model to BF16

*Take a llama.cpp GGUF quant, recover full `BF16` weights, and load the result in any Rust ML framework — candle, burn, or tch.*

*~1000 words · about 4 min read*

<!-- Last updated: 2026-08-12, anamnesis v0.7.3 -->

<!--
STYLE CONVENTIONS for editing this tutorial — keep growth consistent.
(Adapted from the sibling hf-fetch-model tutorials so the two read alike.)

1. Tone: match the FAQ. Conversational, address the reader as "you", short
   paragraphs over bullet lists where prose works.
2. Pinning: the model is `bartowski/SmolLM2-135M-Instruct-GGUF`, file
   `SmolLM2-135M-Instruct-Q4_K_M.gguf`. Every output block below is real,
   captured from `amn` against that file — paste exact output, do not
   paraphrase. If you re-capture, update the counts/sizes here to match.
3. Output blocks: trim only when a block runs long and the trimmed lines
   are representative repetition (note the trim with `…`).
4. Shell: commands are identical on Windows/macOS/Linux; only show a
   PowerShell-vs-bash split when an env var or path actually differs.
5. Length budget: under 300 lines total, including embedded outputs.
   Update the word-count + reading-time line at the top whenever the prose
   changes non-trivially (~250 wpm, code blocks excluded).
6. Scope: features that ship today. The Python equivalent of this walk-
   through arrives with the bindings (v0.8.0) as its own tutorial.
-->

## Contents

- [Why dequantize first?](#why-dequantize-first)
- [The 30-second answer](#the-30-second-answer)
- [Step 1 — Get a GGUF file](#step-1--get-a-gguf-file)
- [Step 2 — Inspect before you commit](#step-2--inspect-before-you-commit)
- [Step 3 — Dequantize to BF16](#step-3--dequantize-to-bf16)
  - [If you want `float32` instead (v0.7.3+)](#if-you-want-float32-instead-v073)
- [Step 4 — Verify the result](#step-4--verify-the-result)
- [What you've learned](#what-youve-learned)

## Why dequantize first?

GGUF is llama.cpp's format, and its k-quants (`Q4_K`, `Q6_K`, …) are designed to be dequantized *inside* llama.cpp's inference loop. The Rust ML frameworks have no loader for them: hand candle or burn a GGUF k-quant and you get nothing usable. If you want to run, fine-tune, or interpret a GGUF-only model in Rust, you first have to recover full-precision weights and write them in a format those frameworks *do* read — standard `BF16` safetensors. That recovery is exactly what `amn remember` does.

The trade-off is size: dequantizing expands the file (a 4-bit quant roughly quadruples on its way to `BF16`). You are spending disk to buy framework compatibility, so it is worth checking the numbers before you commit — which is Step 2.

## The 30-second answer

```
amn inspect SmolLM2-135M-Instruct-Q4_K_M.gguf      # see what you have
amn remember SmolLM2-135M-Instruct-Q4_K_M.gguf -o smol-bf16.safetensors
```

The first command summarizes the file; the second writes a `BF16` safetensors you can load anywhere. The rest of this tutorial walks through each step with real output.

## Step 1 — Get a GGUF file

This walkthrough uses `SmolLM2-135M-Instruct-Q4_K_M.gguf` from the `bartowski/SmolLM2-135M-Instruct-GGUF` repository — small enough to follow along quickly. The easiest way to fetch a single file from HuggingFace is the sibling tool [hf-fetch-model](https://github.com/mi-for-the-rust-of-us/hf-fetch-model):

```
hf-fm download-file bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q4_K_M.gguf
```

Any download method works — `huggingface-cli`, a browser, `curl`. You just need the `.gguf` file on disk.

## Step 2 — Inspect before you commit

`amn inspect` reads only the GGUF header — no weight data — and tells you what is inside and roughly what dequantization will cost:

```
$ amn inspect SmolLM2-135M-Instruct-Q4_K_M.gguf
Format:      GGUF v3
Arch:        llama
Tensors:     272
Total size:  99 MB
Dtypes:      Q8_0, F32, Q6_K, Q5_0, Q4_K
Alignment:   32 bytes
```

Two things to read here. The `Dtypes` line is a *mix* — `Q4_K_M` is a recipe, not a single block type, so the file holds `Q4_K`, `Q6_K`, `Q5_0`, plus some `Q8_0` and `F32` tensors that were never quantized. And `Total size: 99 MB` is the on-disk quantized size; keep it in mind for the next step, because `BF16` will be a good deal larger.

## Step 3 — Dequantize to BF16

Run `amn remember` (alias `amn dequantize`) with `-o` pointing at the output:

```
$ amn remember SmolLM2-135M-Instruct-Q4_K_M.gguf -o smol-bf16.safetensors
Converting SmolLM2-135M-Instruct-Q4_K_M.gguf → smol-bf16.safetensors
  272 tensors
  211 dequantized to BF16, 61 passed through
  Output: smol-bf16.safetensors
```

The breakdown is the interesting part: of 272 tensors, **211 were quantized and are now recovered to `BF16`**, while **61 passed through** unchanged — those are the `F32` tensors (norms, a few small weights) that GGUF never quantized in the first place, so there was nothing to recover. anamnesis dequantizes each k-quant block bit-exactly against the `gguf` Python reference, then writes a single standard safetensors file.

### If you want `float32` instead (v0.7.3+)

`BF16` keeps 8 significand bits, and a `Q8_0` value is an `f16` scale times an `int8` — up to ~18 bits. On this very model, only **3–20 %** of dequantized values land exactly on a `BF16` grid point; the rest get rounded. For `Q4_K` that rounding is dwarfed by the quantization error, but for the high-precision types (`Q8_0`, `Q6_K`) it is the *same order* as the error the format exists to avoid.

If that matters to you, ask `convert` for `f32`:

```
$ amn convert SmolLM2-135M-Instruct-Q4_K_M.gguf --to safetensors --out-dtype f32
Converting SmolLM2-135M-Instruct-Q4_K_M.gguf -> SmolLM2-135M-Instruct-Q4_K_M-f32.safetensors
  211 dequantized to F32
  Wrote 272 tensors -> SmolLM2-135M-Instruct-Q4_K_M-f32.safetensors
```

Note the derived filename ends in `-f32`, not `-bf16`: it tracks the dtype, so a file never claims a width it does not hold.

`f32` output performs **no narrowing at all**, so the values are the `f32` that `gguf-py` itself produces — verified bit-exactly against it across all 22 kernels, with no tolerance. Two things to expect. The file is twice the size (**513 MB against 257 MB** here), and the conversion runs about **1.5–1.6× slower**, because doubling the output bytes costs real bandwidth. That is the price of the precision, not a defect.

There is also `--out-dtype f16`, which buys 3 significand bits over `bf16` at the same 2 bytes — but pays for them with a much narrower exponent range, overflowing to infinity above 65504 where `bf16` matches `f32`'s range. It is not simply the better 2-byte choice.

Note this is `amn convert`, not `amn remember`: the `remember` path stays `BF16`-only until v0.7.4, when its other kernel families get the same treatment.

## Step 4 — Verify the result

Inspect the file you just wrote:

```
$ amn inspect smol-bf16.safetensors
Format:      Unquantized
Quantized:   0 tensors (weights)
Passthrough: 272 tensors (norms, embeddings)
Size:        257 MB (unquantized) -> 257 MB (BF16)
```

`Quantized: 0 tensors` is the confirmation you want — there is nothing left to recover, every tensor is now plain `BF16`/`F32`. Note the size jump: **99 MB of GGUF became 257 MB of `BF16` safetensors**. That ~2.6× expansion is the cost of trading llama.cpp's compact packing for universal framework compatibility — expected, not a bug.

That `smol-bf16.safetensors` now loads directly in candle:

```rust
let vb = VarBuilder::from_mmaped_safetensors(&["smol-bf16.safetensors"], DType::BF16, &device)?;
```

— the same call that would have failed on the original GGUF.

## What you've learned

- GGUF k-quants don't load in Rust ML frameworks; `amn remember` recovers them to standard `BF16` safetensors that do.
- `amn inspect` is a free, header-only preview — use it to read the dtype mix and size *before* spending disk on a full dequantization.
- A `Q4_K_M` file is a mix of block types, and the tensors GGUF left in `F32` simply pass through untouched.
- Dequantization trades size for compatibility (here 99 MB → 257 MB), so check the numbers up front.
- Since v0.7.3, `amn convert --out-dtype f32` skips the `BF16` narrowing entirely when you need the reference `float32`, at twice the bytes and ~1.5× the time.

For the safety angle — what to do when the file came from somewhere you don't trust — see [Inspect before you parse (untrusted input)](inspect-before-you-parse.md) and the FAQ on [parsing untrusted input](../FAQ.md#parsing-untrusted-input). For the other input formats (FP8 / GPTQ / AWQ / BitsAndBytes safetensors), the same `amn remember` command applies — only the source scheme differs. And to change *container* rather than just recover precision — GGUF → `bnb-nf4`, or writing a scalar GGUF with your own metadata — see [Convert a model between formats](convert-between-formats.md).
