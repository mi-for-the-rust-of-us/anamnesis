# Convert a model between formats

*One command takes any input anamnesis reads — FP8 / GPTQ / AWQ / BnB safetensors, GGUF, NPZ, `.pth` — to `safetensors`, scalar `gguf`, or `bnb-nf4`, with quantized inputs recovered to `BF16` on the way through.*

*~1000 words · about 4 min read*

<!-- Last updated: 2026-08-07, anamnesis v0.7.2 -->

<!--
STYLE CONVENTIONS for editing this tutorial — keep growth consistent.
(Adapted from the sibling hf-fetch-model tutorials so the two read alike.)

1. Tone: match the FAQ. Conversational, address the reader as "you", short
   paragraphs over bullet lists where prose works.
2. Pinning: the model is `RedHatAI/Llama-3.2-1B-Instruct-FP8`, file
   `model.safetensors`. Every output block below is real, captured from
   `amn` (release build) against that file — paste exact output, do not
   paraphrase. If you re-capture, update the counts/sizes here to match.
   The `gguf → gguf` block is captured against
   `bartowski/Qwen2.5-0.5B-Instruct-GGUF` (`…-IQ2_M.gguf`).
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

- [Why convert?](#why-convert)
- [The 30-second answer](#the-30-second-answer)
- [Step 1 — Get a model](#step-1--get-a-model)
- [Step 2 — Inspect before you commit](#step-2--inspect-before-you-commit)
- [Step 3 — Convert to the format you need](#step-3--convert-to-the-format-you-need)
- [Step 4 — Stamp GGUF metadata](#step-4--stamp-gguf-metadata)
- [The auto-chain, and what stays out of scope](#the-auto-chain-and-what-stays-out-of-scope)
- [What you've learned](#what-youve-learned)

## Why convert?

Model weights ship in whatever format the publisher picked — an FP8 safetensors from one lab, a GGUF quant from another, an NPZ dump from a research repo. Your *destination* rarely agrees: a `bitsandbytes` workflow wants NF4 safetensors, a llama.cpp-style tool wants GGUF, a Rust framework wants plain `BF16`. Bridging the two used to be a two-hop dance — dequantize to full precision, then re-encode — with a temporary file you had to stage and clean up by hand.

`amn convert` collapses that to one command. Every input routes through an in-memory **`BF16` hub**: the reader normalizes the input (quantized tensors dequantized to `BF16`, scalar tensors kept in their original dtype), and the writer emits your target. A quantized input reaching a quantized target **auto-chains** through `BF16` internally — no temp file, no second command.

## The 30-second answer

```
amn inspect model.safetensors                    # see what you have
amn convert model.safetensors --to bnb-nf4       # or --to gguf, --to safetensors
```

The first command previews the file header-only; the second writes the converted model to a derived path. The rest of this tutorial walks through it with real output.

## Step 1 — Get a model

This walkthrough uses `RedHatAI/Llama-3.2-1B-Instruct-FP8` — a per-tensor FP8 checkpoint, small enough to follow along quickly. Fetch it with the sibling tool [hf-fetch-model](https://github.com/PCfVW/hf-fetch-model) (`hf-fm`):

```
hf-fm RedHatAI/Llama-3.2-1B-Instruct-FP8 --preset safetensors
```

`hf-fm cache path RedHatAI/Llama-3.2-1B-Instruct-FP8` then prints the snapshot directory holding `model.safetensors`. Any download method works — the commands below assume the file is on disk as `model.safetensors`.

## Step 2 — Inspect before you commit

`amn inspect` reads only the header — no weight data — and tells you the scheme and what recovery will cost:

```
$ amn inspect model.safetensors
Format:      Per-tensor FP8 (E4M3)
Quantized:   112 tensors (weights) + 224 scale tensors (BF16)
Passthrough: 35 tensors (norms, embeddings)
Size:        1.88 GB (FP8) -> 2.79 GB (BF16)
Lethe took:  ~928 MB of precision
```

Two things to read. The **112 quantized weights** are what any conversion will recover to `BF16` first — the hub always pivots through full precision. And the **`1.88 GB (FP8) -> 2.79 GB (BF16)`** line is your size budget: the hub materializes `BF16` in memory (peak ≈ 2× the model), so plan for it before converting a large checkpoint.

## Step 3 — Convert to the format you need

Same input, three targets. Pick the one your destination reads.

**To `bnb-nf4`** (a `bitsandbytes`-NF4 safetensors):

```
$ amn convert model.safetensors --to bnb-nf4
Converting model.safetensors -> model-bnb-nf4.safetensors
  112 dequantized to BF16
  114 quantized to NF4, 33 passed through as BF16
  Wrote 147 tensors -> model-bnb-nf4.safetensors
```

Read the breakdown top to bottom — it *is* the hub. The **112** FP8 weights are recovered to `BF16` on the way in; then of the 147 hub tensors, **114** 2-D float weights are re-encoded to NF4 and **33** (norms, 1-D biases) pass through as `BF16`, which is the BnB encoder's contract. One command spanned both halves — the "dequantize first, then re-encode" two-hop is gone.

**To `gguf`** (an unquantized, scalar GGUF):

```
$ amn convert model.safetensors --to gguf
Converting model.safetensors -> model-gguf.gguf
  112 dequantized to BF16
  Wrote 147 tensors -> model-gguf.gguf
```

**To `safetensors`** (plain `BF16`) is `--to safetensors` (alias `--to bf16`) — the same recovery `amn remember` does, exposed through the convert dispatch.

Output paths are derived when you omit `-o`: a known quant suffix is stripped from the stem and `-{target}.{ext}` appended (`model-fp8.safetensors` → `model-bnb-nf4.safetensors`). Pass `-o <path>` to choose your own.

## Step 4 — Stamp GGUF metadata

A bare `--to gguf` writes a **tensor container** — correct tensors, but no key/value metadata, so no `Arch:` line and no runtime can load it as a model. anamnesis stays format-level: it writes whatever KV you hand it and derives none from the tensors. Supply it with `--gguf-kv key=value` (repeatable) or a `--gguf-metadata <file.json>`:

```
$ amn convert model.safetensors --to gguf \
    --gguf-kv general.architecture=llama \
    --gguf-kv general.name=Llama-3.2-1B-Instruct
```

Now the output carries an architecture line:

```
$ amn inspect model-gguf.gguf
Format:      GGUF v3
Arch:        llama
Tensors:     147
Total size:  2.79 GB
Dtypes:      BF16
Alignment:   32 bytes
```

`--gguf-kv` values are always strings; the JSON file form types each value (inference plus an explicit `{"type","value"}` escape hatch) and can carry arrays like a tokenizer vocabulary. The full grammar is in the [CLI reference](../cli-reference.md#gguf-metadata-flags). Producing *model-correct* KV — the architecture hyper-parameters and the multi-thousand-token tokenizer arrays — is a packaging concern for a downstream tool that reads the source `config.json` / `tokenizer.json`; anamnesis only provides the mechanism to write them.

When the **source is itself a GGUF**, its KV is inherited automatically, so `gguf → gguf` (dequantize-in-place) stays loadable without you re-supplying anything:

```
$ amn convert Qwen2.5-0.5B-Instruct-IQ2_M.gguf --to gguf -o qwen-scalar.gguf
Converting Qwen2.5-0.5B-Instruct-IQ2_M.gguf -> qwen-scalar.gguf
  169 dequantized to BF16
  Wrote 290 tensors -> qwen-scalar.gguf
```

```
$ amn inspect qwen-scalar.gguf
Format:      GGUF v3
Arch:        qwen2
Tensors:     290
Total size:  942 MB
Dtypes:      BF16, F32
Alignment:   32 bytes
```

The `Arch: qwen2` line survived the round trip — the source KV was carried through. Any `--gguf-kv` / `--gguf-metadata` you add merges *over* the inherited KV, so you override a key without losing the rest.

## The auto-chain, and what stays out of scope

The headline is that **every input reaches every current target** in one command, because the hub decouples readers from writers: FP8 / GPTQ / AWQ / BnB safetensors, GGUF, NPZ, and `.pth` all read *in*; `safetensors`, scalar `gguf`, and `bnb-nf4` all write *out*. Scalar dtypes are preserved end to end, so `.pth → safetensors` and an `NPZ`-`F32` → `GGUF` stay bit-for-bit lossless; only genuinely quantized tensors become `BF16`.

What is **not** here yet: the **quantized GGUF target columns** (`gguf-q4km`, FP8, IQ, TQ, MXFP4). Writing those needs encode kernels that land in a later phase; until then `--to gguf` always writes a *scalar* GGUF. Asking for a target whose Cargo feature is disabled returns a clear `Unsupported` error naming the feature to rebuild with, never a silent no-op.

## What you've learned

- `amn convert <file> --to <target>` routes every input through one `BF16` hub, so any format anamnesis reads reaches `safetensors` / scalar `gguf` / `bnb-nf4` in a single command.
- A quantized input **auto-chains** through `BF16` — the old dequantize-then-re-encode two-hop, and its temp file, are gone.
- `amn inspect` is a free header-only preview; use it to read the scheme and the `BF16` size budget (peak ≈ 2× the model) *before* converting.
- A `gguf` target is a scalar tensor container until you give it KV: `--gguf-kv` / `--gguf-metadata` stamp it verbatim, and a GGUF *source* has its KV inherited automatically.

To recover precision without changing container — the dequantize-only path — see [Dequantize a GGUF model to BF16](dequantize-a-gguf-model.md). For the safety angle on untrusted files, see [Inspect before you parse](inspect-before-you-parse.md) and the FAQ on [parsing untrusted input](../FAQ.md#parsing-untrusted-input).
