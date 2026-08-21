# Inspect before you parse (untrusted input)

*Read a model file's header — counts, sizes, scheme — before committing to a full parse. The cheap, safe first move for any file you didn't create yourself.*

*~800 words · about 3 min read*

<!-- Last updated: 2026-08-07, anamnesis v0.7.2 -->

<!--
STYLE CONVENTIONS for editing this tutorial — keep growth consistent.
(Adapted from the sibling hf-fetch-model tutorials so the two read alike.)

1. Tone: match the FAQ. Conversational, address the reader as "you", short
   paragraphs over bullet lists where prose works.
2. Output blocks are real, captured from `amn` against the in-repo
   `tests/fixtures/` files (and a throwaway garbage file for the rejection
   demo). Paste exact output, do not paraphrase. The hostile-file header
   length (7309940746704154478) and the 104857600-byte cap are real — if
   the cap constant changes, re-capture.
3. Shell: commands are identical across platforms; only split PowerShell vs
   bash when an env var or path actually differs.
4. Scope: features that ship today. The library snippet uses the public
   `parse_*_with_limits` / `inspect_*_from_reader` API and `ParseLimits`
   (v0.6.3+). Keep it compiling-shaped but illustrative.
5. Length budget: under 300 lines total, including embedded outputs. Update
   the word-count + reading-time line whenever the prose changes (~250 wpm,
   code blocks excluded).
-->

## Contents

- [Why inspect first?](#why-inspect-first)
- [The 30-second answer](#the-30-second-answer)
- [Step 1 — `inspect`: the header-only preview](#step-1--inspect-the-header-only-preview)
- [Step 2 — `parse`: the full commitment](#step-2--parse-the-full-commitment)
- [Step 3 — what a hostile file looks like](#step-3--what-a-hostile-file-looks-like)
- [Step 4 — bound it in code with `ParseLimits`](#step-4--bound-it-in-code-with-parselimits)
- [What you've learned](#what-youve-learned)

## Why inspect first?

A tensor archive you downloaded is attacker-controllable. A malicious `.safetensors`, `.pth`, `.npz`, or `.gguf` can declare arbitrary dimensions, point at arbitrary offsets, or claim a multi-gigabyte allocation from a few kilobytes on disk — the classic decompression-bomb and unchecked-allocation shapes. You do not want to discover that by handing the whole file to a full parser and watching a worker OOM.

So the rule for any file you did not create is: **inspect → check it against your policy → only then parse.** `amn inspect` reads the *header* — counts, dtypes, declared sizes, quantization scheme — and for `.gguf` / `.npz` / `.pth` it never touches the weight bodies at all. It is cheap, and it tells you what a full parse is about to commit to.

## The 30-second answer

```
amn inspect suspicious-model.safetensors     # header-only summary, no weight bodies
# …decide whether the declared sizes are sane for your machine…
amn parse suspicious-model.safetensors        # full parse, only if you're satisfied
```

If the file is hostile or corrupt, `inspect` fails with a clear error and a non-zero exit code — never a crash. The rest of this tutorial shows each step with real output.

## Step 1 — `inspect`: the header-only preview

Point `amn inspect` at a file and you get a compact summary. Here it is on a fine-grained FP8 safetensors fixture:

```
$ amn inspect tests/fixtures/safetensors_reference/fp8.safetensors
Format:      Fine-grained FP8 (E4M3), 128x128 blocks
Quantized:   1 tensors (weights) + 1 scale tensors (F32)
Passthrough: 1 tensors (norms, embeddings)
Size:        96 B (FP8) -> 144 B (BF16)
Lethe took:  ~48 B of precision
```

The headline is what you learn *without committing*: the format and scheme, how many tensors are quantized vs passed through, and the size, both on disk and what it will become if you dequantize. On a real model that size line is the load-bearing one.

On a quantized GGUF the second half of that is the number you actually need, and it is the one you cannot work out yourself: the expansion ratio is per-kernel, so a `Q2_K` tensor and a `Q6_K` tensor of the same element count take different space on disk and the *same* space in memory. `inspect` reports both, at the width you say you intend:

```console
$ amn inspect SmolLM2-135M-Instruct-Q6_K.gguf --to f32
Format:      GGUF v3
Arch:        llama
Tensors:     272
Total size:  130 MB
Dequantized: 513 MB (F32)
Dtypes:      Q8_0, F32, Q6_K
Alignment:   32 bytes
```

`--to` defaults to `bf16` and is worth setting deliberately, because it is the whole point of the line: the same file reports 257 MB at `bf16` and 513 MB at `f32`, and a budget check against the wrong one under-reserves by exactly 2×. (On `.npz` and `.pth` nothing is quantized, so the flag is accepted and changes no figure.) Both numbers come out of the header; not a single weight byte is read.

## Step 2 — `parse`: the full commitment

When you are satisfied, `amn parse` does the real work and reports the full structure:

```
$ amn parse tests/fixtures/safetensors_reference/fp8.safetensors
3 tensors parsed
    1 quantized   Fine-grained FP8 (E4M3), 128x128 blocks
    1 scale       F32
    1 passthrough BF16 (norms, embeddings, lm_head)
File: 96 B
```

On a quantization scheme with more moving parts, `parse` shows them — here GPTQ, with its scale, packed zero-point, and activation-order index:

```
$ amn parse tests/fixtures/safetensors_reference/gptq.safetensors
5 tensors parsed
    1 quantized   GPTQ
    1 scale       F16
    1 zero-point  I32 (packed)
    1 g_idx       I32 (activation-order)
    1 passthrough F16 (norms, embeddings, lm_head)
File: 1.5 KB
```

`parse` is the step that reads structure in full, so it is the one to gate behind an `inspect` you trust.

## Step 3 — what a hostile file looks like

Here is the payoff. A safetensors file begins with an 8-byte little-endian header length. Feed `inspect` a file of random bytes and those first 8 bytes decode to an absurd length — which anamnesis rejects against its 100 MiB header cap *before allocating anything*:

```
$ amn inspect hostile.safetensors
error: parse error: safetensors header length 7309940746704154478 exceeds 104857600-byte cap
```

That is a declared header of ~7.3 **exabytes** turned away in microseconds — no 7-exabyte allocation attempt, no OOM, exit code `1`. The same fail-closed behaviour covers a wrong-format file:

```
$ amn inspect bad.gguf
error: parse error: GGUF: invalid magic (expected `GGUF`/0x46554747, got 0x58585858)
```

Every parser entry point shares this discipline: checked arithmetic on header-derived sizes, an allocation cap *before* any `vec!`, a strict allowlist in the `.pth` pickle VM (it never invokes Python callables), and a vendored read-only ZIP reader for `.npz` / `.pth`. A malformed or hostile file becomes a clean `Err` and a non-zero exit, not a dead process.

## Step 4 — bound it in code with `ParseLimits`

The CLI uses generous default caps. When you embed anamnesis in a service — a multi-tenant backend, an edge device — you want *your* ceilings, not the server-scale defaults. The library API takes a caller-supplied `ParseLimits` budget, enforced fail-fast before allocation:

```rust
use anamnesis::{ParseLimits, parse_with_limits};

// Tighten the parser to this worker's slot — reject anything bigger up front.
// `ParseLimits::default()` is permissive; the `with_*` builders set your ceilings.
let limits = ParseLimits::default()
    .with_max_single_alloc(512 * 1024 * 1024)       // no single tensor over 512 MiB
    .with_max_total_bytes(2 * 1024 * 1024 * 1024)   // 2 GiB cumulative parse-time heap
    .with_max_item_count(4096)                       // cap the declared tensor count
    .with_max_decompression_ratio(100);             // zip-bomb guard for .npz / .pth

let model = parse_with_limits("upload.safetensors", &limits)?; // over budget → Err(LimitExceeded)
```

`ParseLimits::default()` is permissive (the CLI's behaviour); you tighten the axes that matter for your environment. For a header-only gate over data you are streaming or fetching remotely — without a file on disk — use the reader-based `inspect_*_from_reader` calls to read the declared totals first, check them against your policy, and only then parse. That is the same *inspect → check → parse* pattern, made programmatic.

## What you've learned

- For any file you didn't create, the safe order is **inspect → check against your policy → parse**.
- `amn inspect` is a cheap, header-only preview (it doesn't read weight bodies for `.gguf` / `.npz` / `.pth`); `amn parse` is the full commitment, so gate it behind an inspect you trust.
- A hostile file is rejected with a clear error and a non-zero exit — anamnesis caps declared sizes before allocating, so a 7-exabyte header claim costs microseconds, not an OOM.
- In a service, pass a `ParseLimits` budget (or use `inspect_*_from_reader` for a streamed/remote gate) to enforce *your* ceilings instead of the server-scale defaults.

For the questions this raises — *"is it safe to parse a file from a stranger?"*, *"how do I bound memory?"* — see the FAQ on [parsing untrusted input](../FAQ.md#parsing-untrusted-input). To go the other direction and recover full-precision weights once you trust a file, see [Dequantize a GGUF model to BF16](dequantize-a-gguf-model.md).
