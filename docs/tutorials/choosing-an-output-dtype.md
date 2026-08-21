# Choosing an output dtype

*Decide between `bf16`, `f32`, and `f16` when you dequantize, and understand what each one costs you.*

*~1200 words · about 5 min read*

<!-- Last updated: 2026-08-15, anamnesis v0.7.4 -->

<!--
STYLE CONVENTIONS for editing this tutorial — keep growth consistent.
(Adapted from the sibling tutorials so they all read alike.)

1. Tone: match the FAQ. Conversational, address the reader as "you", short
   paragraphs over bullet lists where prose works.
2. NO EM-DASHES in the prose of this file. Rephrase, or use a colon,
   parenthesis, or full stop. Do not substitute an en-dash or a hyphen
   for one either: the point is the sentence, not the glyph.
3. Every number in this file is measured, not estimated, and is sourced
   from docs/perf-experiments.md (Experiments 13, 14, 15) or from the
   cross-validation suites. If you change a number, change it there first
   and cite the same source.
4. Scope: features that ship today. `remember --to` landed in v0.7.4 and
   `convert --out-dtype` in v0.7.3; do not document unshipped widths.
5. Length budget: under 300 lines total. Update the word-count + reading
   time line at the top whenever the prose changes non-trivially
   (~250 wpm, code blocks excluded).
-->

## Contents

- [The short answer](#the-short-answer)
- [What the flag actually controls](#what-the-flag-actually-controls)
- [Why `bf16` is the default](#why-bf16-is-the-default)
- [When `f32` earns its size](#when-f32-earns-its-size)
- [Why `f16` is a trap](#why-f16-is-a-trap)
- [Checking the cost before you commit](#checking-the-cost-before-you-commit)
- [What you've learned](#what-youve-learned)

## The short answer

```
amn remember model-fp8.safetensors --to bf16    # default, omit it
amn remember model-fp8.safetensors --to f32     # the reference value itself
amn remember model-fp8.safetensors --to f16     # only if a consumer demands it
```

On the `convert` path the same choice is spelled `--out-dtype`:

```
amn convert model-Q4_K_M.gguf --to safetensors --out-dtype f32
```

Stay on `bf16` unless you have a specific reason not to. The rest of this page is about what those reasons are.

## What the flag actually controls

This is the part that surprises people, so it comes first.

The flag governs the tensors anamnesis **dequantizes**. It does not touch anything else. A model file is not uniformly quantized: the weight matrices are, but the norms, biases, and often the embeddings are stored as plain `F16`, `BF16`, or `F32` and are copied through untouched. Those are passthrough tensors, and they keep their source dtype whatever you ask for.

So a `remember --to f32` output is legitimately a mixed-dtype file. An `F16` norm in the source is still an `F16` norm in the output. An `F32` tensor is byte-identical to the one you started with.

That is deliberate rather than an oversight. A passthrough tensor is copied, never decoded, so widening it would invent precision that was never in the file while doubling its size on disk. If what you want is a file where every tensor has one dtype, what you want is a cast pass, and that is a different operation from dequantization.

## Why `bf16` is the default

Three reasons, in order of how much they should matter to you.

It is what the ecosystem uses. Hugging Face serves weights in `BF16`, and candle, burn, and tch all load it without special handling. If you are dequantizing in order to load the result somewhere, `bf16` is the format that just works.

It is half the bytes. These kernels are memory-bandwidth bound, so output width translates fairly directly into wall-clock time. Two bytes per element instead of four is the single biggest lever on how long the run takes.

It matches the precision that is actually there. A quantized weight does not contain 24 bits of significand to recover. It is a small integer times a scale, and once you have multiplied those two together, most of the low bits of the `f32` result are an artifact of the multiply rather than information from the file.

That last point has a limit, though, and it is what the next section is about.

## When `f32` earns its size

`f32` is not "more precision than the file contains". It is the value the reference implementation itself computes, with anamnesis's own narrowing step removed entirely.

The difference is real and measurable. Rounding to `BF16` keeps 8 bits of significand. In the cross-validation fixtures this project ships, between 38 and 98 percent of dequantized values carry mantissa bits that `BF16` cannot represent, depending on the family and the model. The low end is the `BnB` `FP4` fixtures, whose 16-entry codebook lands on `BF16`-representable values more often; the `GPTQ`, `AWQ`, and `BnB` `NF4` fixtures all sit above 77 percent. Those bits are not noise. They are the reference implementation's answer.

Three situations where you want them:

**You are cross-validating against PyTorch.** If you are comparing anamnesis output against `gguf-py`, GPTQModel, AutoAWQ, or bitsandbytes, comparing at `BF16` throws away 16 mantissa bits before the comparison starts. This is not hypothetical: widening anamnesis's own test suite to `f32` in v0.7.4 found a genuine 1-ULP defect in the `BnB` `INT8` kernel that five releases of `BF16` cross-validation had reported as 0 mismatches. See the [FAQ entry](../FAQ.md#is-anamnesis-still-bit-exact-against-pytorch-at-f32) for what it was.

**You are debugging a numerical discrepancy.** If a model behaves differently under anamnesis than under its native runtime, you want to know whether the dequantized weights differ or something downstream does. At `bf16` you cannot tell, because the rounding hides differences smaller than half a ULP.

**Your pipeline computes in `float32` anyway.** If the next step widens the weights back to `f32`, doing it from a `bf16` file just means the low bits are zeros instead of values. You paid for a narrowing you did not want.

What it costs: double the output bytes, and slower in roughly the way that implies. On the `convert` path the kernel-level figure was measured at 1.79x slower than `bf16` against exactly 2.00x the output bytes, so the cost is the doubled write and essentially nothing else. End to end it came out at 1.54x to 1.61x, less than the kernel figure because the fixed parse cost does not scale with output width. The full numbers are in [`docs/perf-experiments.md`](../perf-experiments.md), Experiment 13.

## Why `f16` is a trap

`f16` and `bf16` are both 2 bytes per element, which makes `f16` look like a free upgrade: same size, 3 more bits of significand (11 versus 8). It is not free, and it costs you twice: once in exponent range, and once in time.

**It is 2x to 3x slower than `bf16`, at identical output size.** Measured across all seven dequantisation families (criterion medians, 4096 × 11008, x86-64):

| Kernel | `bf16` | `f16` | ratio |
|---|---:|---:|---:|
| `gguf_q4_k` | 25.70 ms | 79.87 ms | **3.11x** |
| `bnb_int8` | 24.84 ms | 76.95 ms | **3.10x** |
| `gptq_int4` | 28.78 ms | 78.04 ms | **2.71x** |
| `awq_int4` | 38.20 ms | 101.10 ms | **2.65x** |
| `fp8_fine_grained` | 43.19 ms | 107.28 ms | **2.48x** |
| `bnb_nf4` | 45.43 ms | 112.04 ms | **2.47x** |
| `fp8_per_tensor` | 46.55 ms | 94.20 ms | **2.02x** |

`aarch64` agrees at 2.10x to 2.93x on the same seven, so this is not one machine's quirk. Note what it means for the comparison above: **`f16` is slower than `f32`**, which writes twice the bytes. The cost is the conversion, not the traffic — on x86-64 the `F16C` instruction narrows 4 lanes per instruction where `bf16`'s shift-and-round does 8; on `aarch64` the narrowing is not inlined at all. The full write-up is [`docs/perf-experiments.md`](../perf-experiments.md), Experiment 18.

The range cost is the one more likely to bite your *results*, so it comes next.

`bf16` has the same exponent range as `f32`. `f16` does not. It overflows to infinity above 65504 and flushes to zero below about 2⁻²⁴. That is a much narrower window, and dequantized values can leave it: the `BnB` `INT8` path multiplies a per-row scale by an integer of up to 127, which is exactly the shape of arithmetic that can exceed 65504 on a large scale value.

anamnesis follows plain IEEE semantics there. It does not saturate, and it does not clamp. A value above the range becomes an infinity, and a value below it becomes a zero, which is what NumPy and PyTorch produce for the same conversion. This is the right behaviour, because silently clamping would turn an overflow into a plausible-looking wrong number, but it does mean a bad `f16` choice fails loudly in your weights rather than at conversion time.

Reach for `f16` when a downstream consumer specifically requires IEEE half and you have reason to believe your values fit. Otherwise `bf16` is the 2-byte format you want.

## Checking the cost before you commit

Two costs, and `inspect` only answers one of them. **It reports size, not time** — and as the table above shows, the two do not track each other: `f16` and `bf16` produce byte-identical output sizes while differing 2x to 3x in run time. For the time cost, use the table above rather than reasoning from the size figure.

For size: the estimate `inspect` reports assumes a width, so ask it for the one you actually intend. From the library:

```rust
use anamnesis::{InspectOptions, TargetDtype, parse};

let model = parse("model-fp8.safetensors")?;
let info = model.inspect_with_options(
    &InspectOptions::new().with_output_dtype(TargetDtype::F32),
);
println!("{info}");
```

The rendered size line names the width it assumed, so the number can never be read without knowing what it was computed for.

`amn inspect --to f32` reports the same figure from the command line, and every format answers it — `GGUF` included, where the estimate is the one you cannot derive from the file size. Both arrived in **v0.7.6**; in v0.7.4 and v0.7.5 the dtype-aware figure was a library-only API and the command reported the `bf16` estimate whatever you intended.

*(`InspectOptions` is taken by reference since v0.7.6, when it gained a `limits` field and stopped being `Copy`. If you are upgrading from v0.7.4, add the `&`.)*

## What you've learned

The output dtype is a real choice with a real cost, and the default is the right answer most of the time.

`bf16` is what the ecosystem loads and is half the bytes. `f32` gives you the reference implementation's own value with no narrowing of anamnesis's, at double the size and roughly 1.5x to 1.8x the time, and it is the right call when you are validating, debugging, or feeding an `f32` pipeline. `f16` trades exponent range for significand bits and should be a deliberate choice, not a default.

And whichever you pick, the flag applies to dequantized tensors only. Your passthrough tensors keep their source dtype, and that mixed-dtype output file is working as designed.

Next: [Dequantize a GGUF model to BF16](dequantize-a-gguf-model.md) walks the whole pipeline end to end, and the [FAQ](../FAQ.md#dequantizing-and-converting) has the short-form answers.
