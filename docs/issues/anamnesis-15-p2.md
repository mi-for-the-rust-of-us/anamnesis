# anamnesis #15, reply 2

- **Target issue:** https://github.com/mi-for-the-rust-of-us/anamnesis/issues/15
- **Status:** Posted 2026-10-01, after v0.7.10 reached crates.io:
  https://github.com/mi-for-the-rust-of-us/anamnesis/issues/15#issuecomment-5932371637
  The issue was closed as completed with it.
- **Context:** The closing reply to [reply 1](anamnesis-15-p1.md). Phase 7.10
  implemented the issue's step 2 (parse **and** dequantise all three types),
  took up its CI suggestion, and found two things the issue did not know: that
  real `NVFP4` checkpoints carry a second, per-tensor scale beside each weight,
  and that `NVFP4` also ships as NVIDIA `ModelOpt` safetensors, which anamnesis
  misread.
- **Outcome:** Fixed in v0.7.10 (2026-10-01); issue closed. The text below is
  what was posted, unchanged except for one sentence about the drift job's
  triggers, corrected before posting when the job gained path-filtered
  `push` / `pull_request` triggers.
- **Lesson / Leverage angle:** The issue's own reproducer had a latent flaw: its
  64-element tensor is half a `Q1_0` block (128), so after the fix `t41.gguf` is
  refused for a different, correct reason. Worth saying on the thread rather
  than leaving a reader to rediscover it, which is why the reply below names it.
- **Accuracy flags:**
  1. **Every number below was measured, and the commands are in the repo.** The
     speed figures come from the matched protocols in
     `tests/bench_gguf_table_adhoc.rs` / `time_gguf_py.py` and
     `tests/cross_validation_nvfp4.rs` / `time_modelopt.py`; the
     correlation figures from comparing `amn remember` output against the
     original `BF16` checkpoints, fetched by range request.
  2. **The fold changes what `remember` writes for `NVFP4` files, deliberately.**
     It is the right output (llama.cpp applies the same factor at inference),
     but it is a behaviour, not only a capability, and the reply says so.
  3. **Commit hashes are from the Phase 7.10 branch.** They survive the
     `--no-ff` merge unchanged; re-check them against `main` before posting.

---

Thanks for the report, and for the reproducer: pinning the failure to a single
`u32` against a type-39 control made the diagnosis immediate. Everything below
is available in **anamnesis 0.7.10**.

**Fixed: all three types parse and dequantise.** `GgufType` gains `NVFP4`,
`Q1_0` and `Q2_0` (the enum is `#[non_exhaustive]`, so this is not breaking),
with kernels for all three. Your file now reads:

```console
$ amn inspect zeta-2.1-NVFP4.gguf
Format:      GGUF v3
Arch:        llama
Tensors:     740
Total size:  5.17 GB
Dequantized: 15.37 GB (BF16)
Dtypes:      NVFP4, F32, BF16
Alignment:   32 bytes
```

The kernels are cross-validated bit for bit at `F32` and with no `BF16`
tolerance. Since `gguf-py` cannot quantise any of the three or dequantise
`Q1_0` / `Q2_0`, their goldens come from ggml's own C (`quantize_row_*_ref` and
`dequantize_row_*` at llama.cpp `37b53fd`); a slice of `output.weight` from the
zeta file is one of the fixtures. `NVFP4` dequantises about 29× faster than
`gguf-py` on 65,536 values.

**One correction to the reproducer.** It declares 64 elements, which is half a
`Q1_0` block (128), so `t41.gguf` is now refused as malformed rather than as an
unknown type:

```
error: parse error: GGUF tensor: element count 64 not a multiple of block size 128 for type Q1_0
```

Any element count that is a multiple of 128 parses.

**Something the issue could not have known: `NVFP4` scales twice.** Every
`NVFP4` weight in this file has a companion `<name>.scale` tensor (`F32`, one
value), NVIDIA's second-level, per-tensor scale. llama.cpp multiplies it in at
inference time; `dequantize_row_nvfp4` alone does not apply it. On this model
the factors are between 9.3e-5 and 2.3e-3, so a weight dequantised without them
is 400 to 11 000 times too large. `amn remember` and `amn convert` now fold the
scale into the weight and leave the folded `.scale` out of the output:

```console
$ amn remember zeta-2.1-NVFP4.gguf -o zeta-bf16.safetensors
Converting zeta-2.1-NVFP4.gguf → zeta-bf16.safetensors
  740 tensors
  225 NVFP4 per-tensor scales folded into their weights
  Output: zeta-bf16.safetensors
```

Checked against `zed-industries/zeta-2.1`'s original `BF16` weights, the folded
output has a least-squares scale of 0.995 to 0.998 and a correlation of 0.9955;
the rest is what 4-bit `E2M1` costs. The per-tensor API
(`dequantize_gguf`, `ParsedGguf::dequantize_tensor_as`) still returns the raw
block decode, as its documentation now says.

**Your CI suggestion, implemented.** `tests/fixtures/ggml_types.txt` is a
snapshot of `ggml.h`'s live `GGML_TYPE_*` entries at a pinned commit; a test
holds the parser to it exactly (every snapshot entry parses under upstream's
name, everything else is refused), and a weekly workflow diffs the snapshot
against llama.cpp `master` and fails naming the new types. Apart from that
schedule it runs only on changes to the check itself, so upstream churn cannot
turn an unrelated pull request red.

**`NVFP4` in safetensors, too.** NVIDIA's `ModelOpt` exports the same format as
safetensors (`nvidia/Llama-3.1-8B-Instruct-NVFP4`), and anamnesis was misreading
those files as fine-grained `FP8`. They are now recognised in every build and
dequantised behind a new `nvfp4` feature, bit-exact against `modelopt`'s own
`NVFP4QTensor.dequantize` and about 21× faster than it.

Thanks again. Closing as fixed in 0.7.10.
