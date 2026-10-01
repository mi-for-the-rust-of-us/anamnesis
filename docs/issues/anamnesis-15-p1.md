# anamnesis #15, reply 1 (Posted)

- **Target issue:** https://github.com/mi-for-the-rust-of-us/anamnesis/issues/15
- **Status:** Posted (2026-10-01). The body below matches what is live verbatim.
- **Context:** A consumer question about `zed-industries/zeta-2` NVFP4 availability
  led to `hf-fm inspect` on `distaste447/zeta-2.1-NVFP-GGUF`, which anamnesis
  refused to parse. Upstream `ggml` has added three `ggml_type` discriminants
  since anamnesis's table was written, and one of them (`NVFP4 = 40`) is already
  on 100+ Hub repositories. This is a **read-side** coverage gap, distinct from
  the encode work in Phase 8.5.
- **Outcome:** Fixed by Phase 7.10 on branch `phase-7.10-gguf-types` (to ship
  as v0.7.10): the three types parse and dequantise (`3259ab8`, `34fa177`),
  cross-validated against ggml's own C (`41fcf30`), `NVFP4`'s per-tensor scale
  is folded by `remember` / `convert` (`6c523d3`), and the suggested CI check
  exists (`0d00874`). The closing reply is drafted in
  [`anamnesis-15-p2.md`](anamnesis-15-p2.md).
- **Lesson / Leverage angle:** The GGUF type enum is a moving upstream target
  that anamnesis tracks by hand, with no mechanism that notices drift. The issue
  proposes one, because the next three types will land the same way.
- **Accuracy flags:**
  1. **Verified on the stock released binary, not a local build.** A plain
     `cargo install anamnesis` gives `amn 0.7.8`, which reproduces directly, with
     no patch and no unusual feature combination. Also confirmed against a local
     build of `a27e58c` and against the previous `0.7.7` release, so this is not
     a regression introduced by either.
  2. **The control is load-bearing and is stated.** `ggml_type 39` (`MXFP4`)
     parses from a byte-identical fixture differing only in that `u32`, so the
     failure is the discriminant and not a malformed fixture.
  3. **Upstream reference is pinned to a commit, not `master`.** `ggml.h` and
     `ggml-common.h` are quoted at
     `37b53fd4545847188fdad29e38ba57875efc8228`, because `master` moves and a
     later reader needs to know what was true when this was written.
  4. **Prevalence figures are approximate, and the method is stated.** The Hub
     search API was queried per type name and the results filtered by repo-name
     substring. NVFP4 returned a full page, so "100+" is a floor, not a count.
     My first pass fuzzy-matched `Q8_0` into the `Q1_0` tally; corrected to 2.
  5. **Block layouts are transcribed from upstream, not derived.** The bits per
     weight are my arithmetic from those structs and are shown so they can be
     checked.
  6. **The mislabeled error prefix is hf-fm's, not anamnesis's.** `hf-fm` reports
     `safetensors header error for <file>.gguf: failed to parse GGUF: ...`.
     `amn inspect` on the same fixture prints no such prefix. Deliberately kept
     out of this issue; it belongs in hf-fetch-model.
  7. **Not a safety bug, and the issue says so.** anamnesis fails closed with a
     precise error and no panic. This is coverage, not hardening.

---

## Summary

`anamnesis` 0.7.8 cannot parse a GGUF file containing any tensor of
`ggml_type` **40, 41 or 42**. `GgufType::from_u32` accepts discriminants up to
`39` (`MXFP4`) and rejects everything above:

```
error: unsupported format `GGUF`: unknown ggml_type discriminant 40
```

Because the rejection happens while reading the tensor-info table, the **whole
file is refused**. `amn inspect` cannot list tensors, print metadata, or report a
size for an otherwise well-formed model. There is no partial or degraded read.

The three types, from
[`ggml/include/ggml.h@37b53fd`](https://github.com/ggml-org/llama.cpp/blob/37b53fd4545847188fdad29e38ba57875efc8228/ggml/include/ggml.h#L425-L433):

```c
GGML_TYPE_TQ2_0   = 35,
GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)        <- anamnesis stops here
GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
GGML_TYPE_Q1_0    = 41,
GGML_TYPE_Q2_0    = 42,
GGML_TYPE_COUNT   = 43,
```

## Reproduction, without downloading a model

A 160-byte GGUF is enough. The control case is the point: the only difference
between the file that parses and the files that do not is a single `u32`.

```python
# mkgguf.py, usage: python mkgguf.py <ggml_type> <out.gguf>
import struct, sys
def s(x): return struct.pack('<Q', len(x)) + x
t, out = int(sys.argv[1]), sys.argv[2]
b  = b'GGUF' + struct.pack('<I', 3)                 # magic + version 3
b += struct.pack('<Q', 1) + struct.pack('<Q', 0)    # 1 tensor, 0 KV pairs
b += s(b'blk.0.attn_q.weight')                      # tensor name
b += struct.pack('<I', 1) + struct.pack('<Q', 64)   # n_dims = 1, dims = [64]
b += struct.pack('<I', t) + struct.pack('<Q', 0)    # ggml_type, data offset
b += b'\0' * ((32 - len(b) % 32) % 32)              # pad to alignment 32
b += b'\0' * 64                                     # tensor data
open(out, 'wb').write(b)
```

```console
$ for t in 39 40 41 42; do python mkgguf.py $t t$t.gguf; done
$ amn inspect t39.gguf | grep Tensors:
  Tensors:     1                      # MXFP4, parses: the control
$ amn inspect t40.gguf
error: unsupported format `GGUF`: unknown ggml_type discriminant 40
$ amn inspect t41.gguf
error: unsupported format `GGUF`: unknown ggml_type discriminant 41
$ amn inspect t42.gguf
error: unsupported format `GGUF`: unknown ggml_type discriminant 42
```

Verified on `anamnesis 0.7.8` as published (`cargo install anamnesis`), and
again on a local build of `a27e58c`, and on the earlier `0.7.7` release.
Windows 11, `x86_64-pc-windows-msvc`.

## Reproduction against a real model

```console
$ hf-fm inspect distaste447/zeta-2.1-NVFP-GGUF zeta-2.1-NVFP4.gguf
error: ... failed to parse GGUF: unknown ggml_type discriminant 40
```

A 5.18 GiB NVFP4 quantization of `zed-industries/zeta-2.1`, tagged `nvfp4` and
`blackwell`. Header-only, so it costs a range request rather than a download.

## What is missing, exactly

Diffing the live (uncommented) entries of upstream's enum against
`GgufType::from_u32`'s arms:

| Discriminant | Upstream | In anamnesis 0.7.8 |
|---|---|:---:|
| 39 | `GGML_TYPE_MXFP4` | yes |
| **40** | **`GGML_TYPE_NVFP4`** | **no** |
| **41** | **`GGML_TYPE_Q1_0`** | **no** |
| **42** | **`GGML_TYPE_Q2_0`** | **no** |

Three, and only three. `GGML_TYPE_COUNT = 43` is a sentinel, and 31/32/33 and
36/37/38 are commented out upstream, so they are correctly absent.

## Block layouts

From
[`ggml/src/ggml-common.h@37b53fd`](https://github.com/ggml-org/llama.cpp/blob/37b53fd4545847188fdad29e38ba57875efc8228/ggml/src/ggml-common.h):

```c
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16  // sub-block size for per-group scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;

#define QK1_0 128
typedef struct { ggml_half d; uint8_t qs[QK1_0 / 8]; } block_q1_0;

#define QK2_0 64
typedef struct { ggml_half d; uint8_t qs[QK2_0 / 4]; } block_q2_0;
```

Which gives the two numbers `block_size` and `type_size` need:

| Type | `block_size` | `type_size` | bits/weight |
|---|---:|---:|---:|
| `NVFP4` | 64 | 36 B (4 + 32) | 4.50 |
| `Q1_0` | 128 | 18 B (2 + 16) | 1.125 |
| `Q2_0` | 64 | 18 B (2 + 16) | 2.25 |

`NVFP4` is the structurally novel one: unlike `MXFP4`'s single E8M0 byte per
32-element block, it carries **four UE4M3 sub-block scales per 64-element
block**, so the scale is per 16 elements and is itself a float8 rather than a
power-of-two exponent.

## How much each one matters

Hub repositories whose names carry the type, via the search API:

| Type | Repos | Note |
|---|---:|---|
| `NVFP4` | 100+ | A full result page, so this is a floor. Widespread on recent Qwen and DeepSeek quantizations, and growing with Blackwell adoption. |
| `Q1_0` | 2 | |
| `Q2_0` | 1 | |

`NVFP4` is the one with real exposure. `Q1_0` and `Q2_0` are rare today, but they
sit in the same contiguous range and cost little once the first is done.

## Where the change goes

Mirroring what `MXFP4` already touches in `src/parse/gguf.rs`:

- the `GgufType` enum variant (around line 222)
- `to_u32` (around line 263) and `from_u32` (around line 307)
- `block_size` (around line 341)
- `type_size` (around line 451)
- the dense index used for the per-type table (around line 508)
- the `Display` name (around line 588)
- the round-trip test that pins `to_u32` against `from_u32`

Dequantization would additionally need a kernel in `src/remember/gguf.rs`, where
`MXFP4`'s lives.

## One scope question

These are separable, and the split may be worth taking:

1. **Parse only.** Add the three variants with their block and type sizes. Small
   and mechanical, and it restores `inspect` / `list` / size reporting on the
   100+ NVFP4 models now on the Hub. **The architecture already degrades
   correctly for this**: the dequant dispatch in `src/remember/gguf.rs:424` ends
   in a `_ =>` arm returning `AnamnesisError::Unsupported`, so a type that
   parses without being dequantizable needs no new error path. `remember` and `convert` would
   refuse it with a clear message while `inspect` works, which is the useful half
   for anyone deciding whether to download a file.
2. **Parse plus dequantize.** Three more kernels, each needing a bit-exact
   fixture from the canonical implementation per the project's own rule. `NVFP4`
   is the interesting one: among the 24 GGUF dequant kernels anamnesis ships,
   none has a float8 per-sub-block scale. `MXFP4` is closest and still differs,
   carrying one E8M0 power-of-two exponent per 32-element block against `NVFP4`'s
   four UE4M3 scales per 64.

Step 1 alone would have turned this report into a successful `inspect`.

## A suggestion beyond the three types

Nothing here notices when upstream adds a discriminant. The failure surfaced only
because someone pointed the tool at a model that happened to use one. A CI job
that fetches `ggml.h`, extracts the uncommented `GGML_TYPE_* = N` entries and
compares them against `from_u32`'s arms would turn a future gap into a failing
check rather than a user's error message. The extraction is a one-line `grep -oE`
against a pinned raw URL, and pinning the commit it last agreed with gives the
diff something to report.

## What this is not

anamnesis fails **closed**, with a precise error naming the offending
discriminant and no panic, which is the correct behaviour for an unknown type
from untrusted input. This is a coverage gap against a moving upstream, not a
hardening defect.
