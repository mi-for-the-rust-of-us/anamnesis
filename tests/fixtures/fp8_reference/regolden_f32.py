#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Add an F32 golden to each committed FP8 fixture, in place.

Why this exists
---------------
Every FP8 cross-validation to date compared anamnesis' output against a
reference **rounded to BF16 first**, discarding 16 mantissa bits. Phase 7.4
makes the output width caller-chosen, so the F32 path needs a golden that was
never narrowed. This script produces one.

Why it re-goldens rather than re-running generate.py
----------------------------------------------------
`generate.py` slices real models out of the HuggingFace cache, which are
gitignored and multi-GB. But the committed ``.bin`` already carries the **raw
FP8 weight bytes and the raw scale bytes** -- everything the reference needs.
So this reads those back, re-runs the canonical PyTorch dequant on them, and
appends the F32 result. No model download, and the inputs are provably the same
bytes the BF16 golden was computed from, because they come from the same file.

Canonical-reference discipline (the v0.6.4 rule)
------------------------------------------------
The reference is PyTorch's own ``float8_e4m3fn`` -> ``float32`` cast, exactly as
`generate.py` uses it. Nothing here reimplements the E4M3 bit layout; the whole
point of the rule is that a hand-rolled decode can agree with a hand-rolled
fixture and both be wrong.

**The F32 golden is NOT derived from the BF16 one.** It is computed from the
raw bytes and then, separately, the existing BF16 golden is re-derived and
checked against what is already in the file. If those disagree the script
refuses to write, because that would mean the raw bytes and the stored golden
had drifted apart and neither could be trusted.

Container format
----------------
v1 (pre-v0.7.4) had no magic and no version::

    scheme u32 | scale_dtype u32 | rows u32 | cols u32
    weight_len u32 | scale_len u32 | expected_len u32
    [weight][scale][expected BF16]

v2 adds a magic + version prefix and the F32 golden, mirroring what
`gguf_reference/generate_gguf.py` did for the GGUF fixtures in v0.7.3::

    "AMNF" | version u32 = 2
    scheme u32 | scale_dtype u32 | rows u32 | cols u32
    weight_len u32 | scale_len u32 | bf16_len u32 | f32_len u32
    [weight][scale][expected BF16][expected F32]

The magic is what lets the Rust reader reject a v1 file with a clear message
instead of reading the header at the wrong offsets and reporting nonsense.

Usage::

    python regolden_f32.py            # rewrite every *.bin in this directory
    python regolden_f32.py --check    # verify only, write nothing
"""

import argparse
import struct
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
MAGIC = b"AMNF"
VERSION = 2

# scale_dtype ids, matching the Rust reader and generate.py
DTYPE_F32, DTYPE_BF16, DTYPE_F16 = 0, 1, 2
_TORCH_SCALE_DTYPE = {
    DTYPE_F32: torch.float32,
    DTYPE_BF16: torch.bfloat16,
    DTYPE_F16: torch.float16,
}

SCHEME_FINE_GRAINED, SCHEME_PER_TENSOR, SCHEME_PER_CHANNEL = 0, 1, 2
BLOCK = 128


def parse_v1(data: bytes) -> dict:
    """Reads the pre-v0.7.4 layout."""
    scheme, scale_dtype, rows, cols, w_len, s_len, e_len = struct.unpack_from("<7I", data, 0)
    off = 28
    weight = data[off : off + w_len]
    off += w_len
    scale = data[off : off + s_len]
    off += s_len
    expected_bf16 = data[off : off + e_len]
    return {
        "scheme": scheme,
        "scale_dtype": scale_dtype,
        "rows": rows,
        "cols": cols,
        "weight": weight,
        "scale": scale,
        "expected_bf16": expected_bf16,
    }


def parse_v2(data: bytes) -> dict:
    """Reads the v0.7.4 layout (already re-goldened)."""
    version = struct.unpack_from("<I", data, 4)[0]
    if version != VERSION:
        raise SystemExit(f"unsupported container version {version}")
    scheme, scale_dtype, rows, cols, w_len, s_len, b_len, f_len = struct.unpack_from(
        "<8I", data, 8
    )
    off = 40
    weight = data[off : off + w_len]
    off += w_len
    scale = data[off : off + s_len]
    off += s_len
    expected_bf16 = data[off : off + b_len]
    off += b_len
    expected_f32 = data[off : off + f_len]
    return {
        "scheme": scheme,
        "scale_dtype": scale_dtype,
        "rows": rows,
        "cols": cols,
        "weight": weight,
        "scale": scale,
        "expected_bf16": expected_bf16,
        "expected_f32": expected_f32,
    }


def dequant_f32(fx: dict) -> torch.Tensor:
    """The canonical PyTorch dequant, stopping at f32.

    Byte-for-byte the arithmetic in ``generate.py``'s ``dequant_pytorch``, with
    its final ``.to(torch.bfloat16)`` removed. Keeping the two in the same shape
    is deliberate: if the formula ever changes, both must change together.
    """
    rows, cols = fx["rows"], fx["cols"]

    weight = torch.frombuffer(bytearray(fx["weight"]), dtype=torch.float8_e4m3fn)
    weight_f32 = weight.reshape(rows, cols).to(torch.float32)

    scale_dtype = _TORCH_SCALE_DTYPE[fx["scale_dtype"]]
    scale = torch.frombuffer(bytearray(fx["scale"]), dtype=scale_dtype)
    scale_f32 = scale.to(torch.float32)

    scheme = fx["scheme"]
    if scheme == SCHEME_FINE_GRAINED:
        scale_rows = (rows + BLOCK - 1) // BLOCK
        scale_cols = (cols + BLOCK - 1) // BLOCK
        scale_grid = scale_f32.reshape(scale_rows, scale_cols)
        result = torch.zeros(rows, cols, dtype=torch.float32)
        for br in range(scale_rows):
            for bc in range(scale_cols):
                r0, r1 = br * BLOCK, min((br + 1) * BLOCK, rows)
                c0, c1 = bc * BLOCK, min((bc + 1) * BLOCK, cols)
                result[r0:r1, c0:c1] = weight_f32[r0:r1, c0:c1] * scale_grid[br, bc]
    elif scheme == SCHEME_PER_TENSOR:
        result = weight_f32 * scale_f32
    elif scheme == SCHEME_PER_CHANNEL:
        result = weight_f32 * scale_f32.reshape(rows, 1)
    else:
        raise SystemExit(f"unknown scheme {scheme}")

    return result


def to_bytes(t: torch.Tensor) -> bytes:
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def process(path: Path, check_only: bool) -> bool:
    data = path.read_bytes()
    already_v2 = data[:4] == MAGIC
    fx = parse_v2(data) if already_v2 else parse_v1(data)

    result_f32 = dequant_f32(fx)
    f32_bytes = to_bytes(result_f32)
    bf16_bytes = to_bytes(result_f32.to(torch.bfloat16))

    # The stored BF16 golden must reproduce from the same raw bytes. If it does
    # not, the fixture's inputs and its golden have drifted and the F32 golden
    # would be anchored to something other than what the BF16 suite validates.
    if bf16_bytes != fx["expected_bf16"]:
        n = sum(1 for a, b in zip(bf16_bytes, fx["expected_bf16"]) if a != b)
        print(f"  {path.name}: REFUSING -- re-derived BF16 differs in {n} bytes")
        return False

    # How much the F32 golden actually adds: the fraction of values carrying
    # mantissa bits BF16 cannot hold. If this were ~0 the new comparison would
    # have no teeth, so it is reported rather than assumed.
    widened = result_f32.to(torch.bfloat16).to(torch.float32)
    informative = int((widened != result_f32).sum())
    total = result_f32.numel()
    pct = 100.0 * informative / total if total else 0.0

    status = "verified" if already_v2 else "upgraded v1 -> v2"
    print(
        f"  {path.name}: {status}, {total} values, "
        f"{informative} ({pct:.1f} %) not BF16-representable"
    )

    if check_only:
        if already_v2 and fx.get("expected_f32") != f32_bytes:
            print(f"  {path.name}: STALE -- stored F32 golden differs")
            return False
        return True

    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += struct.pack(
        "<8I",
        fx["scheme"],
        fx["scale_dtype"],
        fx["rows"],
        fx["cols"],
        len(fx["weight"]),
        len(fx["scale"]),
        len(bf16_bytes),
        len(f32_bytes),
    )
    out += fx["weight"]
    out += fx["scale"]
    out += bf16_bytes
    out += f32_bytes
    path.write_bytes(bytes(out))
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    fixtures = sorted(HERE.glob("*.bin"))
    if not fixtures:
        print("no .bin fixtures found", file=sys.stderr)
        return 1

    print(f"FP8 re-golden ({len(fixtures)} fixtures), torch {torch.__version__}")
    ok = True
    for path in fixtures:
        ok &= process(path, args.check)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
