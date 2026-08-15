#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate AWQ dequantization reference fixtures from real models.

External-reference discipline (the v0.6.4 rule)
-----------------------------------------------
The "expected" output MUST come from the canonical library's own code — never
from a hand-rolled reimplementation of the formula. A previous version of this
script reimplemented the unpack in plain PyTorch with a **sequential**
LSB-first nibble order — but AutoAWQ's GEMM format packs with the interleaved
order ``AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]`` (its dequant applies
``AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]`` after shift-unpacking, to
BOTH qweight and qzeros). The hand-rolled fixture shared the missing reorder
with the Rust decoder, so a column-permuting bug validated green at 0 ULP.
See docs/dogfooding-feedbacks/
bnb-nibble-order-and-circular-fixture-validation.md (Finding 2 / blast
radius).

This version takes the convention-bearing steps — ``unpack_awq`` +
``reverse_awq_order`` + the overflow mask — verbatim from AutoAWQ
(``awq.utils.packing_utils``, pure-torch, CPU). Only the final
``(iweight - izeros) * scales`` is computed in f32 instead of AutoAWQ's f16:
anamnesis dequantizes straight to BF16 with a single f32 rounding, while
AutoAWQ's f16 output would double-round (f32 → f16 → BF16) and shift a small
fraction of elements by 1 ULP. The numerics choice is guarded below by an
allclose check against AutoAWQ's full ``dequantize_gemm`` f16 output, so a
convention regression cannot hide behind the rounding difference.

AWQ packs along out_features (columns), unlike GPTQ which packs along
in_features (rows). The dequantization formula is:
  dequant = (qw - qz) * scale   (NO +1 offset on qzeros)

Note: AutoAWQ's GEMM format is 4-bit only (``raise NotImplementedError``
for any other width), so every fixture here is 4-bit and the Rust decoder
rejects other widths.

Fixture binary format, container v2 (all little-endian):
  4 bytes: magic "AMNA"
  4 bytes: container version (u32, currently 2)
  4 bytes: bits (u32) — 4
  4 bytes: group_size (u32)
  4 bytes: in_features (u32)
  4 bytes: out_features (u32)
  4 bytes: scale_dtype (u32) — 0=F32, 1=BF16, 2=F16
  4 bytes: qweight_len (u32)
  4 bytes: scales_len (u32)
  4 bytes: qzeros_len (u32)
  4 bytes: bf16_len (u32) — bytes of expected BF16 output
  4 bytes: f32_len (u32) — bytes of expected F32 output
  [qweight_len bytes]: raw packed qweight data
  [scales_len bytes]: raw scale data
  [qzeros_len bytes]: raw packed qzeros data
  [bf16_len bytes]: expected BF16 output
  [f32_len bytes]: expected F32 output

The F32 golden (v0.7.4) is what lets the kernel be compared at full width.
Every comparison before it rounded the reference to BF16 first, discarding 16
mantissa bits — and the great majority of the values in these fixtures carry
bits BF16 cannot hold, so that was hiding most of the available signal. The
exact percentage is printed per fixture.

**The F32 golden is taken from `result_f32` before the BF16 narrowing**, never
by widening the BF16 back, which would make the comparison circular with our
own rounding.

v1 (pre-v0.7.4) had no magic and no version, and stopped after the BF16 golden.
The magic is what lets the Rust reader reject a stale fixture with a clear
message instead of reading the header at the wrong offsets.

**Drift guard.** Regenerating rewrites the BF16 golden that every existing
cross-validation asserts against, so an AutoAWQ/torch upgrade could silently
re-anchor the suite. This script compares the regenerated BF16 against what is
already committed and REFUSES to write on any difference. As of 2026-08-15,
AutoAWQ 0.2.9 + torch 2.10.0 reproduce both committed goldens byte for byte.

Usage:
  python generate_awq.py
"""

import json
import struct
import time
from pathlib import Path

import torch
from safetensors import safe_open

from awq.utils.packing_utils import dequantize_gemm, reverse_awq_order, unpack_awq


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# v2 container (v0.7.4): magic + version prefix, and an F32 golden appended
# after the BF16 one. v1 had neither, so the magic is what lets the Rust reader
# reject a stale fixture with a clear message instead of reading the header at
# the wrong offsets. Both goldens come from AutoAWQ's own unpack/reorder; the
# F32 one is taken before the BF16 narrowing, never by widening the BF16 back.
FIXTURE_MAGIC = b"AMNA"
FIXTURE_VERSION = 2

SCALE_DTYPE_MAP = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
SCALE_DTYPE_NAMES = {0: "F32", 1: "BF16", 2: "F16"}


def find_snapshot(model_dir: Path) -> Path:
    snapshots = model_dir / "snapshots"
    dirs = list(snapshots.iterdir())
    return dirs[0]


def load_quantize_config(snapshot: Path) -> dict:
    for cfg_name in ["quantize_config.json", "config.json"]:
        cfg_path = snapshot / cfg_name
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
                if cfg_name == "config.json":
                    cfg = cfg.get("quantization_config", {})
                if cfg:
                    return cfg
    return {}


MODELS = [
    {
        "name": "llama_3_2_1b_awq",
        "model_dir": HF_CACHE / "models--casperhansen--llama-3.2-1b-instruct-awq",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "falcon3_1b_awq",
        "model_dir": HF_CACHE / "models--tiiuae--Falcon3-1B-Instruct-AWQ",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
]


def dequant_awq_canonical_f32(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    """AutoAWQ's own unpack + reorder, with the final multiply in f32.

    Mirrors ``awq.utils.packing_utils.dequantize_gemm`` step for step —
    ``unpack_awq`` → ``reverse_awq_order`` → overflow mask — and replaces only
    the final ``(iweight - izeros) * scales`` (f16 in AutoAWQ) with f32 math
    so the single rounding to BF16 matches anamnesis' output path exactly.
    """
    iweight, izeros = unpack_awq(qweight, qzeros, bits)
    iweight, izeros = reverse_awq_order(iweight, izeros, bits)

    iweight = torch.bitwise_and(iweight, (2**bits) - 1)
    izeros = torch.bitwise_and(izeros, (2**bits) - 1)

    scales_f32 = scales.float().repeat_interleave(group_size, dim=0)
    izeros_f32 = izeros.float().repeat_interleave(group_size, dim=0)
    return (iweight.float() - izeros_f32) * scales_f32


def generate_fixture(model_info: dict) -> None:
    name = model_info["name"]
    model_dir = model_info["model_dir"]
    layer_base = model_info["layer_base"]

    if not model_dir.exists():
        print(f"\nSKIP {name}: model not found at {model_dir}")
        return

    snapshot = find_snapshot(model_dir)
    print(f"\n{name}:")

    config = load_quantize_config(snapshot)
    bits = config.get("bits", config.get("w_bit", 4))
    group_size = config.get("group_size", config.get("q_group_size", 128))
    print(f"  Config: bits={bits}, group_size={group_size}")
    if bits != 4:
        print(f"  SKIP: AutoAWQ GEMM is 4-bit only (got bits={bits})")
        return

    st_files = list(snapshot.glob("*.safetensors"))
    if not st_files:
        print("  ERROR: no safetensors files found")
        return

    with safe_open(str(st_files[0]), framework="pt") as f:
        qweight = f.get_tensor(f"{layer_base}.qweight")
        scales = f.get_tensor(f"{layer_base}.scales")
        qzeros = f.get_tensor(f"{layer_base}.qzeros")

    pack_factor = 32 // bits
    in_features_full = qweight.shape[0]
    out_features_full = qweight.shape[1] * pack_factor

    print(f"  qweight: {list(qweight.shape)}, dtype={qweight.dtype}")
    print(f"  scales:  {list(scales.shape)}, dtype={scales.dtype}")
    print(f"  qzeros:  {list(qzeros.shape)}, dtype={qzeros.dtype}")
    print(f"  in_features={in_features_full}, out_features={out_features_full}")

    # Extract a 256×256 slice
    slice_in = min(256, in_features_full)
    slice_in = (slice_in // group_size) * group_size
    if slice_in == 0:
        slice_in = group_size
    slice_out = min(256, out_features_full)
    slice_out = (slice_out // pack_factor) * pack_factor
    if slice_out == 0:
        slice_out = pack_factor

    packed_slice_cols = slice_out // pack_factor
    num_groups = slice_in // group_size

    print(f"  Slice: {slice_in}×{slice_out} (packed cols: {packed_slice_cols})")

    qw_slice = qweight[:slice_in, :packed_slice_cols].contiguous()
    sc_slice = scales[:num_groups, :slice_out].contiguous()
    qz_slice = qzeros[:num_groups, :packed_slice_cols].contiguous()

    qw_bytes = qw_slice.numpy().tobytes()
    sc_bytes = sc_slice.view(torch.uint8).reshape(-1).cpu().numpy().tobytes() if scales.dtype != torch.float32 else sc_slice.numpy().tobytes()
    qz_bytes = qz_slice.numpy().tobytes()

    scale_dtype_id = SCALE_DTYPE_MAP.get(scales.dtype)
    if scale_dtype_id is None:
        print(f"  ERROR: unexpected scale dtype {scales.dtype}")
        return

    print(f"  Scale dtype: {SCALE_DTYPE_NAMES[scale_dtype_id]}")

    # Dequantize with AutoAWQ's own unpack/reorder (best of 5)
    best_us = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        result_f32 = dequant_awq_canonical_f32(qw_slice, qz_slice, sc_slice, bits, group_size)
        t1 = time.perf_counter()
        best_us = min(best_us, (t1 - t0) * 1e6)
    result_bf16 = result_f32.to(torch.bfloat16)

    # Guard: the f32-multiply variant must agree with AutoAWQ's full
    # canonical dequantize_gemm (f16 math) to within f16 rounding. A
    # convention error (wrong reorder, wrong zero handling) would blow
    # far past this bound.
    canonical_f16 = dequantize_gemm(qw_slice, qz_slice, sc_slice, bits, group_size)
    max_diff = (result_f32 - canonical_f16.float()).abs().max().item()
    scale_mag = canonical_f16.float().abs().max().item()
    assert max_diff <= 2e-3 * max(scale_mag, 1.0), (
        f"f32-multiply variant diverges from canonical dequantize_gemm: "
        f"max|diff|={max_diff:.6e} (magnitude {scale_mag:.3f}) — convention regression?"
    )
    print(
        f"  AutoAWQ unpack+reverse_awq_order: {best_us:.1f} µs (best of 5); "
        f"guard vs dequantize_gemm f16: max|diff|={max_diff:.2e}"
    )

    expected_bytes = result_bf16.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    # The F32 golden, added in v0.7.4. Taken from `result_f32` **before** the
    # narrowing above, not by widening `result_bf16` back: the whole point is a
    # golden that no rounding of ours has touched.
    expected_f32_bytes = (
        result_f32.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    )

    # How much the F32 golden adds over the BF16 one. If this were ~0 the new
    # comparison would have no teeth, so it is measured rather than assumed.
    widened = result_bf16.float()
    informative = int((widened != result_f32).sum())
    total = result_f32.numel()
    print(
        f"  F32 golden: {informative}/{total} "
        f"({100.0 * informative / total:.1f} %) values are not BF16-representable"
    )

    output_path = Path(__file__).parent / f"{name}.bin"

    # Drift guard. Regenerating rewrites the BF16 golden that every existing
    # cross-validation asserts against, so an AutoAWQ/torch upgrade since the
    # fixture was made would silently re-anchor the suite. Compare against what
    # is committed and refuse rather than overwrite; a real convention change
    # should be an explicit decision, not a side effect of adding F32.
    if output_path.exists():
        old = output_path.read_bytes()
        old_is_v2 = old[:4] == FIXTURE_MAGIC
        if old_is_v2:
            (o_bf16_len,) = struct.unpack_from("<I", old, 8 + 8 * 4)
            o_start = 8 + 10 * 4 + len(qw_bytes) + len(sc_bytes) + len(qz_bytes)
        else:
            (o_bf16_len,) = struct.unpack_from("<I", old, 8 * 4)
            o_start = 9 * 4 + len(qw_bytes) + len(sc_bytes) + len(qz_bytes)
        old_bf16 = old[o_start : o_start + o_bf16_len]
        if old_bf16 != expected_bytes:
            n = sum(1 for a, b in zip(old_bf16, expected_bytes) if a != b)
            raise SystemExit(
                f"REFUSING to overwrite {output_path.name}: the regenerated BF16 "
                f"golden differs from the committed one in {n} bytes. The library "
                f"stack has drifted since this fixture was made. Investigate before "
                f"re-goldening — that is a finding, not a formality."
            )
        print("  Drift guard: regenerated BF16 golden matches the committed one")

    with open(output_path, "wb") as out:
        out.write(FIXTURE_MAGIC)
        out.write(struct.pack("<I", FIXTURE_VERSION))
        out.write(struct.pack("<I", bits))
        out.write(struct.pack("<I", group_size))
        out.write(struct.pack("<I", slice_in))
        out.write(struct.pack("<I", slice_out))
        out.write(struct.pack("<I", scale_dtype_id))
        out.write(struct.pack("<I", len(qw_bytes)))
        out.write(struct.pack("<I", len(sc_bytes)))
        out.write(struct.pack("<I", len(qz_bytes)))
        out.write(struct.pack("<I", len(expected_bytes)))
        out.write(struct.pack("<I", len(expected_f32_bytes)))
        out.write(qw_bytes)
        out.write(sc_bytes)
        out.write(qz_bytes)
        out.write(expected_bytes)
        out.write(expected_f32_bytes)

    print(f"  Written: {output_path.name} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    print("Generating AWQ cross-validation fixtures (AutoAWQ packing_utils)...")
    for model in MODELS:
        generate_fixture(model)
    print("\nDone.")
