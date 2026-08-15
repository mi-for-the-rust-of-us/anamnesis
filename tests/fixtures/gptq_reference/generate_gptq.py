#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate GPTQ dequantization reference fixtures from real models.

External-reference discipline (the v0.6.4 rule)
-----------------------------------------------
The "expected" output MUST come from the canonical library's own code — never
from a hand-rolled reimplementation of the formula. A previous version of this
script reimplemented the dequant in plain PyTorch ("the standard GPTQ
formula") without ever importing the real library, so the fixture was circular
with the Rust decoder. See docs/dogfooding-feedbacks/
bnb-nibble-order-and-circular-fixture-validation.md (Finding 2 / blast
radius). Re-anchoring CONFIRMED the GPTQ convention (sequential LSB-first
unpack, +1 zero-point offset on v1 checkpoints) — unlike BnB and AWQ, where
re-anchoring exposed real bugs — but the fixture must still come from the
canonical code so future regressions cannot hide.

This version drives **GPTQModel** (the maintained AutoGPTQ successor):

1. A ``TorchLinear`` module is built over the exact on-disk tensor slices.
2. ``convert_gptq_v1_to_v2_format_module`` — GPTQModel's own loader step —
   converts the v1 checkpoint zeros (stored as ``z - 1``) to runtime v2 form.
   This is where the canonical ``+1`` lives: GPTQModel's dequant itself does
   NOT add 1; its loader does. anamnesis reads on-disk v1 bytes directly, so
   its ``(qz + 1)`` at dequant time must equal loader-conversion + plain
   dequant.
3. ``dequantize_weight()`` — the canonical unpack + dequant.

The module's ``scales`` buffer is cast to f32 before the dequant call so the
final multiply happens in f32 with a single rounding to BF16 (anamnesis'
output path). GPTQModel's stock f16 math would double-round (f32 → f16 →
BF16); the guard below asserts the f32 variant agrees with the stock f16
output to within f16 rounding, so a convention regression cannot hide behind
the rounding difference.

Note: ``import pcre`` shim — GPTQModel's logger imports the ``pcre`` module
(PyPI ``pypcre``), which has no Python 3.14 Windows wheel. The shim aliases
the stdlib ``re`` (PCRE-compatible for the logger's ANSI-escape regex); the
dequant code path never touches it.

Fixture binary format, container v2 (all little-endian):
  4 bytes: magic "AMNQ"
  4 bytes: container version (u32, currently 2)
  4 bytes: bits (u32) — 4 or 8
  4 bytes: group_size (u32)
  4 bytes: in_features (u32) — unpacked input dimension
  4 bytes: out_features (u32)
  4 bytes: scale_dtype (u32) — 0=F32, 1=BF16, 2=F16
  4 bytes: has_g_idx (u32) — 0 or 1
  4 bytes: qweight_len (u32) — bytes of packed qweight data
  4 bytes: scales_len (u32) — bytes of scale data
  4 bytes: qzeros_len (u32) — bytes of packed qzeros data (v1 on-disk form)
  4 bytes: g_idx_len (u32) — bytes of g_idx data (0 if absent)
  4 bytes: bf16_len (u32) — bytes of expected BF16 output
  4 bytes: f32_len (u32) — bytes of expected F32 output
  [qweight_len bytes]: raw packed qweight data
  [scales_len bytes]: raw scale data
  [qzeros_len bytes]: raw packed qzeros data
  [g_idx_len bytes]: raw g_idx data (empty if has_g_idx=0)
  [bf16_len bytes]: expected BF16 output
  [f32_len bytes]: expected F32 output

The F32 golden (v0.7.4) is what lets the kernel be compared at full width.
Every comparison before it rounded the reference to BF16 first, discarding 16
mantissa bits — and 77-96 % of the values in these fixtures carry bits BF16
cannot hold, so that was hiding most of the available signal.

**The F32 golden is taken from `result_f32` before the BF16 narrowing**, never
by widening the BF16 back, which would make the comparison circular with our
own rounding.

v1 (pre-v0.7.4) had no magic and no version, and stopped after the BF16 golden.
The magic is what lets the Rust reader reject a stale fixture with a clear
message instead of reading the header at the wrong offsets.

**Drift guard.** Regenerating rewrites the BF16 golden that every existing
cross-validation asserts against, so a GPTQModel/torch upgrade could silently
re-anchor the suite. This script compares the regenerated BF16 against what is
already committed and REFUSES to write on any difference. As of 2026-08-15,
GPTQModel 7.1.0 + torch 2.10.0 reproduce all four committed goldens byte for
byte.

Usage:
  python generate_gptq.py
"""

import json
import re as _re
import struct
import sys
import time
import types
import warnings
from pathlib import Path

# ---- pcre shim (see module docstring) — must precede the gptqmodel import ----
_shim = types.ModuleType("pcre")
for _k in dir(_re):
    if not _k.startswith("_"):
        setattr(_shim, _k, getattr(_re, _k))
_shim.Flag = types.SimpleNamespace(
    CASELESS=_re.IGNORECASE, IGNORECASE=_re.IGNORECASE,
    MULTILINE=_re.MULTILINE, DOTALL=_re.DOTALL,
    EXTENDED=_re.VERBOSE, VERBOSE=_re.VERBOSE,
    UNICODE=_re.UNICODE, UTF=_re.UNICODE, ANCHORED=0, NO_AUTO_CAPTURE=0,
)
sys.modules["pcre"] = _shim

warnings.filterwarnings("ignore")

import torch
from safetensors import safe_open

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# v2 container (v0.7.4): magic + version prefix, and an F32 golden appended
# after the BF16 one. v1 had neither, so the magic is what lets the Rust reader
# reject a stale fixture with a clear message instead of reading the header at
# the wrong offsets. Both goldens come from GPTQModel; the F32 one is taken
# before the BF16 narrowing, never by widening the BF16 back.
FIXTURE_MAGIC = b"AMNQ"
FIXTURE_VERSION = 2

SCALE_DTYPE_MAP = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
SCALE_DTYPE_NAMES = {0: "F32", 1: "BF16", 2: "F16"}


def find_snapshot(model_dir: Path) -> Path:
    """Find the snapshot directory for a cached model."""
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"No snapshots directory in {model_dir}")
    dirs = list(snapshots.iterdir())
    if not dirs:
        raise FileNotFoundError(f"No snapshot found in {snapshots}")
    return dirs[0]


def load_quantize_config(snapshot: Path) -> dict:
    """Load GPTQ quantization config from various sources."""
    qc_path = snapshot / "quantize_config.json"
    if qc_path.exists():
        with open(qc_path) as f:
            return json.load(f)

    cfg_path = snapshot / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
            if "quantization_config" in cfg:
                return cfg["quantization_config"]

    raise FileNotFoundError(f"No quantize config found in {snapshot}")


# Model definitions
MODELS = [
    # --- 4-bit models ---
    {
        "name": "falcon3_1b_int4",
        "model_dir": HF_CACHE / "models--tiiuae--Falcon3-1B-Instruct-GPTQ-Int4",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "llama_3_2_1b_int4",
        "model_dir": HF_CACHE / "models--shuyuej--Llama-3.2-1B-Instruct-GPTQ",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    # --- 8-bit models ---
    {
        "name": "falcon3_1b_int8",
        "model_dir": HF_CACHE / "models--tiiuae--Falcon3-1B-Instruct-GPTQ-Int8",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
    {
        "name": "llama_3_2_1b_gptqmodel_int8",
        "model_dir": HF_CACHE / "models--iproskurina--Llama-3.2-1B-gptqmodel-8bit",
        "layer_base": "model.layers.0.mlp.gate_proj",
    },
]


def dequant_gptq_canonical(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: torch.Tensor | None,
    bits: int,
    group_size: int,
    sym: bool,
    desc_act: bool,
    in_features: int,
    out_features: int,
    checkpoint_format: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize via GPTQModel's own module pipeline.

    Returns ``(f32_result, f16_result)``: the f32 variant (scales cast to
    f32 so the final multiply single-rounds to BF16 like anamnesis) and the
    stock-precision variant (scales dtype as stored) used as the guard.
    """

    def build_module(scales_tensor: torch.Tensor) -> TorchLinear:
        lin = TorchLinear(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        )
        lin.qweight.copy_(qweight)
        lin.qzeros.copy_(qzeros)
        lin.scales = scales_tensor.clone()
        if g_idx is not None:
            lin.g_idx.copy_(g_idx.to(lin.g_idx.dtype))
        else:
            lin.g_idx.copy_(
                (torch.arange(in_features, dtype=torch.int32) // group_size)
            )
        if checkpoint_format == "gptq":
            # GPTQModel's own loader-side v1 → v2 conversion (adds the +1
            # to the packed zeros). Its dequantize_weight assumes v2.
            convert_gptq_v1_to_v2_format_module(
                lin, bits=bits, pack_dtype=torch.int32
            )
        return lin

    lin_f32 = build_module(scales.float())
    result_f32 = lin_f32.dequantize_weight()
    lin_stock = build_module(scales)
    result_stock = lin_stock.dequantize_weight()
    return result_f32, result_stock


def generate_fixture(model_info: dict) -> None:
    name = model_info["name"]
    model_dir = model_info["model_dir"]
    layer_base = model_info["layer_base"]

    if not model_dir.exists():
        print(f"\nSKIP {name}: model not found at {model_dir}")
        return

    snapshot = find_snapshot(model_dir)
    print(f"\n{name}:")
    print(f"  Snapshot: {snapshot}")

    # Load config
    config = load_quantize_config(snapshot)
    bits = config.get("bits", 4)
    group_size = config.get("group_size", 128)
    sym = bool(config.get("sym", True))
    desc_act = bool(config.get("desc_act", False))
    checkpoint_format = config.get("checkpoint_format") or "gptq"
    print(
        f"  Config: bits={bits}, group_size={group_size}, sym={sym}, "
        f"desc_act={desc_act}, checkpoint_format={checkpoint_format}"
    )

    # Find safetensors file
    st_files = list(snapshot.glob("*.safetensors"))
    if not st_files:
        print("  ERROR: no safetensors files found")
        return

    # Load tensors
    qweight_name = f"{layer_base}.qweight"
    scales_name = f"{layer_base}.scales"
    qzeros_name = f"{layer_base}.qzeros"
    g_idx_name = f"{layer_base}.g_idx"

    with safe_open(str(st_files[0]), framework="pt") as f:
        qweight = f.get_tensor(qweight_name)
        scales = f.get_tensor(scales_name)
        qzeros = f.get_tensor(qzeros_name)
        g_idx = f.get_tensor(g_idx_name) if g_idx_name in f.keys() else None

    pack_factor = 32 // bits
    packed_rows = qweight.shape[0]
    out_features_full = qweight.shape[1]
    in_features_full = packed_rows * pack_factor

    print(f"  qweight: {list(qweight.shape)}, dtype={qweight.dtype}")
    print(f"  scales:  {list(scales.shape)}, dtype={scales.dtype}")
    print(f"  qzeros:  {list(qzeros.shape)}, dtype={qzeros.dtype}")
    print(f"  g_idx:   {'present' if g_idx is not None else 'absent'}")
    print(f"  in_features={in_features_full}, out_features={out_features_full}")

    # Extract a 256×256 slice (or smaller if model is smaller)
    slice_in = min(256, in_features_full)
    slice_out = min(256, out_features_full)
    # Ensure slice_in is a multiple of group_size and pack_factor
    slice_in = (slice_in // group_size) * group_size
    if slice_in == 0:
        slice_in = group_size
    # Ensure slice_out is a multiple of pack_factor
    slice_out = (slice_out // pack_factor) * pack_factor
    if slice_out == 0:
        slice_out = pack_factor

    packed_slice_rows = slice_in // pack_factor
    packed_slice_cols = slice_out // pack_factor
    num_groups = slice_in // group_size

    print(f"  Slice: {slice_in}×{slice_out} (packed: {packed_slice_rows}×{slice_out})")

    # Extract slices
    qw_slice = qweight[:packed_slice_rows, :slice_out].contiguous()
    sc_slice = scales[:num_groups, :slice_out].contiguous()
    qz_slice = qzeros[:num_groups, :packed_slice_cols].contiguous()
    gi_slice = g_idx[:slice_in].contiguous() if g_idx is not None else None

    # For g_idx, the slice must reference only groups within [0, num_groups).
    # Sequential (desc_act=False) checkpoints satisfy this naturally; an
    # act-order slice would not, so fall back to sequential assignment.
    if gi_slice is not None:
        gi_max = gi_slice.max().item()
        if gi_max >= num_groups:
            print(f"  Note: remapping g_idx (max={gi_max} >= num_groups={num_groups})")
            gi_slice = torch.arange(slice_in, dtype=torch.int32) // group_size

    # Get raw bytes
    qw_bytes = qw_slice.numpy().tobytes()
    sc_bytes = sc_slice.view(torch.uint8).reshape(-1).cpu().numpy().tobytes() if scales.dtype != torch.float32 else sc_slice.numpy().tobytes()
    qz_bytes = qz_slice.numpy().tobytes()
    gi_bytes = gi_slice.numpy().tobytes() if gi_slice is not None else b""

    scale_dtype_id = SCALE_DTYPE_MAP.get(scales.dtype)
    if scale_dtype_id is None:
        print(f"  ERROR: unexpected scale dtype {scales.dtype}")
        return

    print(f"  Scale dtype: {SCALE_DTYPE_NAMES[scale_dtype_id]}")

    # Dequantize with GPTQModel's own module pipeline (best of 5)
    best_us = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        result_f32, result_stock = dequant_gptq_canonical(
            qw_slice, sc_slice, qz_slice, gi_slice,
            bits, group_size, sym, desc_act,
            slice_in, slice_out, checkpoint_format,
        )
        t1 = time.perf_counter()
        best_us = min(best_us, (t1 - t0) * 1e6)

    # Guard: the f32-scales variant must agree with GPTQModel's stock-dtype
    # output to within stock rounding. A convention error (unpack order,
    # missing v1→v2 zero conversion) would blow far past this bound.
    max_diff = (result_f32 - result_stock.float()).abs().max().item()
    scale_mag = result_stock.float().abs().max().item()
    assert max_diff <= 2e-3 * max(scale_mag, 1.0), (
        f"f32-scales variant diverges from stock dequantize_weight: "
        f"max|diff|={max_diff:.6e} (magnitude {scale_mag:.3f}) — convention regression?"
    )

    result_bf16 = result_f32.to(torch.bfloat16)
    expected_bytes = result_bf16.view(torch.uint8).reshape(-1).numpy().tobytes()
    # The F32 golden, added in v0.7.4. Taken from `result_f32` **before** the
    # narrowing above, not by widening `result_bf16` back: the whole point is a
    # golden that no rounding of ours has touched. See the container note below.
    expected_f32_bytes = (
        result_f32.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    )
    print(
        f"  GPTQModel dequantize_weight: {best_us:.1f} µs (best of 5, "
        f"{slice_in}×{slice_out} = {slice_in*slice_out} elements); "
        f"guard vs stock dtype: max|diff|={max_diff:.2e}"
    )
    print(f"  Output shape: {list(result_bf16.shape)}")

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
    has_g_idx = 1 if gi_slice is not None else 0

    # Drift guard. Regenerating rewrites the BF16 golden that every existing
    # cross-validation asserts against, so a GPTQModel/torch upgrade since the
    # fixture was made would silently re-anchor the suite. Compare against what
    # is committed and refuse rather than overwrite; a real convention change
    # should be an explicit decision, not a side effect of adding F32.
    if output_path.exists():
        old = output_path.read_bytes()
        old_is_v2 = old[:4] == FIXTURE_MAGIC
        if old_is_v2:
            (o_bf16_len,) = struct.unpack_from("<I", old, 8 + 10 * 4)
            o_start = 8 + 12 * 4 + len(qw_bytes) + len(sc_bytes) + len(qz_bytes) + len(gi_bytes)
        else:
            (o_bf16_len,) = struct.unpack_from("<I", old, 10 * 4)
            o_start = 11 * 4 + len(qw_bytes) + len(sc_bytes) + len(qz_bytes) + len(gi_bytes)
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
        out.write(struct.pack("<I", has_g_idx))
        out.write(struct.pack("<I", len(qw_bytes)))
        out.write(struct.pack("<I", len(sc_bytes)))
        out.write(struct.pack("<I", len(qz_bytes)))
        out.write(struct.pack("<I", len(gi_bytes)))
        out.write(struct.pack("<I", len(expected_bytes)))
        out.write(struct.pack("<I", len(expected_f32_bytes)))
        out.write(qw_bytes)
        out.write(sc_bytes)
        out.write(qz_bytes)
        out.write(gi_bytes)
        out.write(expected_bytes)
        out.write(expected_f32_bytes)

    print(f"  Written: {output_path.name} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    print("Generating GPTQ cross-validation fixtures (GPTQModel TorchLinear)...")
    for model in MODELS:
        generate_fixture(model)
    print("\nDone.")
