#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate the ModelOpt NVFP4 cross-validation fixture from a real model.

NVIDIA TensorRT Model Optimizer (`modelopt`) exports NVFP4 checkpoints as, per
quantised linear layer X:
  X.weight          U8       [rows, cols/2]   two E2M1 codes per byte
  X.weight_scale    F8_E4M3  [rows, cols/16]  one scale per 16 values
  X.weight_scale_2  F32      []               one scale per tensor
  X.input_scale     F32      []               activation scale (not used here)

This script takes the first 16 rows (65 536 values) of one such layer from
nvidia/Llama-3.1-8B-Instruct-NVFP4 at a pinned revision, fetching only the
header and those byte ranges (no 5.5 GB download), and dequantises them with
modelopt's own `NVFP4QTensor.dequantize`: the canonical implementation, per the
project's fixture rule. It writes:

  llama31_8b_nvfp4_q_proj.safetensors  the layer's four tensors, original names
                                       and dtypes, sliced to 16 rows, so the
                                       Rust test can run the public parse ->
                                       remember path on a real ModelOpt layout;
  llama31_8b_nvfp4_q_proj.golden       "AMNV", u32 version 1, u32 n_values,
                                       then the BF16 golden, then the F32 golden
                                       (all little-endian).

Both goldens come from modelopt (dequantize(dtype=torch.float32) and
dequantize(dtype=torch.bfloat16)); the script also checks that the BF16 one is
the F32 one rounded to nearest even, so the two cannot drift apart.

The checkpoint's producer is modelopt 0.37.0.dev5 (its config.json). This uses
0.47.0: `NVFP4QTensor.dequantize` is identical between the 0.37.0 and 0.47.0
tags apart from the import path of the Triton fast path (`fast=True`), which
this does not use.

Usage (WSL / Linux; torch on CPU is enough):
  python3 -m venv ~/amn-modelopt-venv && . ~/amn-modelopt-venv/bin/activate
  pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
  pip install "nvidia-modelopt==0.47.0" numpy requests huggingface_hub
  python tests/fixtures/nvfp4_reference/generate_nvfp4.py
"""

import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

REPO = "nvidia/Llama-3.1-8B-Instruct-NVFP4"
REVISION = "bdb54e24298451af785c0ac63c1b485e9b7400a2"
SHARD = "model-00001-of-00002.safetensors"
LAYER = "model.layers.0.self_attn.q_proj"
ROWS = 16
BLOCK = 16
OUT = Path(__file__).parent
NAME = "llama31_8b_nvfp4_q_proj"

URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{SHARD}"


def fetch(start: int, end_inclusive: int) -> bytes:
    req = urllib.request.Request(
        URL,
        headers={"User-Agent": "anamnesis-fixture", "Range": f"bytes={start}-{end_inclusive}"},
    )
    data = urllib.request.urlopen(req, timeout=120).read()
    assert len(data) == end_inclusive - start + 1, f"short range read: {len(data)}"
    return data


def main() -> None:
    header_len = struct.unpack("<Q", fetch(0, 7))[0]
    header = json.loads(fetch(8, 8 + header_len - 1))
    base = 8 + header_len

    def entry(name: str) -> dict:
        return header[name]

    w, s, s2, si = (entry(f"{LAYER}.{k}") for k in ("weight", "weight_scale", "weight_scale_2", "input_scale"))
    assert w["dtype"] == "U8" and s["dtype"] == "F8_E4M3", (w, s)
    assert s2["dtype"] == "F32" and si["dtype"] == "F32", (s2, si)
    rows, packed_cols = w["shape"]
    cols = packed_cols * 2
    assert s["shape"] == [rows, cols // BLOCK], (w, s)

    def rows_of(e: dict, row_bytes: int) -> bytes:
        start = base + e["data_offsets"][0]
        return fetch(start, start + ROWS * row_bytes - 1)

    def whole(e: dict) -> bytes:
        a, b = e["data_offsets"]
        return fetch(base + a, base + b - 1)

    weight = rows_of(w, packed_cols)
    scale = rows_of(s, cols // BLOCK)
    scale_2 = whole(s2)
    input_scale = whole(si)

    # modelopt's own dequantisation.
    packed = torch.frombuffer(bytearray(weight), dtype=torch.uint8).reshape(ROWS, packed_cols)
    fp8 = torch.frombuffer(bytearray(scale), dtype=torch.uint8).reshape(ROWS, cols // BLOCK).view(torch.float8_e4m3fn)
    double_scale = torch.frombuffer(bytearray(scale_2), dtype=torch.float32).reshape(())
    qt = NVFP4QTensor(torch.Size([ROWS, cols]), torch.bfloat16, packed)
    kwargs = {"scale": fp8, "double_scale": double_scale, "block_sizes": {-1: BLOCK}}
    f32 = qt.dequantize(dtype=torch.float32, **kwargs).contiguous()
    bf16 = qt.dequantize(dtype=torch.bfloat16, **kwargs).contiguous()

    # The BF16 golden must be the F32 golden rounded to nearest even.
    bits = f32.view(torch.int32).numpy().view(np.uint32)
    rne = ((bits + (np.uint32(0x7FFF) + ((bits >> 16) & 1))) >> 16).astype(np.uint16)
    assert np.array_equal(rne, bf16.view(torch.int16).numpy().view(np.uint16)), "BF16 != RNE(F32)"

    # The layer in its original layout, sliced to ROWS rows.
    tensors = [
        (f"{LAYER}.weight", "U8", [ROWS, packed_cols], weight),
        (f"{LAYER}.weight_scale", "F8_E4M3", [ROWS, cols // BLOCK], scale),
        (f"{LAYER}.weight_scale_2", "F32", s2["shape"], scale_2),
        (f"{LAYER}.input_scale", "F32", si["shape"], input_scale),
    ]
    meta, data = {}, b""
    for name, dtype, shape, payload in tensors:
        meta[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(payload)]}
        data += payload
    meta["__metadata__"] = {"source": f"{REPO}@{REVISION}", "layer": LAYER, "rows": str(ROWS)}
    j = json.dumps(meta, separators=(",", ":")).encode()
    j += b" " * ((8 - len(j) % 8) % 8)
    (OUT / f"{NAME}.safetensors").write_bytes(struct.pack("<Q", len(j)) + j + data)

    n = ROWS * cols
    golden = b"AMNV" + struct.pack("<II", 1, n) + bf16.view(torch.int16).numpy().tobytes() + f32.numpy().tobytes()
    (OUT / f"{NAME}.golden").write_bytes(golden)

    print(
        f"{NAME}: {LAYER} rows 0..{ROWS} of [{rows}, {cols}], {n} values, "
        f"weight_scale_2={float(double_scale):.9g}, input_scale={struct.unpack('<f', input_scale)[0]:.9g}, "
        f"|w| mean={f32.abs().mean().item():.4g}"
    )


if __name__ == "__main__":
    main()
