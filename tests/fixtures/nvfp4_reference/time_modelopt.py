#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The modelopt half of the NVFP4 speed figure (see bench_nvfp4_kernel in
tests/cross_validation_nvfp4.rs): NVFP4QTensor.dequantize to bfloat16 on the
fixture's 65 536 values, 20 warm-up calls, then 200 timed calls, min and median.
Single-threaded torch, matching anamnesis's single call. Same environment as
generate_nvfp4.py."""

import json
import statistics
import struct
import time
from pathlib import Path

import torch
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

torch.set_num_threads(1)
data = (Path(__file__).parent / "llama31_8b_nvfp4_q_proj.safetensors").read_bytes()
n = struct.unpack("<Q", data[:8])[0]
header = json.loads(data[8 : 8 + n])
base = 8 + n
layer = "model.layers.0.self_attn.q_proj"


def raw(name):
    a, b = header[f"{layer}.{name}"]["data_offsets"]
    return bytearray(data[base + a : base + b])


rows, packed = header[f"{layer}.weight"]["shape"]
qt = NVFP4QTensor(torch.Size([rows, packed * 2]), torch.bfloat16, torch.frombuffer(raw("weight"), dtype=torch.uint8).reshape(rows, packed))
kwargs = {
    "scale": torch.frombuffer(raw("weight_scale"), dtype=torch.uint8).reshape(rows, -1).view(torch.float8_e4m3fn),
    "double_scale": torch.frombuffer(raw("weight_scale_2"), dtype=torch.float32).reshape(()),
    "block_sizes": {-1: 16},
}
samples = []
for i in range(220):
    t0 = time.perf_counter()
    qt.dequantize(dtype=torch.bfloat16, **kwargs)
    if i >= 20:
        samples.append(time.perf_counter() - t0)
print(f"modelopt NVFP4 -> BF16, {rows * packed * 2} values: min {min(samples) * 1e6:.1f} us, median {statistics.median(samples) * 1e6:.1f} us (20 warm-up, 200 timed, 1 thread)")
