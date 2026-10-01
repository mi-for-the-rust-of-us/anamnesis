#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The gguf-py half of the GGUF speed table in docs/validation.md.

Times gguf-py's ``dequantize`` on every GGUF cross-validation fixture with the
protocol ``tests/bench_gguf_table_adhoc.rs`` applies to anamnesis: 20 warm-up
calls, then 200 timed calls in one process, min and median reported. Run the
two one after the other, never together, on an idle machine. Types gguf-py does
not implement (Q1_0, Q2_0) are reported as n/a.

Usage (gguf-py pinned to the commit the fixtures were generated with):
  pip install "git+https://github.com/ggml-org/llama.cpp@37b53fd4545847188fdad29e38ba57875efc8228#subdirectory=gguf-py" numpy
  python tests/fixtures/gguf_reference/time_gguf_py.py
"""

import statistics
import struct
import time
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, dequantize

WARMUP = 20
ITERATIONS = 200

print(f"fixture\tkernel\tmin_us\tmedian_us\t({WARMUP} warm-up, {ITERATIONS} timed calls)")
for path in sorted(Path(__file__).parent.glob("*.bin")):
    data = path.read_bytes()
    if data[:4] != b"AMNG":
        continue
    disc, _n_elements, raw_len = struct.unpack("<III", data[8:20])
    qtype = GGMLQuantizationType(disc)
    raw = np.frombuffer(data[28 : 28 + raw_len], dtype=np.uint8)
    try:
        for _ in range(WARMUP):
            dequantize(raw, qtype)
    except NotImplementedError:
        print(f"{path.stem}\t{qtype.name}\tn/a\tn/a")
        continue
    samples = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        dequantize(raw, qtype)
        samples.append(time.perf_counter() - t0)
    print(f"{path.stem}\t{qtype.name}\t{min(samples) * 1e6:.1f}\t{statistics.median(samples) * 1e6:.1f}")
