#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""The gguf-py half of the GGUF speed table in docs/validation.md.

Times gguf-py's ``dequantize`` on every GGUF cross-validation fixture with the
protocol ``tests/bench_gguf_table_adhoc.rs`` applies to anamnesis: 20 warm-up
calls, then 200 timed calls in one process, min and median reported. Run five
passes of each, alternating; the table publishes each side's best-of-5 median
(see the Rust file for why). Run the
two one after the other, never together, on an idle machine. Types gguf-py does
not implement (Q1_0, Q2_0) are reported as n/a. Only the 65 536-element
fixtures are timed, the size every table row is measured at; the exhaustive
F8_E4M3 fixtures are cross-validation inputs, not table rows.

Usage (gguf-py pinned to the commit the fixtures were generated with; F8_E4M3
needs 1b6935b, the first with type 43, and the Phase 7.10 rows were timed at
37b53fd):
  pip install "git+https://github.com/ggml-org/llama.cpp@1b6935b227b9f9aa89deea8016c0e208f547446f#subdirectory=gguf-py" numpy
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
TABLE_ELEMENTS = 65_536

print(f"fixture\tkernel\tmin_us\tmedian_us\t({WARMUP} warm-up, {ITERATIONS} timed calls)")
for path in sorted(Path(__file__).parent.glob("*.bin")):
    data = path.read_bytes()
    if data[:4] != b"AMNG":
        continue
    disc, n_elements, raw_len = struct.unpack("<III", data[8:20])
    if n_elements != TABLE_ELEMENTS:
        continue
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
