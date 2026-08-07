# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate the Python-side baseline for whole-model GGUF dequantisation.

This is the Phase 7.2 counterpart to
``tests/fixtures/convert_reference/generate_convert_timings.py``: it times
the *same unit of work* anamnesis's ``convert::read_hub`` performs — open a
quantised ``.gguf``, walk every tensor, dequantise the block-quantised ones
into a dense float array in memory — using the reference Python stack.

Baseline library
----------------
``gguf`` (gguf-py), the package maintained in the llama.cpp tree and
published to PyPI by the GGML project. It is the right comparison for three
reasons:

1. It is what a Python user reaches for **today** to read a GGUF outside an
   inference runtime — there is no other general-purpose GGUF dequantiser on
   PyPI.
2. The two are cross-validated against each other, so they provably agree on
   the *values* — though **not on the delivered type**, see below.
3. Its dequant kernels are NumPy-vectorised, not naive Python loops. This is
   deliberately the *strong* baseline: a hand-rolled pure-Python dequantiser
   would be orders of magnitude slower and the comparison would be
   meaningless.

Honest caveats, recorded in the sidecar so the Rust side can print them
----------------------------------------------------------------------
* **Output width differs — this is not a like-for-like comparison.** gguf-py's
  ``dequantize()`` always produces ``float32``; anamnesis produces ``BF16``
  (``TargetDtype`` has exactly one variant). gguf-py therefore writes **2x**
  the output bytes for the same tensor. On a memory-bandwidth-bound workload
  that is a real handicap for gguf-py which is *not* attributable to Python.

  What the cross-validation actually establishes is narrower than "identical
  output": it takes gguf-py's ``float32``, rounds it to ``BF16``
  round-to-nearest-even (``f32_array_to_bf16_bytes`` in the fixture
  generators), and asserts anamnesis matches at **0 ULP across all 22
  kernels**. So the two agree on the *numbers* and differ in the delivered
  *width*; both do their block arithmetic in ``f32`` internally, so this is an
  output-type difference, not a kernel-precision one. Treat the speedup as an
  upper bound and see ``docs/perf-experiments.md`` Experiment 12.
* **Single-threaded.** gguf-py is single-threaded, so the fair like-for-like
  comparison is against anamnesis at ``--threads 1``; the multi-threaded
  number is reported separately.
* The GGUF is memory-mapped by both sides, and the file is read once before
  timing starts so neither pays first-touch page-fault cost.

Usage::

    python generate_gguf_dequant_timings.py

Writes ``<model-stem>.dequant.timing.json`` next to itself, one per model
found in ``models/`` (that directory is gitignored — see ``generate_gguf.py``
for the download recipe). Not invoked by ``cargo test``; this is a
refresh-when-the-environment-changes utility.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
MODELS = HERE / "models"

# Best-of-N, matching the Rust harness's `SAMPLES` so the medians are
# directly comparable.
ITERATIONS = 5

# The two models the Rust harness (`tests/bench_gguf_convert_adhoc.rs`,
# `src/convert.rs::hub_scaling_bench`) reports on.
TARGET_MODELS = [
    "SmolLM2-135M-Instruct-Q4_K_M.gguf",
    "tinyllama-1.1b-chat-v1.0.Q5_0.gguf",
]


def median_seconds(fn, iterations: int = ITERATIONS) -> float:
    """One warm-up call, then `iterations` timed calls; returns the median."""
    fn()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def dequantize_whole_model(path: Path) -> tuple[int, int, int]:
    """Dequantise every tensor in `path`, mirroring `convert::read_hub`.

    Returns ``(n_tensors, n_dequantized, output_bytes)``.
    """
    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    reader = GGUFReader(path)
    n_dequantized = 0
    output_bytes = 0
    for tensor in reader.tensors:
        qtype = GGMLQuantizationType(tensor.tensor_type)
        if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
            # Scalar passthrough — anamnesis copies the bytes; the closest
            # equivalent here is materialising the array.
            out = tensor.data.copy()
        else:
            out = dequantize(tensor.data, qtype)
            n_dequantized += 1
        output_bytes += out.nbytes
        del out
    return len(reader.tensors), n_dequantized, output_bytes


def main() -> int:
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("gguf (gguf-py) not installed: pip install gguf", file=sys.stderr)
        return 1

    from importlib.metadata import version

    gguf_version = version("gguf")

    if not MODELS.is_dir():
        print(f"no models/ directory at {MODELS} (gitignored)", file=sys.stderr)
        return 1

    wrote_any = False
    for model_name in TARGET_MODELS:
        model_path = MODELS / model_name
        if not model_path.exists():
            print(f"SKIP {model_name}: not present", file=sys.stderr)
            continue

        input_bytes = model_path.stat().st_size
        n_tensors, n_dequantized, output_bytes = dequantize_whole_model(model_path)
        seconds = median_seconds(lambda p=model_path: dequantize_whole_model(p))

        sidecar = {
            "path_label": "gguf_dequant_whole_model",
            "model": model_name,
            "py_seconds": seconds,
            "py_library": f"gguf {gguf_version}",
            "py_output_dtype": "float32",
            "note": (
                "gguf-py dequantises to float32; anamnesis dequantises to BF16 "
                "(half the output bytes). Single-threaded on both sides of this "
                "number - compare against anamnesis at --threads 1."
            ),
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "tensors": n_tensors,
            "dequantized": n_dequantized,
            "iterations": ITERATIONS,
        }
        out_path = HERE / f"{model_path.stem}.dequant.timing.json"
        out_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        wrote_any = True

        mib = input_bytes / (1024 * 1024)
        print(
            f"{model_name}: {seconds * 1000:.1f} ms median "
            f"({n_dequantized}/{n_tensors} dequantised, {mib:.1f} MiB in, "
            f"{output_bytes / (1024 * 1024):.1f} MiB out) -> {out_path.name}"
        )

    return 0 if wrote_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
