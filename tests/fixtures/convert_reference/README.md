# Phase 6 — convert-pipeline timing sidecars

This directory holds **optional** `*.timing.json` files that record
Python-side wall-clock measurements for each conversion the
`tests/cross_validation_convert.rs` suite exercises. The Rust tests
print a "vs Python" comparison line when a sidecar is present, and
silently skip the comparison when one is absent (the test still passes
on its byte-exactness assertion either way).

## Sidecar shape

```json
{
  "path_label": "st_to_gguf",
  "py_seconds": 0.0234,
  "py_library": "gguf 0.10.0",
  "shape": [4096, 4096]
}
```

The Rust test looks up the sidecar by `path_label` and reports
`rust=… µs, python=… µs (Nx)`.

## Generating / refreshing sidecars

Run `generate_convert_timings.py` (in this directory) against a Python
environment with `numpy`, `safetensors`,
`gguf`, and `bitsandbytes` installed. The script writes one JSON per
path label listed in `tests/cross_validation_convert.rs::report_vs_python`.

The sidecar files are **checked into the repo** so users who don't
have the Python deps installed still see meaningful "vs Python"
numbers in their test output. The script exists to refresh the
numbers when the environment changes — it is not a test-time
dependency.

Path labels currently emitted by the Rust test:

| Label | Conversion | Python equivalent |
|---|---|---|
| `npz_to_st` | NPZ → safetensors | `numpy.savez_compressed` + `safetensors.numpy.save` |
| `pth_to_st` | PTH → safetensors | `torch.load` + `safetensors.torch.save` |
| `st_to_gguf` | safetensors-BF16 → GGUF | `gguf.GGUFWriter.add_tensor` |
| `st_to_bnb_nf4` | safetensors-BF16 → BnB-NF4 | `bitsandbytes.functional.quantize_4bit(quant_type="nf4")` |
