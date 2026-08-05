// SPDX-License-Identifier: MIT OR Apache-2.0

/// `NPZ`/`NPY` archive parsing — custom `NPY` header parser with bulk `read_exact`.
#[cfg(feature = "npz")]
pub mod npz;

/// Safetensors header parsing, tensor classification, and quantization scheme detection.
pub mod safetensors;

/// PyTorch `.pth` state_dict parsing — minimal pickle VM.
#[cfg(feature = "pth")]
pub mod pth;

/// Vendored read-only `ZIP` central-directory reader shared by the `.pth` and
/// `.npz` parsers (Phase 6.12). Owns the container parsing so the per-entry
/// metadata footprint is bounded; `DEFLATE` inflate stays in `flate2`.
#[cfg(any(feature = "npz", feature = "pth"))]
pub(crate) mod zip;

/// `GGUF` file parsing — header, metadata key-value pairs, and tensor info table.
#[cfg(feature = "gguf")]
pub mod gguf;

/// `GGUF` file writing — the format-symmetric inverse of [`gguf`]. Scalar
/// dtype passthrough only in Phase 6; quantised emit lands in Phase 7.5.
#[cfg(feature = "gguf")]
pub mod gguf_write;

/// `Ollama` model-cache path resolver — turns `llama3.2:1b` into the
/// local `GGUF` blob path. Feature-gated behind `ollama` (which implies
/// [`gguf`] because every Ollama blob is a `GGUF`).
#[cfg(feature = "ollama")]
pub mod ollama;

/// Shared parsing utilities (byte-swap, checked shape-product, etc.) used by
/// multiple format parsers and the dequant/encode layers. Always compiled —
/// `checked_num_elements` is used by the always-on `ParsedModel`; the
/// byte-swap helper inside is itself `npz`/`pth`-gated.
pub(crate) mod utils;

#[cfg(feature = "gguf")]
pub use gguf::{
    inspect_gguf_from_reader, parse_gguf, parse_gguf_bytes, parse_gguf_bytes_with_limits,
    parse_gguf_from_reader, parse_gguf_from_reader_with_limits,
    parse_gguf_front_matter_from_reader, parse_gguf_front_matter_from_reader_with_limits,
    parse_gguf_with_limits, GgufFrontMatter, GgufInspectInfo, GgufMetadataArray, GgufMetadataValue,
    GgufTensor, GgufTensorInfo, GgufType, ParsedGguf,
};
#[cfg(feature = "gguf")]
pub use gguf_write::{write_gguf, write_gguf_to_writer, GgufWriteTensor};
#[cfg(feature = "npz")]
pub use npz::{
    inspect_npz, inspect_npz_from_reader, parse_npz, parse_npz_with_limits, NpzDtype,
    NpzInspectInfo, NpzTensor, NpzTensorInfo,
};
#[cfg(feature = "ollama")]
pub use ollama::resolve_ollama_model;
#[cfg(feature = "pth")]
pub use pth::{
    inspect_pth_from_reader, parse_pth, parse_pth_bytes, parse_pth_bytes_with_limits,
    parse_pth_from_reader, parse_pth_from_reader_with_limits, parse_pth_with_limits, ParsedPth,
    PthDtype, PthInspectInfo, PthTensor, PthTensorInfo,
};
pub use safetensors::{
    parse_safetensors_header, parse_safetensors_header_from_reader,
    parse_safetensors_header_from_reader_with_limits, parse_safetensors_header_with_limits,
    AwqCompanions, AwqConfig, Bnb4Companions, BnbConfig, Dtype, GptqCompanions, GptqConfig,
    QuantScheme, SafetensorsHeader, TensorEntry, TensorRole,
};
