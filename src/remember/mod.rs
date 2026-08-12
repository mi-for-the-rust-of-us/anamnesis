// SPDX-License-Identifier: MIT OR Apache-2.0

//! Precision recovery (dequantization) — built on [`parse`](mod@crate::parse).
//!
//! Each submodule handles one quantization family. All operations take raw byte
//! slices from the parsed `.safetensors` file and produce dequantized output as
//! raw `BF16` bytes suitable for writing back to a standard `.safetensors` file.

#[cfg(feature = "awq")]
pub mod awq;
#[cfg(feature = "bnb")]
pub mod bnb;
pub mod fp8;
#[cfg(feature = "gguf")]
pub mod gguf;
#[cfg(feature = "gptq")]
pub mod gptq;
#[cfg(feature = "npz")]
pub mod npz;
#[cfg(feature = "pth")]
pub mod pth;
#[cfg(any(feature = "gptq", feature = "awq"))]
pub(crate) mod quant_utils;

#[cfg(feature = "awq")]
pub use awq::dequantize_awq_to_bf16;
#[cfg(feature = "bnb")]
pub use bnb::{dequantize_bnb_int8_to_bf16, dequantize_bnb4_to_bf16};
pub use fp8::{
    dequantize_fp8_to_bf16, dequantize_per_channel_fp8_to_bf16, dequantize_per_tensor_fp8_to_bf16,
};
#[cfg(feature = "gguf")]
pub use gguf::{dequantize_gguf_blocks_to_bf16, dequantize_gguf_to_bf16};
#[cfg(feature = "gptq")]
pub use gptq::dequantize_gptq_to_bf16;
#[cfg(feature = "npz")]
pub use npz::{npz_to_safetensors, npz_to_safetensors_bytes};
#[cfg(feature = "pth")]
pub use pth::{pth_to_safetensors, pth_to_safetensors_bytes};
