// SPDX-License-Identifier: MIT OR Apache-2.0

//! `NPZ` archive → safetensors conversion.
//!
//! Mirrors the lossless format-conversion role of
//! `pth_to_safetensors` (the `.pth` sibling, behind the `pth` feature): `NPZ`
//! tensors are already full-precision (`F32`, `F64`, `BF16`, `F16`, integer types), so
//! no dequantisation happens. Tensor names, shapes, dtypes, and raw bytes
//! are preserved exactly. Every [`NpzDtype`] variant has a direct
//! `safetensors::Dtype` counterpart, so every `NPZ` archive round-trips
//! losslessly into the safetensors ecosystem.
//!
//! [`NpzDtype`]: crate::parse::npz::NpzDtype

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::path::Path;

use crate::convert::map_serialize_err;
use crate::error::AnamnesisError;
use crate::parse::npz::{NpzDtype, NpzTensor};

/// Maps an [`NpzDtype`] to its `safetensors::Dtype` counterpart.
///
/// Every variant maps successfully — `safetensors` covers the full `NPZ`
/// dtype range (`Bool`, `U8`/`I8`/…/`U64`/`I64`, `F16`, `BF16`, `F32`,
/// `F64`). The function exists as an exhaustive match rather than a `From`
/// impl so the wildcard arm reads as a compile-time error if `NpzDtype`
/// ever gains a variant `safetensors` cannot represent.
fn npz_dtype_to_safetensors(dtype: NpzDtype) -> safetensors::Dtype {
    match dtype {
        NpzDtype::Bool => safetensors::Dtype::BOOL,
        NpzDtype::U8 => safetensors::Dtype::U8,
        NpzDtype::I8 => safetensors::Dtype::I8,
        NpzDtype::U16 => safetensors::Dtype::U16,
        NpzDtype::I16 => safetensors::Dtype::I16,
        NpzDtype::U32 => safetensors::Dtype::U32,
        NpzDtype::I32 => safetensors::Dtype::I32,
        NpzDtype::U64 => safetensors::Dtype::U64,
        NpzDtype::I64 => safetensors::Dtype::I64,
        NpzDtype::F16 => safetensors::Dtype::F16,
        NpzDtype::BF16 => safetensors::Dtype::BF16,
        NpzDtype::F32 => safetensors::Dtype::F32,
        NpzDtype::F64 => safetensors::Dtype::F64,
    }
}

/// Converts the tensors parsed from an `NPZ` archive to a safetensors file
/// on disk.
///
/// Each tensor is written with its original `NpzDtype` mapped to the
/// matching `safetensors::Dtype`. No dequantisation, no shape changes —
/// `NPZ` is already a passthrough format, so the operation is lossless.
///
/// Tensor names are preserved verbatim from the `NPZ` archive's entry names
/// (without the `.npy` suffix, as
/// [`parse_npz`](crate::parse::npz::parse_npz) already strips it).
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the output file cannot be written.
///
/// Returns [`AnamnesisError::Parse`] if safetensors serialisation fails
/// (e.g., duplicate tensor names — `parse_npz` returns a `HashMap` so the
/// inputs are already unique, but the safetensors writer may still reject
/// other shape/data combinations).
///
/// # Memory
///
/// Allocates a `Vec` of `TensorView` references (one per tensor, metadata
/// only — no data copy). The `safetensors::serialize_to_file` call writes
/// the entire output file in one pass, reading tensor data directly from
/// the input slices. Peak heap ≈ output-file header + view metadata; the
/// bulk tensor bytes stay in the supplied `NpzTensor::data` buffers.
pub fn npz_to_safetensors<S: BuildHasher>(
    tensors: &HashMap<String, NpzTensor, S>,
    output: impl AsRef<Path>,
) -> crate::Result<()> {
    safetensors::tensor::serialize_to_file(npz_views(tensors)?, None, output.as_ref())
        .map_err(map_serialize_err)
}

/// Converts the tensors parsed from an `NPZ` archive to an in-memory
/// safetensors byte buffer.
///
/// Identical to [`npz_to_safetensors`] but returns the serialised bytes
/// instead of writing to a file. Useful for pipelines that feed the buffer
/// directly to downstream consumers (or for round-trip tests that avoid
/// disk I/O).
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if safetensors serialisation fails.
///
/// # Memory
///
/// Allocates a `Vec` of `TensorView` references (metadata only — no data
/// copy) plus the serialised output buffer. Peak heap ≈ the output
/// safetensors size. When `NpzTensor::data` borrows from a memory-mapped
/// source the input data is not duplicated; with the current
/// `parse_npz` API the data is already owned, so the cost is one `Vec`
/// allocation per output buffer (the data buffers themselves are reused).
pub fn npz_to_safetensors_bytes<S: BuildHasher>(
    tensors: &HashMap<String, NpzTensor, S>,
) -> crate::Result<Vec<u8>> {
    safetensors::tensor::serialize(npz_views(tensors)?, None).map_err(map_serialize_err)
}

/// Builds the safetensors views for [`npz_to_safetensors`] and
/// [`npz_to_safetensors_bytes`], sorted by tensor name so the output header
/// has a deterministic order whatever the source `HashMap`'s iteration order.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] if the upstream crate rejects a
/// shape/length pairing.
fn npz_views<S: BuildHasher>(
    tensors: &HashMap<String, NpzTensor, S>,
) -> crate::Result<Vec<(String, safetensors::tensor::TensorView<'_>)>> {
    let mut sorted: Vec<(&String, &NpzTensor)> = tensors.iter().collect();
    sorted.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut views: Vec<(String, safetensors::tensor::TensorView<'_>)> =
        Vec::with_capacity(sorted.len());
    for (name, tensor) in sorted {
        let st_dtype = npz_dtype_to_safetensors(tensor.dtype);
        let view =
            safetensors::tensor::TensorView::new(st_dtype, tensor.shape.clone(), &tensor.data)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("failed to create TensorView for `{name}`: {e}"),
                })?;
        views.push((name.clone(), view));
    }
    Ok(views)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::expect_used,
    clippy::as_conversions,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::*;

    fn make_tensor(name: &str, dtype: NpzDtype, shape: Vec<usize>, data: Vec<u8>) -> NpzTensor {
        NpzTensor {
            name: name.to_owned(),
            shape,
            dtype,
            data,
        }
    }

    #[test]
    fn empty_archive() {
        let tensors: HashMap<String, NpzTensor> = HashMap::new();
        let bytes = npz_to_safetensors_bytes(&tensors).unwrap();
        // safetensors with zero tensors is still a valid file (header `{}`).
        let parsed = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        assert!(parsed.names().is_empty());
    }

    #[test]
    fn roundtrip_mixed_dtypes() {
        let f32_data: Vec<u8> = (0..4u32)
            .flat_map(|i| f32::from(i as u16).to_le_bytes())
            .collect();
        let i32_data: Vec<u8> = (0..2i32).flat_map(i32::to_le_bytes).collect();
        let u8_data: Vec<u8> = vec![1, 2, 3, 4];
        let mut map = HashMap::new();
        map.insert(
            "w".into(),
            make_tensor("w", NpzDtype::F32, vec![2, 2], f32_data.clone()),
        );
        map.insert(
            "idx".into(),
            make_tensor("idx", NpzDtype::I32, vec![2], i32_data.clone()),
        );
        map.insert(
            "bytes".into(),
            make_tensor("bytes", NpzDtype::U8, vec![4], u8_data.clone()),
        );

        let bytes = npz_to_safetensors_bytes(&map).unwrap();
        let parsed = safetensors::SafeTensors::deserialize(&bytes).unwrap();

        let w = parsed.tensor("w").unwrap();
        assert_eq!(w.dtype(), safetensors::Dtype::F32);
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.data(), f32_data.as_slice());

        let idx = parsed.tensor("idx").unwrap();
        assert_eq!(idx.dtype(), safetensors::Dtype::I32);
        assert_eq!(idx.shape(), &[2]);
        assert_eq!(idx.data(), i32_data.as_slice());

        let b = parsed.tensor("bytes").unwrap();
        assert_eq!(b.dtype(), safetensors::Dtype::U8);
        assert_eq!(b.shape(), &[4]);
        assert_eq!(b.data(), u8_data.as_slice());
    }

    #[test]
    fn bf16_dtype_roundtrips() {
        // BF16 has no native NumPy representation, so the NPZ parser
        // produces it from JAX's V2 void dtype. The safetensors writer
        // still emits it as BF16.
        let bf16_data: Vec<u8> = (0..4u16).flat_map(u16::to_le_bytes).collect();
        let mut map = HashMap::new();
        map.insert(
            "w".into(),
            make_tensor("w", NpzDtype::BF16, vec![2, 2], bf16_data.clone()),
        );

        let bytes = npz_to_safetensors_bytes(&map).unwrap();
        let parsed = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let w = parsed.tensor("w").unwrap();
        assert_eq!(w.dtype(), safetensors::Dtype::BF16);
        assert_eq!(w.data(), bf16_data.as_slice());
    }

    #[test]
    fn deterministic_output() {
        // Insert in different orders, expect the same serialised bytes.
        let mut map1 = HashMap::new();
        let mut map2 = HashMap::new();
        for (i, name) in ["alpha", "beta", "gamma"].iter().enumerate() {
            // CAST: usize → u32, indices fit in u32
            #[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
            let bytes_val = (i as u32).to_le_bytes().to_vec();
            map1.insert(
                (*name).into(),
                make_tensor(name, NpzDtype::U32, vec![1], bytes_val.clone()),
            );
            // Reverse insert order.
            map2.insert(
                (*name).into(),
                make_tensor(name, NpzDtype::U32, vec![1], bytes_val),
            );
        }
        let b1 = npz_to_safetensors_bytes(&map1).unwrap();
        let b2 = npz_to_safetensors_bytes(&map2).unwrap();
        assert_eq!(b1, b2, "NPZ→safetensors output must be deterministic");
    }
}
