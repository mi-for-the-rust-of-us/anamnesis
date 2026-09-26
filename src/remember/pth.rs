// SPDX-License-Identifier: MIT OR Apache-2.0

//! `.pth` → safetensors lossless format conversion.
//!
//! Unlike the other `remember` submodules (`FP8`, `GPTQ`, `AWQ`, `BnB`), this module
//! performs **no dequantization** — `.pth` tensors are already full-precision.
//! The conversion is a pure format change: tensor names, shapes, dtypes, and
//! raw bytes are preserved exactly.

use std::path::Path;

use crate::convert::map_serialize_err;
use crate::error::AnamnesisError;
use crate::parse::pth::PthTensor;

/// Converts parsed `.pth` tensors to a safetensors file.
///
/// Each tensor is written with its original dtype — no dequantization or
/// dtype conversion. Tensor names are preserved as-is from the `state_dict`
/// keys (e.g., `"rnn.weight_ih_l0"`, `"linear.weight"`).
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if a tensor dtype has no
/// safetensors equivalent (currently all `PthDtype` variants map
/// successfully).
///
/// Returns [`AnamnesisError::Io`] if the output file cannot be written.
///
/// Returns [`AnamnesisError::Parse`] if safetensors serialization fails
/// (e.g., duplicate tensor names, shape/data mismatch).
///
/// # Memory
///
/// Allocates a `Vec` of `TensorView` references (one per tensor, metadata
/// only — no data copy). The `safetensors::serialize_to_file` call writes
/// the entire output file in one pass, reading tensor data directly from
/// the input slices. When tensors are `Cow::Borrowed` (zero-copy from an
/// mmap), peak heap usage ≈ output file header + view metadata.
pub fn pth_to_safetensors(
    tensors: &[PthTensor<'_>],
    output: impl AsRef<Path>,
) -> crate::Result<()> {
    safetensors::tensor::serialize_to_file(pth_views(tensors)?, None, output.as_ref())
        .map_err(map_serialize_err)
}

/// Converts parsed `.pth` tensors to an in-memory safetensors byte buffer.
///
/// Identical to [`pth_to_safetensors`] but returns the serialised bytes
/// instead of writing to a file. Useful for pipelines that feed the buffer
/// directly to `VarBuilder::from_buffered_safetensors` without touching the
/// filesystem.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if a tensor dtype has no
/// safetensors equivalent (currently all `PthDtype` variants map
/// successfully).
///
/// Returns [`AnamnesisError::Parse`] if safetensors serialization fails
/// (e.g., duplicate tensor names, shape/data mismatch).
///
/// # Memory
///
/// Allocates a `Vec` of `TensorView` references (metadata only — no data
/// copy) plus the serialised output buffer. Peak heap ≈ the output
/// safetensors size. When tensors are `Cow::Borrowed` (zero-copy from an
/// mmap), input data is not duplicated.
pub fn pth_to_safetensors_bytes(tensors: &[PthTensor<'_>]) -> crate::Result<Vec<u8>> {
    safetensors::tensor::serialize(pth_views(tensors)?, None).map_err(map_serialize_err)
}

/// Builds the safetensors views for [`pth_to_safetensors`] and
/// [`pth_to_safetensors_bytes`], so the file and in-memory destinations cannot
/// drift on dtype mapping, shape, or order.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if a dtype has no safetensors
/// equivalent, and [`AnamnesisError::Parse`] if two tensors share a name or
/// the upstream crate rejects a shape/length pairing.
fn pth_views<'a>(
    tensors: &'a [PthTensor<'_>],
) -> crate::Result<Vec<(String, safetensors::tensor::TensorView<'a>)>> {
    crate::parse::utils::reject_duplicate_names(tensors.iter().map(|t| t.name.as_str()))?;
    let mut views: Vec<(String, safetensors::tensor::TensorView<'a>)> =
        Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let st_dtype = tensor.dtype.to_safetensors_dtype()?;
        // Cow<[u8]> derefs to &[u8] — zero-copy when Borrowed (from mmap).
        let view =
            safetensors::tensor::TensorView::new(st_dtype, tensor.shape.clone(), &tensor.data)
                .map_err(|e| AnamnesisError::Parse {
                    reason: format!("failed to create TensorView for `{}`: {e}", tensor.name),
                })?;
        views.push((tensor.name.clone(), view));
    }
    Ok(views)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::parse::pth::PthDtype;
    use std::borrow::Cow;

    #[test]
    fn roundtrip_simple() {
        let weight_data: Vec<u8> = vec![
            0x00, 0x00, 0x80, 0x3F, // 1.0f32 LE
            0x00, 0x00, 0x00, 0x40, // 2.0f32 LE
            0x00, 0x00, 0x40, 0x40, // 3.0f32 LE
            0x00, 0x00, 0x80, 0x40, // 4.0f32 LE
        ];
        let bias_data: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x3F, // 0.5f32 LE
            0x00, 0x00, 0x00, 0xBF, // -0.5f32 LE
        ];
        let tensors = vec![
            PthTensor {
                name: "weight".into(),
                shape: vec![2, 2],
                dtype: PthDtype::F32,
                data: Cow::Borrowed(&weight_data),
            },
            PthTensor {
                name: "bias".into(),
                shape: vec![2],
                dtype: PthDtype::F32,
                data: Cow::Borrowed(&bias_data),
            },
        ];

        // Use a TempDir + fresh path (not NamedTempFile.path()): safetensors'
        // `serialize_to_file` opens/maps the path itself, and on Windows that
        // collides with the open handle a NamedTempFile keeps on the same file.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.safetensors");
        pth_to_safetensors(&tensors, &path).unwrap();

        // Read back with safetensors crate and verify.
        let data = std::fs::read(&path).unwrap();
        let st = safetensors::SafeTensors::deserialize(&data).unwrap();

        assert_eq!(st.len(), 2);

        let w = st.tensor("weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.dtype(), safetensors::Dtype::F32);
        assert_eq!(w.data(), weight_data.as_slice());

        let b = st.tensor("bias").unwrap();
        assert_eq!(b.shape(), &[2]);
        assert_eq!(b.dtype(), safetensors::Dtype::F32);
        assert_eq!(b.data(), bias_data.as_slice());
    }

    #[test]
    fn empty_tensors() {
        let tensors: Vec<PthTensor<'_>> = vec![];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.safetensors");
        pth_to_safetensors(&tensors, &path).unwrap();

        let data = std::fs::read(&path).unwrap();
        let st = safetensors::SafeTensors::deserialize(&data).unwrap();
        assert_eq!(st.len(), 0);
    }

    #[test]
    fn roundtrip_bytes() {
        let weight_data: Vec<u8> = vec![
            0x00, 0x00, 0x80, 0x3F, // 1.0f32 LE
            0x00, 0x00, 0x00, 0x40, // 2.0f32 LE
            0x00, 0x00, 0x40, 0x40, // 3.0f32 LE
            0x00, 0x00, 0x80, 0x40, // 4.0f32 LE
        ];
        let bias_data: Vec<u8> = vec![
            0x00, 0x00, 0x00, 0x3F, // 0.5f32 LE
            0x00, 0x00, 0x00, 0xBF, // -0.5f32 LE
        ];
        let tensors = vec![
            PthTensor {
                name: "weight".into(),
                shape: vec![2, 2],
                dtype: PthDtype::F32,
                data: Cow::Borrowed(&weight_data),
            },
            PthTensor {
                name: "bias".into(),
                shape: vec![2],
                dtype: PthDtype::F32,
                data: Cow::Borrowed(&bias_data),
            },
        ];

        let bytes = pth_to_safetensors_bytes(&tensors).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();

        assert_eq!(st.len(), 2);

        let w = st.tensor("weight").unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(w.dtype(), safetensors::Dtype::F32);
        assert_eq!(w.data(), weight_data.as_slice());

        let b = st.tensor("bias").unwrap();
        assert_eq!(b.shape(), &[2]);
        assert_eq!(b.dtype(), safetensors::Dtype::F32);
        assert_eq!(b.data(), bias_data.as_slice());
    }

    #[test]
    fn empty_tensors_bytes() {
        let tensors: Vec<PthTensor<'_>> = vec![];
        let bytes = pth_to_safetensors_bytes(&tensors).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(st.len(), 0);
    }
}
