// SPDX-License-Identifier: MIT OR Apache-2.0

//! In-memory synthetic fixture builders for the `convert` suites
//! (`cli_convert.rs`, `cross_validation_convert.rs`, `bench_convert_adhoc.rs`).

use std::path::PathBuf;

/// Builds an in-memory `BF16` `safetensors` file from a tensor list. Inputs:
/// (name, row-major shape, `BF16` LE bytes). Tensors are emitted in the
/// `safetensors` crate's iteration order: for round-trip tests, that ordering
/// is what we compare against.
pub fn build_safetensors_bf16(tensors: &[(&str, &[usize], &[u8])]) -> Vec<u8> {
    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, data)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::BF16,
                shape.to_vec(),
                data,
            )
            .unwrap();
            (*name, view)
        })
        .collect();
    safetensors::tensor::serialize(views, None).unwrap()
}

/// Builds a tiny `F32` `NPZ` archive (a `ZIP` of `.npy` entries) in-memory.
pub fn build_npz_f32(tensors: &[(&str, &[usize], &[u8])]) -> Vec<u8> {
    use std::io::Write;
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::<u8>::new()));
    let options: zip::write::SimpleFileOptions =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, shape, data) in tensors {
        let entry = format!("{name}.npy");
        zip.start_file(&entry, options).unwrap();
        // Write minimal NPY v1.0 header: magic + version + header_len + dict + padding.
        let shape_str: Vec<String> = shape.iter().map(usize::to_string).collect();
        let shape_tuple = if shape.len() == 1 {
            format!("({},)", shape_str[0])
        } else {
            format!("({})", shape_str.join(", "))
        };
        let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_tuple}, }}");
        // Header layout: 10 bytes (magic+version+u16 len) + dict + padding + newline,
        // padded to 64-byte boundary.
        let mut header = dict.into_bytes();
        let header_total = 10 + header.len() + 1;
        let pad = (64 - header_total % 64) % 64;
        for _ in 0..pad {
            header.push(b' ');
        }
        header.push(b'\n');
        let header_len_u16 = u16::try_from(header.len()).unwrap();
        let mut entry_bytes: Vec<u8> = Vec::with_capacity(10 + header.len() + data.len());
        entry_bytes.extend_from_slice(&[0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0]);
        entry_bytes.extend_from_slice(&header_len_u16.to_le_bytes());
        entry_bytes.extend_from_slice(&header);
        entry_bytes.extend_from_slice(data);
        zip.write_all(&entry_bytes).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

/// Writes `bytes` to `fixture.<ext>` in a fresh temp directory. The returned
/// `TempDir` must outlive every use of the path.
pub fn write_temp(bytes: &[u8], ext: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("create temp dir");
    let path = dir.path().join(format!("fixture.{ext}"));
    std::fs::write(&path, bytes).unwrap();
    (dir, path)
}
