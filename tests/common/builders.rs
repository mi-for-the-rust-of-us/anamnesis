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

/// Builds a `safetensors` file from raw tensors of any dtype, writing the
/// header by hand so dtypes the `safetensors` crate would reject for a view
/// (packed `I32` weights, `F8_E4M3`) can be laid out as quantised checkpoints
/// store them. Inputs: (name, dtype string as in the header, shape, bytes),
/// stored in the given order.
pub fn build_safetensors_raw(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&json);
    out.extend_from_slice(&data);
    out
}

/// A synthetic per-tensor `FP8` checkpoint large enough that its dequantised
/// output dwarfs the header a parse charges: one `F8_E4M3` `[256, 256]` weight
/// with a scalar `F32` scale, and one `BF16` `[256]` passthrough.
pub fn build_fp8_checkpoint() -> Vec<u8> {
    build_safetensors_raw(&[
        ("l.weight", "F8_E4M3", &[256, 256], &vec![0x38u8; 256 * 256]),
        ("l.weight_scale", "F32", &[], &1.0f32.to_le_bytes()),
        ("norm.weight", "BF16", &[256], &vec![0u8; 512]),
    ])
}

/// A synthetic 4-bit `GPTQ` checkpoint, 256 x 256: packed `I32` `qweight`,
/// `qzeros` and `F16` scales in one group.
pub fn build_gptq_checkpoint() -> Vec<u8> {
    let (inf, outf) = (256usize, 256usize);
    build_safetensors_raw(&[
        (
            "l.qweight",
            "I32",
            &[inf / 8, outf],
            &vec![0x11u8; inf / 8 * outf * 4],
        ),
        ("l.qzeros", "I32", &[1, outf / 8], &vec![0u8; outf / 8 * 4]),
        ("l.scales", "F16", &[1, outf], &[0x00, 0x3c].repeat(outf)),
    ])
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

/// Builds an `NPZ` archive from raw `NPY` v2 entries, each `(array name, header
/// dict, data)`, all `STORED`. The header dict is written verbatim (padded and
/// newline-terminated as `NumPy` does), so a test can declare shapes, orders and
/// dtypes `build_npz_f32` would never produce.
pub fn build_npz_raw(entries: &[(&str, &str, &[u8])]) -> Vec<u8> {
    use std::io::Write;
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::<u8>::new()));
    let options: zip::write::SimpleFileOptions =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, dict, data) in entries {
        let mut header = dict.as_bytes().to_vec();
        // v2: magic (6) + version (2) + u32 length (4), then the dict, padded
        // with spaces to a 16-byte boundary and ended by a newline.
        while (12 + header.len() + 1) % 16 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');
        let mut npy = b"\x93NUMPY\x02\x00".to_vec();
        npy.extend_from_slice(&u32::try_from(header.len()).unwrap().to_le_bytes());
        npy.extend_from_slice(&header);
        npy.extend_from_slice(data);
        zip.start_file(format!("{name}.npy"), options).unwrap();
        zip.write_all(&npy).unwrap();
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
