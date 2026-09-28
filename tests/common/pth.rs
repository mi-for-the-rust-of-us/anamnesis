// SPDX-License-Identifier: MIT OR Apache-2.0

//! In-memory `.pth` builder for hostile-input tests.
//!
//! No `.pth` builder existed before Phase 7.9: the reference fixtures come from
//! `torch.save`, and the in-crate unit tests hand-write opcode bytes. The
//! security regressions need *crafted* state dicts (zero strides, aliased
//! storages, huge shapes) that `torch.save` would never produce, so this writes
//! the pickle opcodes a `state_dict` save emits, and wraps them in the `STORED`
//! `ZIP` layout `torch.save` uses (`archive/data.pkl`, `archive/data/<key>`).

use std::io::Write;

/// One tensor of a state dict: `name → _rebuild_tensor_v2(storage, offset,
/// shape, strides)`.
pub struct TensorSpec<'a> {
    /// State-dict key.
    pub name: &'a str,
    /// `torch` storage class, e.g. `"ByteStorage"` (`U8`) or `"FloatStorage"`
    /// (`F32`).
    pub storage_class: &'a str,
    /// Storage key: the tensor's bytes live in `archive/data/<key>`.
    pub storage_key: &'a str,
    /// Storage offset, in elements.
    pub offset: i64,
    /// Shape, outermost first.
    pub shape: &'a [i64],
    /// Strides, in elements.
    pub strides: &'a [i64],
}

/// Appends a `BINUNICODE` string.
fn push_str(p: &mut Vec<u8>, text: &str) {
    p.push(b'X');
    p.extend_from_slice(&u32::try_from(text.len()).unwrap().to_le_bytes());
    p.extend_from_slice(text.as_bytes());
}

/// Appends a `LONG1` 8-byte integer.
fn push_int(p: &mut Vec<u8>, value: i64) {
    p.push(0x8a);
    p.push(8);
    p.extend_from_slice(&value.to_le_bytes());
}

/// Appends a tuple of integers (`MARK … TUPLE`).
fn push_int_tuple(p: &mut Vec<u8>, values: &[i64]) {
    p.push(b'(');
    for &v in values {
        push_int(p, v);
    }
    p.push(b't');
}

/// Appends `_rebuild_tensor_v2(PersistentId(("storage", torch.<class>, key,
/// "cpu", 1)), offset, shape, strides)`, leaving the tensor on the stack.
fn push_rebuild(p: &mut Vec<u8>, t: &TensorSpec<'_>) {
    p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n");
    p.push(b'(');
    p.push(b'(');
    push_str(p, "storage");
    p.extend_from_slice(format!("ctorch\n{}\n", t.storage_class).as_bytes());
    push_str(p, t.storage_key);
    push_str(p, "cpu");
    push_int(p, 1);
    p.push(b't');
    p.push(b'Q'); // BINPERSID
    push_int(p, t.offset);
    push_int_tuple(p, t.shape);
    push_int_tuple(p, t.strides);
    p.push(b't');
    p.push(b'R'); // REDUCE
}

/// The `data.pkl` of a state dict holding `tensors`, in order.
pub fn state_dict_pickle(tensors: &[TensorSpec<'_>]) -> Vec<u8> {
    let mut p = vec![0x80, 0x02, b'}']; // PROTO 2, EMPTY_DICT
    for t in tensors {
        push_str(&mut p, t.name);
        push_rebuild(&mut p, t);
        p.push(b's'); // SETITEM
    }
    p.push(b'.');
    p
}

/// The `data.pkl` of a state dict whose `count` keys (`t0`, `t1`, …) all name
/// the **same** tensor object: built once, memoised, then replayed with `BINGET`.
/// This is how `torch.save` writes tied weights, with `count` chosen by the
/// attacker instead of by the model.
pub fn aliased_state_dict_pickle(tensor: &TensorSpec<'_>, count: usize) -> Vec<u8> {
    let mut p = vec![0x80, 0x02, b'}'];
    push_str(&mut p, "t0");
    push_rebuild(&mut p, tensor);
    p.extend_from_slice(&[b'q', 0]); // BINPUT 0
    p.push(b's');
    for i in 1..count {
        push_str(&mut p, &format!("t{i}"));
        p.extend_from_slice(&[b'h', 0]); // BINGET 0
        p.push(b's');
    }
    p.push(b'.');
    p
}

/// A `.pth` archive: `archive/data.pkl`, then `archive/data/<key>` for each
/// storage, then `archive/byteorder` if given, all `STORED`.
pub fn pth_archive(pkl: &[u8], storages: &[(&str, &[u8])], byteorder: Option<&[u8]>) -> Vec<u8> {
    let mut zip = zip::ZipWriter::new(std::io::Cursor::new(Vec::<u8>::new()));
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(false);
    zip.start_file("archive/data.pkl", options).unwrap();
    zip.write_all(pkl).unwrap();
    for (key, bytes) in storages {
        zip.start_file(format!("archive/data/{key}"), options)
            .unwrap();
        zip.write_all(bytes).unwrap();
    }
    if let Some(order) = byteorder {
        zip.start_file("archive/byteorder", options).unwrap();
        zip.write_all(order).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

/// A one-tensor `.pth` of `U8` elements over a single storage `"0"`.
pub fn single_u8_view(shape: &[i64], strides: &[i64], storage: &[u8]) -> Vec<u8> {
    let pkl = state_dict_pickle(&[TensorSpec {
        name: "w",
        storage_class: "ByteStorage",
        storage_key: "0",
        offset: 0,
        shape,
        strides,
    }]);
    pth_archive(&pkl, &[("0", storage)], None)
}
