// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inputs a fuzz campaign found crashing, pinned so they stay fixed.
//!
//! Each fixture under `tests/fixtures/fuzz_regressions/` is the body of a
//! libFuzzer crash artefact with the target's limits prefix stripped (every
//! one here ran under unbounded limits, so the default `ParseLimits` reproduce
//! it). Each test asserts a clean `Err`; before its fix, each input panicked,
//! which in a release build (`panic = "abort"`) kills the process and under the
//! Python bindings would surface as a `PanicException` instead of a typed
//! error. Found by the v0.7.8 campaign in WSL (`fuzz/README.md` § Status).

#![allow(clippy::panic)]

/// A `.pth` whose pickled `OrderedDict` names one tensor twice. The duplicate
/// reached the upstream `safetensors` serializer, which indexes a table sized
/// by the distinct names and panicked (`fuzz_convert_bytes`). Now rejected
/// where the names enter, and before every write.
#[cfg(feature = "pth")]
#[test]
fn pth_duplicate_state_dict_key_is_a_clean_error() {
    let bytes = include_bytes!("fixtures/fuzz_regressions/pth_duplicate_state_dict_key.pth");

    let parse = anamnesis::parse_pth_bytes(bytes.to_vec());
    assert!(
        matches!(&parse, Err(anamnesis::AnamnesisError::Parse { reason }) if reason.contains("duplicate tensor name")),
        "parse: {parse:?}"
    );

    let target = anamnesis::ConvertTarget::parse("safetensors").unwrap_or_else(|e| panic!("{e}"));
    let convert = anamnesis::convert_bytes(bytes, target, &anamnesis::ConvertOptions::new());
    assert!(
        matches!(&convert, Err(anamnesis::AnamnesisError::Parse { .. })),
        "convert: {:?}",
        convert.as_ref().map(|(_, stats)| stats)
    );
}

/// A safetensors file whose `__metadata__` declares `"gptq_bits": "0"`.
/// `remember` computed `32 / bits` before the kernel validated the width and
/// divided by zero (`fuzz_safetensors_limits`).
///
/// With the fix the header parser ignores the unsupported width and infers
/// bits = 4 from the shapes, so this input no longer reaches the division at
/// all; `remember` then fails on the fuzzer's out-of-range `g_idx` bytes. The
/// invariant pinned here is therefore "parses, and `remember` returns an error
/// rather than panicking". The division guard itself is driven directly by
/// `gptq_bits_zero_in_config_is_unsupported` below.
#[cfg(feature = "gptq")]
#[test]
fn gptq_bits_zero_is_a_clean_error() {
    let bytes = include_bytes!("fixtures/fuzz_regressions/gptq_bits_zero.safetensors");
    let model = anamnesis::parse_bytes(bytes.to_vec())
        .unwrap_or_else(|e| panic!("falls back to shape inference, so it parses: {e}"));
    assert_eq!(model.header.gptq_config.as_ref().map(|c| c.bits), Some(4));
    let _ = model.inspect();
    assert!(
        model
            .remember_to_bytes(anamnesis::TargetDtype::BF16)
            .is_err()
    );
}

/// `GptqConfig`'s fields are public, so a caller can hand `remember` a width
/// the header parser would never produce. The division site re-checks it and
/// answers `Unsupported` instead of dividing by zero.
#[cfg(feature = "gptq")]
#[test]
fn gptq_bits_zero_in_config_is_unsupported() {
    let bytes = std::fs::read("tests/fixtures/safetensors_reference/gptq.safetensors")
        .unwrap_or_else(|e| panic!("gptq fixture: {e}"));
    let mut model = anamnesis::parse_bytes(bytes).unwrap_or_else(|e| panic!("parse: {e}"));
    let config = model
        .header
        .gptq_config
        .as_mut()
        .unwrap_or_else(|| panic!("fixture carries a GPTQ config"));
    config.bits = 0;
    let result = model.remember_to_bytes(anamnesis::TargetDtype::BF16);
    assert!(
        matches!(&result, Err(anamnesis::AnamnesisError::Unsupported { detail, .. })
            if detail.contains("0-bit")),
        "{:?}",
        result.as_ref().map(Vec::len)
    );
}

/// Found by the v0.7.8 consistency review, the same class as the `.pth` case
/// above: `remember` renames `<layer>.qweight` to `<layer>.weight`, so a GPTQ
/// file that also carries a plain `<layer>.weight` handed the upstream
/// serializer two tensors with one name, which it panics on.
#[cfg(feature = "gptq")]
#[test]
fn gptq_rename_collision_is_a_clean_error() {
    use safetensors::tensor::TensorView;

    let base = std::fs::read("tests/fixtures/safetensors_reference/gptq.safetensors")
        .unwrap_or_else(|e| panic!("gptq fixture: {e}"));
    let (_, metadata) = safetensors::SafeTensors::read_metadata(&base)
        .unwrap_or_else(|e| panic!("fixture header: {e}"));
    let tensors = safetensors::SafeTensors::deserialize(&base)
        .unwrap_or_else(|e| panic!("fixture body: {e}"));

    let layer = tensors
        .names()
        .into_iter()
        .find_map(|n| n.strip_suffix(".qweight"))
        .unwrap_or_else(|| panic!("fixture has no .qweight"))
        .to_owned();
    let clash = [0u8; 4]; // two BF16 values
    let mut views: Vec<(String, TensorView<'_>)> = tensors.tensors();
    views.push((
        format!("{layer}.weight"),
        TensorView::new(safetensors::Dtype::BF16, vec![2], &clash)
            .unwrap_or_else(|e| panic!("clash view: {e}")),
    ));
    let bytes = safetensors::tensor::serialize(views, metadata.metadata().clone())
        .unwrap_or_else(|e| panic!("serialize: {e}"));

    let model = anamnesis::parse_bytes(bytes).unwrap_or_else(|e| panic!("parse: {e}"));
    let result = model.remember_to_bytes(anamnesis::TargetDtype::BF16);
    assert!(
        matches!(&result, Err(anamnesis::AnamnesisError::Parse { reason })
            if reason.contains("duplicate tensor name")),
        "{:?}",
        result.as_ref().map(Vec::len)
    );
}
