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

use anamnesis::AnamnesisError;

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
        matches!(&parse, Err(AnamnesisError::Parse { reason }) if reason.contains("duplicate tensor name")),
        "parse: {parse:?}"
    );

    let target = anamnesis::ConvertTarget::parse("safetensors").unwrap_or_else(|e| panic!("{e}"));
    let convert = anamnesis::convert_bytes(bytes, target, &anamnesis::ConvertOptions::new());
    assert!(
        matches!(&convert, Err(AnamnesisError::Parse { .. })),
        "convert: {:?}",
        convert.as_ref().map(|(_, stats)| stats)
    );
}

/// A safetensors file whose `__metadata__` declares `"gptq_bits": "0"`.
/// `remember` computed `32 / bits` before the kernel validated the width and
/// divided by zero (`fuzz_safetensors_limits`). The header parser now ignores
/// an unsupported width (falling back to shape inference) and the division
/// site re-checks it.
#[cfg(feature = "gptq")]
#[test]
fn gptq_bits_zero_is_a_clean_error() {
    let bytes = include_bytes!("fixtures/fuzz_regressions/gptq_bits_zero.safetensors");
    // The header may or may not parse; what matters is that nothing panics.
    if let Ok(model) = anamnesis::parse_bytes(bytes.to_vec()) {
        let _ = model.inspect();
        let result = model.remember_to_bytes(anamnesis::TargetDtype::BF16);
        assert!(result.is_err(), "a crafted GPTQ layer should not dequantise");
    }
}
