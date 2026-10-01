// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-validation of NVIDIA `ModelOpt` `NVFP4` safetensors against `modelopt`.
//!
//! The fixture is a real layer: the first 16 rows (65 536 values) of
//! `model.layers.0.self_attn.q_proj` from `nvidia/Llama-3.1-8B-Instruct-NVFP4`
//! at revision `bdb54e24`, kept in its original `ModelOpt` layout (packed `U8`
//! weight, `F8_E4M3` `weight_scale`, `F32` `weight_scale_2` and
//! `input_scale`). Both goldens come from `modelopt` 0.47.0's own
//! `NVFP4QTensor.dequantize` (`tests/fixtures/nvfp4_reference/generate_nvfp4.py`).
//!
//! Two paths are checked, each at `F32` bit for bit and at `BF16` with no
//! tolerance: the kernel called directly, and the whole public path a user's
//! file takes (`parse_bytes`, scheme detection, `remember_to_bytes`), where
//! `inspect` must also report exactly the bytes written.

#![cfg(feature = "nvfp4")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]

use anamnesis::{Bf16Out, F32Out, QuantScheme, TargetDtype};

const FIXTURE: &[u8] =
    include_bytes!("fixtures/nvfp4_reference/llama31_8b_nvfp4_q_proj.safetensors");
const GOLDEN: &[u8] = include_bytes!("fixtures/nvfp4_reference/llama31_8b_nvfp4_q_proj.golden");
const LAYER: &str = "model.layers.0.self_attn.q_proj";

/// `(bf16 golden, f32 golden)` from the `AMNV` container.
fn goldens() -> (&'static [u8], &'static [u8]) {
    assert_eq!(&GOLDEN[..4], b"AMNV");
    assert_eq!(
        u32::from_le_bytes(GOLDEN[4..8].try_into().unwrap()),
        1,
        "version"
    );
    let n = usize::try_from(u32::from_le_bytes(GOLDEN[8..12].try_into().unwrap())).unwrap();
    assert_eq!(n, 65_536);
    let bf16 = &GOLDEN[12..12 + 2 * n];
    let f32 = &GOLDEN[12 + 2 * n..12 + 6 * n];
    assert_eq!(GOLDEN.len(), 12 + 6 * n);
    (bf16, f32)
}

fn mismatches(actual: &[u8], expected: &[u8], width: usize) -> usize {
    assert_eq!(actual.len(), expected.len());
    actual
        .chunks_exact(width)
        .zip(expected.chunks_exact(width))
        .filter(|(a, b)| a != b)
        .count()
}

#[test]
fn kernel_matches_modelopt_bit_for_bit() {
    let st = safetensors::SafeTensors::deserialize(FIXTURE).unwrap();
    let weight = st.tensor(&format!("{LAYER}.weight")).unwrap();
    let scale = st.tensor(&format!("{LAYER}.weight_scale")).unwrap();
    let scale_2 = st.tensor(&format!("{LAYER}.weight_scale_2")).unwrap();
    let global = f32::from_le_bytes(scale_2.data().try_into().unwrap());
    let (rows, cols) = (weight.shape()[0], weight.shape()[1] * 2);
    let (bf16, f32) = goldens();

    let out =
        anamnesis::dequantize_nvfp4::<F32Out>(weight.data(), scale.data(), global, rows, cols)
            .unwrap();
    assert_eq!(mismatches(&out, f32, 4), 0, "F32 differs from modelopt");
    let out =
        anamnesis::dequantize_nvfp4::<Bf16Out>(weight.data(), scale.data(), global, rows, cols)
            .unwrap();
    assert_eq!(mismatches(&out, bf16, 2), 0, "BF16 differs from modelopt");
}

#[test]
fn remember_matches_modelopt_through_the_public_path() {
    let model = anamnesis::parse_bytes(FIXTURE.to_vec()).unwrap();
    assert_eq!(model.header.scheme, QuantScheme::Nvfp4);
    let (bf16, f32) = goldens();

    for (dtype, golden, width) in [(TargetDtype::F32, f32, 4), (TargetDtype::BF16, bf16, 2)] {
        let written = model.remember_to_bytes(dtype).unwrap();
        let st = safetensors::SafeTensors::deserialize(&written).unwrap();
        // Only the weight survives: both scales and the activation scale are
        // the quantised layer's companions.
        assert_eq!(st.names(), vec![&format!("{LAYER}.weight")], "{dtype:?}");
        let weight = st.tensor(&format!("{LAYER}.weight")).unwrap();
        assert_eq!(
            weight.shape(),
            [16, 4096],
            "logical shape, not packed bytes"
        );
        assert_eq!(mismatches(weight.data(), golden, width), 0, "{dtype:?}");

        let estimate = model
            .inspect_with_options(&anamnesis::InspectOptions::new().with_output_dtype(dtype))
            .dequantized_size;
        assert_eq!(
            estimate,
            u64::try_from(weight.data().len()).unwrap(),
            "{dtype:?}"
        );
    }
}

/// The anamnesis half of the `NVFP4` speed figure; the `modelopt` half is
/// `tests/fixtures/nvfp4_reference/time_modelopt.py`. Same protocol as the
/// `GGUF` table (`tests/bench_gguf_table_adhoc.rs`): 20 warm-up calls, 200
/// timed calls, min and median, on the fixture's 65 536 values. Run both in
/// the same environment, one after the other:
///
/// ```text
/// cargo test --release --features nvfp4 --test cross_validation_nvfp4 -- --ignored --nocapture
/// python tests/fixtures/nvfp4_reference/time_modelopt.py
/// ```
#[test]
#[ignore = "ad-hoc benchmark; run with --release --features nvfp4 --ignored --nocapture"]
fn bench_nvfp4_kernel() {
    let st = safetensors::SafeTensors::deserialize(FIXTURE).unwrap();
    let weight = st.tensor(&format!("{LAYER}.weight")).unwrap();
    let scale = st.tensor(&format!("{LAYER}.weight_scale")).unwrap();
    let scale_2 = st.tensor(&format!("{LAYER}.weight_scale_2")).unwrap();
    let global = f32::from_le_bytes(scale_2.data().try_into().unwrap());
    let (rows, cols) = (weight.shape()[0], weight.shape()[1] * 2);
    let mut samples: Vec<std::time::Duration> = Vec::with_capacity(200);
    for i in 0..220 {
        let start = std::time::Instant::now();
        let out =
            anamnesis::dequantize_nvfp4::<Bf16Out>(weight.data(), scale.data(), global, rows, cols)
                .unwrap();
        let elapsed = start.elapsed();
        assert_eq!(out.len(), rows * cols * 2);
        if i >= 20 {
            samples.push(elapsed);
        }
    }
    samples.sort_unstable();
    eprintln!(
        "anamnesis NVFP4 -> BF16, {} values: min {:.1} us, median {:.1} us (20 warm-up, 200 timed)",
        rows * cols,
        samples[0].as_secs_f64() * 1e6,
        samples[100].as_secs_f64() * 1e6
    );
}
