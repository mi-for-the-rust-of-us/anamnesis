// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-validation tests for `AWQ` dequantization against `PyTorch` reference.

#![cfg(feature = "awq")]
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::wildcard_enum_match_arm
)]

use std::time::Instant;

use anamnesis::{Dtype, dequantize_awq_to_bf16};

// ---------------------------------------------------------------------------
// Fixture parsing
// ---------------------------------------------------------------------------

/// Binary fixture layout (all little-endian):
///
/// - 4 bytes: `bits` (`u32`)
/// - 4 bytes: `group_size` (`u32`)
/// - 4 bytes: `in_features` (`u32`)
/// - 4 bytes: `out_features` (`u32`)
/// - 4 bytes: `scale_dtype` (0=`F32`, 1=`BF16`, 2=`F16`)
/// - 4 bytes: `qweight_len` (`u32`)
/// - 4 bytes: `scales_len` (`u32`)
/// - 4 bytes: `qzeros_len` (`u32`)
/// - 4 bytes: `expected_len` (`u32`)
/// - qweight bytes, scales bytes, qzeros bytes, expected `BF16` bytes
struct AwqFixture {
    bits: u8,
    group_size: usize,
    in_features: usize,
    out_features: usize,
    scale_dtype: Dtype,
    qweight_data: Vec<u8>,
    scales_data: Vec<u8>,
    qzeros_data: Vec<u8>,
    expected_bf16: Vec<u8>,
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
    u32::from_le_bytes(bytes)
}

fn parse_awq_fixture(data: &[u8]) -> AwqFixture {
    let bits = read_u32_le(data, 0) as u8;
    let group_size = read_u32_le(data, 4) as usize;
    let in_features = read_u32_le(data, 8) as usize;
    let out_features = read_u32_le(data, 12) as usize;
    let scale_dtype_id = read_u32_le(data, 16);
    let qweight_len = read_u32_le(data, 20) as usize;
    let scales_len = read_u32_le(data, 24) as usize;
    let qzeros_len = read_u32_le(data, 28) as usize;
    let expected_len = read_u32_le(data, 32) as usize;

    let header_size = 36;
    let qw_start = header_size;
    let sc_start = qw_start + qweight_len;
    let qz_start = sc_start + scales_len;
    let ex_start = qz_start + qzeros_len;

    let scale_dtype = match scale_dtype_id {
        0 => Dtype::F32,
        1 => Dtype::BF16,
        2 => Dtype::F16,
        other => panic!("unknown scale dtype id: {other}"),
    };

    AwqFixture {
        bits,
        group_size,
        in_features,
        out_features,
        scale_dtype,
        qweight_data: data[qw_start..qw_start + qweight_len].to_vec(),
        scales_data: data[sc_start..sc_start + scales_len].to_vec(),
        qzeros_data: data[qz_start..qz_start + qzeros_len].to_vec(),
        expected_bf16: data[ex_start..ex_start + expected_len].to_vec(),
    }
}

// ---------------------------------------------------------------------------
// BF16 comparison
// ---------------------------------------------------------------------------

fn compare_bf16(actual: &[u8], expected: &[u8], max_ulp_diff: u16) -> (usize, u16) {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");
    let mut mismatches = 0;
    let mut max_diff: u16 = 0;

    for (i, (a_pair, e_pair)) in actual
        .chunks_exact(2)
        .zip(expected.chunks_exact(2))
        .enumerate()
    {
        let a_bits = u16::from_le_bytes([a_pair[0], a_pair[1]]);
        let e_bits = u16::from_le_bytes([e_pair[0], e_pair[1]]);

        // BITWISE: BF16 exponent is 8 bits [14:7], mask = 0x7F80
        let a_is_nan = (a_bits & 0x7F80 == 0x7F80) && (a_bits & 0x007F != 0);
        let e_is_nan = (e_bits & 0x7F80 == 0x7F80) && (e_bits & 0x007F != 0);
        if a_is_nan && e_is_nan {
            continue;
        }
        if a_is_nan != e_is_nan {
            mismatches += 1;
            continue;
        }

        let diff = a_bits.abs_diff(e_bits);
        if diff > max_ulp_diff {
            mismatches += 1;
            if i < 5 {
                eprintln!(
                    "  element {i}: actual=0x{a_bits:04X}, expected=0x{e_bits:04X}, diff={diff} ULP"
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    (mismatches, max_diff)
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

fn run_awq_cross_validation(name: &str, data: &[u8], max_ulp: u16) {
    let fixture = parse_awq_fixture(data);
    let total = fixture.in_features * fixture.out_features;

    eprintln!(
        "{name}: {}-bit, group_size={}, {}×{} = {total} elements",
        fixture.bits, fixture.group_size, fixture.in_features, fixture.out_features,
    );

    let start = Instant::now();
    let actual = dequantize_awq_to_bf16(
        &fixture.qweight_data,
        &fixture.scales_data,
        &fixture.qzeros_data,
        fixture.in_features,
        fixture.out_features,
        fixture.group_size,
        fixture.bits,
        fixture.scale_dtype,
    )
    .expect("AWQ dequant failed");
    let elapsed = start.elapsed();

    assert_eq!(actual.len(), fixture.expected_bf16.len());

    let (mismatches, max_diff) = compare_bf16(&actual, &fixture.expected_bf16, max_ulp);
    eprintln!(
        "  {mismatches} mismatches, max ULP diff = {max_diff}, anamnesis = {:.1} µs",
        elapsed.as_secs_f64() * 1e6
    );
    assert_eq!(
        mismatches, 0,
        "{name}: {mismatches}/{total} elements differ by more than {max_ulp} ULP"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn cross_validate_llama_3_2_1b_awq() {
    run_awq_cross_validation(
        "Llama-3.2-1B AWQ INT4",
        include_bytes!("fixtures/awq_reference/llama_3_2_1b_awq.bin"),
        0,
    );
}

#[test]
fn cross_validate_falcon3_1b_awq() {
    run_awq_cross_validation(
        "Falcon3-1B AWQ INT4",
        include_bytes!("fixtures/awq_reference/falcon3_1b_awq.bin"),
        0,
    );
}
