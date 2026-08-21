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

use anamnesis::{Dtype, F32Out, dequantize_awq, dequantize_awq_to_bf16};

// ---------------------------------------------------------------------------
// Fixture parsing
// ---------------------------------------------------------------------------

/// Magic prefix identifying a v2 `AWQ` fixture container.
const FIXTURE_MAGIC: &[u8; 4] = b"AMNA";

/// Container version this reader understands.
const FIXTURE_VERSION: u32 = 2;

/// Binary fixture layout (all little-endian):
///
/// - 4 bytes: magic `AMNA`
/// - 4 bytes: container version (`u32`, currently 2)
/// - 4 bytes: `bits` (`u32`)
/// - 4 bytes: `group_size` (`u32`)
/// - 4 bytes: `in_features` (`u32`)
/// - 4 bytes: `out_features` (`u32`)
/// - 4 bytes: `scale_dtype` (0=`F32`, 1=`BF16`, 2=`F16`)
/// - 4 bytes: `qweight_len` (`u32`)
/// - 4 bytes: `scales_len` (`u32`)
/// - 4 bytes: `qzeros_len` (`u32`)
/// - 4 bytes: golden `BF16` byte count (`u32`)
/// - 4 bytes: golden `F32` byte count (`u32`)
/// - qweight, scales, qzeros, expected `BF16`, expected `F32`
///
/// v1 carried neither magic nor version and stopped after the `BF16` golden, so
/// the magic is what lets a stale checkout fail loudly here rather than read the
/// header at the wrong offsets.
///
/// **Both goldens come from `AutoAWQ`**: the `F32` one is taken from its
/// `unpack_awq` + `reverse_awq_order` result *before* the `BF16` narrowing,
/// never by widening the `BF16` back, which would make the comparison circular
/// with anamnesis' own rounding.
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
    expected_f32: Vec<u8>,
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
    u32::from_le_bytes(bytes)
}

fn parse_awq_fixture(data: &[u8]) -> AwqFixture {
    assert_eq!(
        &data[..4],
        FIXTURE_MAGIC,
        "fixture is not a v2 `AMNA` container — regenerate with \
         tests/fixtures/awq_reference/generate_awq.py"
    );
    let version = read_u32_le(data, 4);
    assert_eq!(
        version, FIXTURE_VERSION,
        "unsupported fixture container version {version} (this reader understands \
         {FIXTURE_VERSION})"
    );

    let bits = read_u32_le(data, 8) as u8;
    let group_size = read_u32_le(data, 12) as usize;
    let in_features = read_u32_le(data, 16) as usize;
    let out_features = read_u32_le(data, 20) as usize;
    let scale_dtype_id = read_u32_le(data, 24);
    let qweight_len = read_u32_le(data, 28) as usize;
    let scales_len = read_u32_le(data, 32) as usize;
    let qzeros_len = read_u32_le(data, 36) as usize;
    let expected_len = read_u32_le(data, 40) as usize;
    let f32_len = read_u32_le(data, 44) as usize;

    let header_size = 48;
    let qw_start = header_size;
    let sc_start = qw_start + qweight_len;
    let qz_start = sc_start + scales_len;
    let ex_start = qz_start + qzeros_len;
    let f32_start = ex_start + expected_len;

    assert_eq!(
        expected_len,
        in_features * out_features * 2,
        "BF16 golden length"
    );
    assert_eq!(f32_len, in_features * out_features * 4, "F32 golden length");

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
        expected_f32: data[f32_start..f32_start + f32_len].to_vec(),
    }
}

/// Compares `AWQ` `F32` output against `AutoAWQ` **bit for bit**.
///
/// No tolerance, by design. The `BF16` comparison carries a `ULP` budget
/// because narrowing to 8 significand bits is a lossy step both sides perform.
/// `F32` has no such excuse: both compute in `f32`, so identical operations in
/// identical order must give identical bits. A difference here is a real
/// disagreement about the arithmetic — the `(qw - qz) × scale` association, the
/// `AWQ_REVERSE_ORDER` nibble permutation, a lost widening, a contraction — and
/// ~82 % of these fixtures' values carry mantissa bits `BF16` could not show.
fn compare_awq_f32_exact(name: &str, fx: &AwqFixture) {
    let actual = dequantize_awq::<F32Out>(
        &fx.qweight_data,
        &fx.scales_data,
        &fx.qzeros_data,
        fx.in_features,
        fx.out_features,
        fx.group_size,
        fx.bits,
        fx.scale_dtype,
    )
    .expect("AWQ F32 dequant failed");

    assert_eq!(
        actual.len(),
        fx.expected_f32.len(),
        "{name}: F32 output length mismatch"
    );

    let mut mismatches = 0usize;
    let mut first: Option<(usize, f32, f32)> = None;
    for (i, (a_word, e_word)) in actual
        .as_chunks::<4>()
        .0
        .iter()
        .zip(fx.expected_f32.as_chunks::<4>().0)
        .enumerate()
    {
        let a = f32::from_le_bytes([a_word[0], a_word[1], a_word[2], a_word[3]]);
        let e = f32::from_le_bytes([e_word[0], e_word[1], e_word[2], e_word[3]]);
        // Bit equality, not `==`: `NaN` payloads and signed zero must count.
        if a.to_bits() != e.to_bits() {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, a, e));
            }
        }
    }

    if let Some((i, a, e)) = first {
        eprintln!(
            "  F32 element {i}: actual={a:e} (0x{:08X}), expected={e:e} (0x{:08X})",
            a.to_bits(),
            e.to_bits()
        );
    }
    assert_eq!(
        mismatches,
        0,
        "{name}: {mismatches}/{} F32 elements differ from AutoAWQ. This is NOT a \
         tolerance question — both sides compute in f32. Treat it as a real \
         finding before touching the test.",
        actual.len() / 4
    );
}

// ---------------------------------------------------------------------------
// BF16 comparison
// ---------------------------------------------------------------------------

fn compare_bf16(actual: &[u8], expected: &[u8], max_ulp_diff: u16) -> (usize, u16) {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");
    let mut mismatches = 0;
    let mut max_diff: u16 = 0;

    for (i, (a_pair, e_pair)) in actual
        .as_chunks::<2>()
        .0
        .iter()
        .zip(expected.as_chunks::<2>().0)
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

    compare_awq_f32_exact(name, &fixture);
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
