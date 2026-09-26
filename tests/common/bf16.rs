// SPDX-License-Identifier: MIT OR Apache-2.0

//! `BF16` golden comparison and synthesis.

/// How [`compare_bf16_with`] treats `+0` against `-0`.
#[derive(Clone, Copy, Debug)]
pub enum SignedZero {
    /// `+0` and `-0` are different bit patterns and count as a mismatch
    /// (`0x8000` apart). The default, used by every suite except `BnB`.
    Distinct,
    /// `+0` and `-0` count as a match. IEEE 754 makes them arithmetically
    /// identical; `cross_validation_bnb.rs` documents why its kernels need it.
    Equivalent,
}

/// Compare two `BF16` byte slices, allowing up to `max_ulp_diff` `ULP`
/// (unit in the last place) difference per element.
///
/// `NaN` counts as a match when both sides are `NaN`, and `+0` / `-0` are
/// distinct. Returns the number of mismatched elements and the maximum `ULP`
/// diff found.
pub fn compare_bf16(actual: &[u8], expected: &[u8], max_ulp_diff: u16) -> (usize, u16) {
    compare_bf16_with(actual, expected, max_ulp_diff, SignedZero::Distinct)
}

/// [`compare_bf16`] with an explicit rule for signed zero.
///
/// Returns the number of mismatched elements and the maximum `ULP` diff found.
pub fn compare_bf16_with(
    actual: &[u8],
    expected: &[u8],
    max_ulp_diff: u16,
    signed_zero: SignedZero,
) -> (usize, u16) {
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
        // INDEX: as_chunks::<2>() guarantees exactly 2 bytes per pair
        let a_bits = u16::from_le_bytes([a_pair[0], a_pair[1]]);
        let e_bits = u16::from_le_bytes([e_pair[0], e_pair[1]]);

        // Handle NaN: both NaN is a match.
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

        // BITWISE: low 15 bits zero means the value is +0 or -0 (sign-only diff).
        if matches!(signed_zero, SignedZero::Equivalent)
            && a_bits.trailing_zeros() >= 15
            && e_bits.trailing_zeros() >= 15
        {
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

/// Encodes `f32` values as little-endian `BF16` bytes by truncation.
pub fn bf16_bytes_from_f32_iter<I: IntoIterator<Item = f32>>(values: I) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        // CAST: u32 → u16, the shift leaves only the upper 16 bits.
        // BITWISE: BF16 = upper 16 bits of f32 (truncate).
        let bits = (v.to_bits() >> 16) as u16;
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}
