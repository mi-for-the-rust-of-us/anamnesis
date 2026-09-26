// SPDX-License-Identifier: MIT OR Apache-2.0

//! Little-endian readers for the checked-in binary fixture containers.

/// Reads a little-endian `u32` at `offset`.
///
/// Panics if `data` is shorter than `offset + 4`, which for a checked-in
/// fixture means the file is truncated or the header layout is misread.
pub fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    // INDEX: callers pass offsets bounded by the fixture header they are
    // decoding; fixture data is checked in, so an out-of-range read is a
    // test-side assertion, not an attacker-controllable surface.
    let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap();
    u32::from_le_bytes(bytes)
}
