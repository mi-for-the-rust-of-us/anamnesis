// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `AMNB` v2 `BnB` fixture container, shared by the decode suite
//! (`cross_validation_bnb.rs`) and the encode suite
//! (`cross_validation_bnb_encode.rs`). Both read the same `.bin` files from
//! `tests/fixtures/bnb_reference/`.

use super::fixture::read_u32_le;

/// Magic prefix identifying a v2 `BnB` fixture container.
pub const FIXTURE_MAGIC: &[u8; 4] = b"AMNB";

/// Container version this reader understands.
pub const FIXTURE_VERSION: u32 = 2;

/// Asserts the v2 container prefix shared by both `BnB` fixture layouts.
///
/// v1 carried neither magic nor version, so this is what lets a stale checkout
/// fail loudly rather than read the header at the wrong offsets.
pub fn check_container(data: &[u8]) {
    assert_eq!(
        &data[..4],
        FIXTURE_MAGIC,
        "fixture is not a v2 `AMNB` container — regenerate with \
         tests/fixtures/bnb_reference/generate_bnb.py"
    );
    let version = read_u32_le(data, 4);
    assert_eq!(
        version, FIXTURE_VERSION,
        "unsupported fixture container version {version} (this reader understands \
         {FIXTURE_VERSION})"
    );
}

/// Parsed `NF4`/`FP4` fixture (`format_id` = 0 or 2).
///
/// Both goldens come from the canonical `bitsandbytes` `CUDA` kernel. The `F32`
/// one is a **separate** `dequantize_4bit` at `QuantState(dtype=float32)`, not
/// a widening of the stock-dtype result: the kernel rounds at the output store,
/// so widening a `bf16`/`f16` result yields a different number on 38-91 % of
/// elements (measured per fixture by `generate_bnb.py`).
pub struct Bnb4Fixture {
    pub format_id: u32,
    pub total_elements: usize,
    pub block_size: usize,
    /// Double-quant absmax offset from the `quant_state` JSON blob
    /// (`nested_offset`); `0.0` for plain (non-double-quant) fixtures.
    pub nested_offset: f32,
    pub weight_data: Vec<u8>,
    pub absmax_data: Vec<u8>,
    pub quant_map_data: Vec<u8>,
    /// Empty for `format_id == 0` (plain `NF4`/`FP4`); populated for
    /// `format_id == 2` (double-quant).
    pub nested_absmax_data: Vec<u8>,
    /// Empty for plain; 1024 bytes (256 x `F32`) for double-quant.
    pub nested_quant_map_data: Vec<u8>,
    pub expected_bf16: Vec<u8>,
    pub expected_f32: Vec<u8>,
}

/// Parsed `INT8` fixture (`format_id` = 1).
///
/// `int8_vectorwise_dequant` returns `f32`, so the `F32` golden is its result
/// before the `BF16` narrowing, no second call needed.
pub struct BnbInt8Fixture {
    pub out_features: usize,
    pub in_features: usize,
    pub weight_data: Vec<u8>,
    pub scb_data: Vec<u8>,
    pub expected_bf16: Vec<u8>,
    pub expected_f32: Vec<u8>,
}

/// Parses an `NF4`/`FP4` (plain or double-quant) fixture.
pub fn parse_bnb4_fixture(data: &[u8]) -> Bnb4Fixture {
    check_container(data);
    let format_id = read_u32_le(data, 8);
    let total_elements = read_u32_le(data, 12) as usize;
    let block_size = read_u32_le(data, 16) as usize;
    let weight_len = read_u32_le(data, 20) as usize;
    let absmax_len = read_u32_le(data, 24) as usize;
    let quant_map_len = read_u32_le(data, 28) as usize;
    let nested_absmax_len = read_u32_le(data, 32) as usize;
    let nested_quant_map_len = read_u32_le(data, 36) as usize;
    let expected_len = read_u32_le(data, 40) as usize;
    let f32_len = read_u32_le(data, 44) as usize;
    let nested_offset = f32::from_le_bytes(data[48..52].try_into().unwrap());

    let header_size = 52;
    let mut offset = header_size;

    let weight_data = data[offset..offset + weight_len].to_vec();
    offset += weight_len;
    let absmax_data = data[offset..offset + absmax_len].to_vec();
    offset += absmax_len;
    let quant_map_data = data[offset..offset + quant_map_len].to_vec();
    offset += quant_map_len;
    let nested_absmax_data = data[offset..offset + nested_absmax_len].to_vec();
    offset += nested_absmax_len;
    let nested_quant_map_data = data[offset..offset + nested_quant_map_len].to_vec();
    offset += nested_quant_map_len;
    let expected_bf16 = data[offset..offset + expected_len].to_vec();
    offset += expected_len;
    let expected_f32 = data[offset..offset + f32_len].to_vec();

    assert_eq!(expected_len, total_elements * 2, "BF16 golden length");
    assert_eq!(f32_len, total_elements * 4, "F32 golden length");

    Bnb4Fixture {
        format_id,
        total_elements,
        block_size,
        nested_offset,
        weight_data,
        absmax_data,
        quant_map_data,
        nested_absmax_data,
        nested_quant_map_data,
        expected_bf16,
        expected_f32,
    }
}

/// Parses an `INT8` fixture.
pub fn parse_int8_fixture(data: &[u8]) -> BnbInt8Fixture {
    check_container(data);
    let _format_id = read_u32_le(data, 8); // = 1
    let out_features = read_u32_le(data, 12) as usize;
    let in_features = read_u32_le(data, 16) as usize;
    let weight_len = read_u32_le(data, 20) as usize;
    let scb_len = read_u32_le(data, 24) as usize;
    let expected_len = read_u32_le(data, 28) as usize;
    let f32_len = read_u32_le(data, 32) as usize;

    let header_size = 36;
    let mut offset = header_size;

    let weight_data = data[offset..offset + weight_len].to_vec();
    offset += weight_len;
    let scb_data = data[offset..offset + scb_len].to_vec();
    offset += scb_len;
    let expected_bf16 = data[offset..offset + expected_len].to_vec();
    offset += expected_len;
    let expected_f32 = data[offset..offset + f32_len].to_vec();

    assert_eq!(
        expected_len,
        out_features * in_features * 2,
        "BF16 golden length"
    );
    assert_eq!(f32_len, out_features * in_features * 4, "F32 golden length");

    BnbInt8Fixture {
        out_features,
        in_features,
        weight_data,
        scb_data,
        expected_bf16,
        expected_f32,
    }
}
