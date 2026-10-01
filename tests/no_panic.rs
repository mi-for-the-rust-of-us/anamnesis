// SPDX-License-Identifier: MIT OR Apache-2.0

//! Panic-freedom invariant (Phase 6.13 Step 3): **no public parse/inspect entry
//! point may panic or abort on any input** — a malformed or hostile artefact is
//! always a clean `Ok`/`Err`, never an unwinding panic (which under the shipped
//! `panic = "abort"` profile would be an uncatchable process kill, and which the
//! Phase 8 bindings must be able to surface as a catchable `PanicException`).
//!
//! Every re-exported parse/inspect entry point of all four formats is driven
//! here — the owned-bytes, reader, **and path/mmap** variants, plus (since
//! v0.7.6) the budget-taking `inspect_*_with_options` forms and the
//! format-agnostic `detect_format_from_bytes*` / `convert_bytes` — each under
//! both `ParseLimits::default()` and a deliberately hostile tight budget (so the
//! `check_alloc` / bounded-reader / `Budget::charge_alloc` / count / ratio
//! rejection arithmetic is on the hook too) — over a battery of adversarial
//! inputs (synthetic malformed shapes + truncations and bit-flips of the
//! committed fixtures), each call wrapped in [`std::panic::catch_unwind`] and
//! asserted never to unwind. The suite runs under the default (debug) test
//! profile, so debug-only integer-overflow panics are in scope.
//!
//! Phase 7.10 extended the battery to what it added: crafted `GGUF` files of
//! `ggml_type` 40, 41 and 42 (with every shape of `NVFP4` per-tensor `.scale`
//! the fold applies or must leave alone), crafted and real `ModelOpt` `NVFP4`
//! safetensors, and the new public kernels called directly with lengths and
//! shapes that disagree.
//!
//! Two boundaries are deliberate, not omissions:
//! - **`SIGBUS` is out of scope by nature.** The `parse*` path/mmap variants map
//!   the file; a *post-map* truncation faults with `SIGBUS` — an OS signal
//!   `catch_unwind` cannot catch. That hazard is addressed by *recommending the
//!   copy-based `parse_bytes` / `parse_*_from_reader` paths* for untrusted input
//!   (Step 1), not by this test. Mapping a complete file exercises the identical
//!   `parsed_*_from_backing` parse core the owned paths use.
//! - **Exhaustive structural exploration is the fuzzer's job.** This test pins
//!   the entry-point surface against a fixed corpus in stable CI; the coverage-
//!   guided `cargo fuzz` harness (incl. the owned-bytes `fuzz_*_bytes` targets)
//!   drives deep-nesting / rare-opcode discovery the fixed battery cannot.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::wildcard_enum_match_arm
)]

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use anamnesis::ParseLimits;

/// Runs `f` and asserts it returns normally (`Ok`/`Err`) rather than unwinding.
/// The produced value is irrelevant — only "did it panic?" matters.
fn assert_no_panic<T>(label: &str, f: impl FnOnce() -> T) {
    let outcome = catch_unwind(AssertUnwindSafe(f));
    assert!(outcome.is_ok(), "PANICKED on input `{label}`");
}

/// A deliberately hostile budget (16-byte single-allocation ceiling): forces
/// every parser onto its `ParseLimits` rejection branches, so the checked
/// arithmetic that backs them must reject without panicking.
fn tight() -> ParseLimits {
    ParseLimits::default().with_max_single_alloc(16)
}

/// Writes `bytes` to `path` for the path/mmap entry points. A failure here is a
/// fault in the *test harness*, not the code under test, so it may panic.
fn stage(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).expect("test harness: write temp input");
}

/// Committed (always-present) fixtures, truncated and bit-flipped below to build
/// realistic "almost-valid" adversarial inputs.
const FIXTURES: &[&str] = &[
    "tests/fixtures/safetensors_reference/fp8.safetensors",
    "tests/fixtures/safetensors_reference/gptq.safetensors",
    "tests/fixtures/safetensors_reference/awq.safetensors",
    "tests/fixtures/safetensors_reference/bnb_nf4.safetensors",
    "tests/fixtures/pth_reference/algzoo_rnn_small.pth",
    "tests/fixtures/npz_reference/gemma_scope_small.npz",
    // Crash bodies from the v0.7.8 fuzz campaign (`tests/fuzz_regressions.rs`):
    // their neighbours are exactly the "almost valid" inputs this battery wants.
    "tests/fixtures/fuzz_regressions/pth_duplicate_state_dict_key.pth",
    "tests/fixtures/fuzz_regressions/gptq_bits_zero.safetensors",
    // A real NVIDIA `ModelOpt` `NVFP4` layer (Phase 7.10): recognised in every
    // build, dequantised under `nvfp4`, refused by name without it.
    "tests/fixtures/nvfp4_reference/llama31_8b_nvfp4_q_proj.safetensors",
];

/// Pushes `bytes` with the near-misses every crafted input gets: cut in half,
/// one byte flipped in the middle, and whole.
fn push_with_near_misses(inputs: &mut Vec<(String, Vec<u8>)>, label: &str, bytes: Vec<u8>) {
    let len = bytes.len();
    inputs.push((
        format!("{label}@trunc{}", len / 2),
        bytes[..len / 2].to_vec(),
    ));
    let mut flipped = bytes.clone();
    if let Some(byte) = flipped.get_mut(len / 2) {
        *byte ^= 0xFF;
    }
    inputs.push((format!("{label}@flip"), flipped));
    inputs.push((format!("{label}@whole"), bytes));
}

/// `n` bytes of a fixed pattern that reaches every nibble and scale-byte value.
fn patterned(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(37).wrapping_add(11)) as u8)
        .collect()
}

/// Little-endian `f32` bytes.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A `GGUF` file of `(name, dims, ggml_type, data)` tensors, each data range
/// placed at the next 32-byte boundary. No validation: these are hostile.
#[cfg(feature = "gguf")]
fn crafted_gguf(tensors: &[(&str, &[u64], u32, Vec<u8>)]) -> Vec<u8> {
    use common::gguf::{RawTensorInfo, raw_gguf};
    let mut data = Vec::new();
    let mut infos = Vec::new();
    for (name, dims, ggml_type, bytes) in tensors {
        infos.push(RawTensorInfo {
            name,
            dims,
            ggml_type: *ggml_type,
            offset: data.len() as u64,
        });
        data.extend_from_slice(bytes);
        data.resize(data.len().next_multiple_of(32), 0);
    }
    raw_gguf(&[], &infos, 32, &data)
}

/// `ggml_type` discriminants the Phase 7.10 inputs use.
#[cfg(feature = "gguf")]
mod ggml {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const NVFP4: u32 = 40;
    pub const Q1_0: u32 = 41;
    pub const Q2_0: u32 = 42;
}

/// Label prefix of the crafted `GGUF` inputs for `ggml_type` 40, 41 and 42.
#[cfg(feature = "gguf")]
const GGUF_NEW_TYPES: &str = "gguf-t";

/// Crafted `GGUF` files holding the `ggml_type`s added in Phase 7.10, well
/// formed and not: whole blocks, a partial block, data cut short, and every
/// shape of `NVFP4` per-tensor `.scale` the fold either applies or must leave
/// alone, including values that are not finite and dimensions whose products
/// overflow.
#[cfg(feature = "gguf")]
fn new_gguf_type_inputs() -> Vec<(String, Vec<u8>)> {
    use ggml::{F16, F32, NVFP4, Q1_0, Q2_0};
    const W: &str = "blk.0.attn_q.weight";
    const S: &str = "blk.0.attn_q.scale";
    // NVFP4 and Q2_0 hold 64 values per block, Q1_0 128; 36, 18 and 18 bytes.
    let nvfp4 = |blocks: usize| patterned(36 * blocks);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("40", crafted_gguf(&[(W, &[64, 2], NVFP4, nvfp4(2))])),
        ("41", crafted_gguf(&[(W, &[128, 2], Q1_0, patterned(36))])),
        ("42", crafted_gguf(&[(W, &[64, 2], Q2_0, patterned(36))])),
        // All bits set: the largest nibble everywhere, and UE4M3 `0xFF`, which
        // ggml decodes to 240 rather than NaN.
        (
            "40-ones",
            crafted_gguf(&[(W, &[64], NVFP4, vec![0xFF; 36])]),
        ),
        (
            "41-ones",
            crafted_gguf(&[(W, &[128], Q1_0, vec![0xFF; 18])]),
        ),
        ("42-ones", crafted_gguf(&[(W, &[64], Q2_0, vec![0xFF; 18])])),
        // Half a Q1_0 block: issue #15's own reproducer shape.
        (
            "41-half-block",
            crafted_gguf(&[(W, &[64], Q1_0, patterned(18))]),
        ),
        // Four blocks declared, one present.
        (
            "40-short-data",
            crafted_gguf(&[(W, &[64, 4], NVFP4, nvfp4(1))]),
        ),
        (
            "40-zero-dim",
            crafted_gguf(&[(W, &[64, 0], NVFP4, Vec::new())]),
        ),
        // The fold's slab arithmetic over dimensions whose product overflows.
        (
            "40-huge-dims+scale",
            crafted_gguf(&[
                (W, &[64, 1 << 40, 1 << 30], NVFP4, nvfp4(1)),
                (S, &[1 << 30], F32, f32_bytes(&[1.0])),
            ]),
        ),
        (
            "40+scale",
            crafted_gguf(&[
                (W, &[64, 2], NVFP4, nvfp4(2)),
                (S, &[1], F32, f32_bytes(&[3.0e-4])),
            ]),
        ),
        (
            "40+scale-nan",
            crafted_gguf(&[
                (W, &[64, 2], NVFP4, nvfp4(2)),
                (S, &[1], F32, f32_bytes(&[f32::NAN])),
            ]),
        ),
        (
            "40+scale-inf",
            crafted_gguf(&[
                (W, &[64, 2], NVFP4, nvfp4(2)),
                (S, &[1], F32, f32_bytes(&[f32::INFINITY])),
            ]),
        ),
        // Three experts, one scale each; then a count that matches no axis.
        (
            "40+per-expert-scale",
            crafted_gguf(&[
                (W, &[64, 2, 3], NVFP4, nvfp4(6)),
                (S, &[3], F32, f32_bytes(&[1.0, 0.5, 2.0])),
            ]),
        ),
        (
            "40+scale-wrong-count",
            crafted_gguf(&[
                (W, &[64, 2, 3], NVFP4, nvfp4(6)),
                (S, &[2], F32, f32_bytes(&[1.0, 0.5])),
            ]),
        ),
        (
            "40+scale-f16",
            crafted_gguf(&[
                (W, &[64, 2], NVFP4, nvfp4(2)),
                (S, &[1], F16, vec![0x00, 0x3C]),
            ]),
        ),
        (
            "40+scale-empty",
            crafted_gguf(&[(W, &[64, 2], NVFP4, nvfp4(2)), (S, &[0], F32, Vec::new())]),
        ),
        // A `.scale` beside a weight that is not NVFP4.
        (
            "42+scale",
            crafted_gguf(&[
                (W, &[64, 2], Q2_0, patterned(36)),
                (S, &[1], F32, f32_bytes(&[0.5])),
            ]),
        ),
    ];
    cases
        .into_iter()
        .map(|(label, bytes)| (format!("{GGUF_NEW_TYPES}{label}"), bytes))
        .collect()
}

/// Label prefix of the crafted `ModelOpt` `NVFP4` inputs.
const MODELOPT: &str = "modelopt-nvfp4";

/// One `ModelOpt` tensor for [`modelopt_inputs`]: `(dtype, shape, bytes)`.
type Part<'a> = (&'a str, &'a [usize], &'a [u8]);

/// Crafted `ModelOpt` `NVFP4` layers (`U8` weight `[rows, cols / 2]`,
/// `F8_E4M3` `weight_scale` `[rows, cols / 16]`, `F32` `weight_scale_2`), well
/// formed and not: every companion with the wrong shape, dtype or count, a
/// column count that is no whole block, and scales that are not finite.
fn modelopt_inputs() -> Vec<(String, Vec<u8>)> {
    use common::builders::build_safetensors_raw;
    const W: &str = "mlp.down_proj.weight";
    const S1: &str = "mlp.down_proj.weight_scale";
    const S2: &str = "mlp.down_proj.weight_scale_2";
    let weight = patterned(128);
    let scales = vec![0x38u8; 16];
    let global = f32_bytes(&[0.01]);
    let layer = |w: Part<'_>, s1: Part<'_>, s2: Part<'_>| {
        build_safetensors_raw(&[
            (W, w.0, w.1, w.2),
            (S1, s1.0, s1.1, s1.2),
            (S2, s2.0, s2.1, s2.2),
            ("mlp.down_proj.input_scale", "F32", &[], &global),
        ])
    };
    let ok_w: Part<'_> = ("U8", &[4, 32], &weight);
    let ok_s1: Part<'_> = ("F8_E4M3", &[4, 4], &scales);
    let ok_s2: Part<'_> = ("F32", &[], &global);
    let two = f32_bytes(&[0.01, 0.02]);
    let nan = f32_bytes(&[f32::NAN]);
    let s1_f32 = f32_bytes(&[1.0; 16]);
    let nan_scales = vec![0xFFu8; 16]; // `float8_e4m3fn` NaN
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("ok", layer(ok_w, ok_s1, ok_s2)),
        ("scale2-two-values", layer(ok_w, ok_s1, ("F32", &[2], &two))),
        (
            "scale2-bf16",
            layer(ok_w, ok_s1, ("BF16", &[], &[0x80, 0x3F])),
        ),
        ("scale2-nan", layer(ok_w, ok_s1, ("F32", &[], &nan))),
        ("scale1-f32", layer(ok_w, ("F32", &[4, 4], &s1_f32), ok_s2)),
        (
            "scale1-nan",
            layer(ok_w, ("F8_E4M3", &[4, 4], &nan_scales), ok_s2),
        ),
        (
            "scale1-wrong-shape",
            layer(ok_w, ("F8_E4M3", &[4, 3], &scales[..12]), ok_s2),
        ),
        // Shape and byte count disagree.
        (
            "scale1-short-data",
            layer(ok_w, ("F8_E4M3", &[4, 4], &scales[..8]), ok_s2),
        ),
        // 62 values per row: not a whole 16-value block.
        (
            "odd-cols",
            layer(("U8", &[4, 31], &weight[..124]), ok_s1, ok_s2),
        ),
        (
            "one-dim",
            layer(("U8", &[128], &weight), ("F8_E4M3", &[16], &scales), ok_s2),
        ),
        (
            "three-dim",
            layer(
                ("U8", &[2, 2, 32], &weight),
                ("F8_E4M3", &[2, 2, 4], &scales),
                ok_s2,
            ),
        ),
        (
            "zero-rows",
            layer(("U8", &[0, 32], &[]), ("F8_E4M3", &[0, 4], &[]), ok_s2),
        ),
        (
            "no-scale2",
            build_safetensors_raw(&[
                (W, "U8", &[4, 32], &weight),
                (S1, "F8_E4M3", &[4, 4], &scales),
            ]),
        ),
    ];
    cases
        .into_iter()
        .map(|(label, bytes)| (format!("{MODELOPT}-{label}"), bytes))
        .collect()
}

/// The adversarial input battery: `(label, bytes)`.
fn adversarial_inputs() -> Vec<(String, Vec<u8>)> {
    let mut inputs: Vec<(String, Vec<u8>)> = vec![
        ("empty".to_owned(), Vec::new()),
        ("one-zero".to_owned(), vec![0u8]),
        ("zeros-64".to_owned(), vec![0u8; 64]),
        ("ones-64".to_owned(), vec![0xFFu8; 64]),
        // 8 bytes of 0xFF: a safetensors `u64` header length of `u64::MAX`.
        ("u64-max-prefix".to_owned(), vec![0xFFu8; 8]),
        ("u64-max-prefix+junk".to_owned(), {
            let mut b = vec![0xFFu8; 8];
            b.extend_from_slice(b"junk-after-an-absurd-length");
            b
        }),
        // Small declared length + a partial JSON header.
        ("small-len+partial-json".to_owned(), {
            let mut b = 2u64.to_le_bytes().to_vec();
            b.extend_from_slice(b"{");
            b
        }),
        // ZIP local-file magic + junk (drives the `.npz` / `.pth` container).
        ("zip-magic+junk".to_owned(), {
            let mut b = b"PK\x03\x04".to_vec();
            b.extend_from_slice(&[0xABu8; 64]);
            b
        }),
        // GGUF magic + junk.
        ("gguf-magic+junk".to_owned(), {
            let mut b = b"GGUF".to_vec();
            b.extend_from_slice(&[0xCDu8; 64]);
            b
        }),
        // Raw pickle protocol bytes (legacy `.pth` detection path), then a MARK
        // flood — pushes the pickle VM toward its nesting/stack guards.
        ("pickle-proto".to_owned(), vec![0x80, 0x02, b'}', b'.']),
        ("pickle-proto+mark-flood".to_owned(), {
            let mut b = vec![0x80u8, 0x02];
            b.extend_from_slice(&[0x28u8; 8192]); // `(` = MARK
            b
        }),
        // Deeply repetitive bytes — stress recursion / nesting guards.
        ("open-brackets-8k".to_owned(), vec![b'['; 8192]),
        ("repeating-2byte-100k".to_owned(), {
            b"\x80\x02".iter().copied().cycle().take(100_000).collect()
        }),
    ];

    // A couple of deterministic pseudo-random blobs (no RNG dependency).
    for seed in [0x9E37_79B9u32, 0x1234_5678u32] {
        let blob: Vec<u8> = (0u32..512)
            .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 13) as u8)
            .collect();
        inputs.push((format!("prng-{seed:08x}"), blob));
    }

    // Truncations and single-byte flips of each committed fixture: "almost
    // valid" inputs are the ones most likely to slip past a length check into a
    // downstream panic.
    for path in FIXTURES {
        let Ok(bytes) = std::fs::read(path) else {
            continue;
        };
        let len = bytes.len();
        let name = fixture_basename(path);
        // The intact file too: every parse succeeds on the reference fixtures,
        // which is what lets `methods_on_parsed_results_never_panic` reach the
        // methods a binding calls on a result.
        inputs.push((format!("{name}@whole"), bytes.clone()));
        for cut in [1usize, 4, 8, 16, len / 4, len / 2, len.saturating_sub(1)] {
            let cut = cut.min(len);
            inputs.push((format!("{name}@trunc{cut}"), bytes[..cut].to_vec()));
        }
        for pos in [0usize, len / 2, len.saturating_sub(1)] {
            if pos < len {
                let mut flipped = bytes.clone();
                flipped[pos] ^= 0xFF;
                inputs.push((format!("{name}@flip{pos}"), flipped));
            }
        }
    }

    // Crafted hostile `.pth` files from the Phase 7.9 audit, with their
    // near-misses: before the fix, the first one aborted the process on any call
    // that materialised it, which `catch_unwind` cannot observe.
    #[cfg(feature = "pth")]
    for (label, bytes) in [
        (
            "pth-expanded-2^62",
            common::pth::single_u8_view(&[1 << 62, 3], &[0, 0], &[7]),
        ),
        (
            "pth-expanded-2^30",
            common::pth::single_u8_view(&[1 << 30], &[0], &[7]),
        ),
        (
            "pth-position-ids",
            common::pth::single_u8_view(&[1, 4], &[0, 1], &[1, 2, 3, 4]),
        ),
    ] {
        let len = bytes.len();
        for cut in [len / 2, len.saturating_sub(1)] {
            inputs.push((format!("{label}@trunc{cut}"), bytes[..cut].to_vec()));
        }
        let mut flipped = bytes.clone();
        flipped[len / 2] ^= 0xFF;
        inputs.push((format!("{label}@flip"), flipped));
        inputs.push((format!("{label}@whole"), bytes));
    }

    // Crafted hostile `GGUF` files from the Phase 7.9 audit: tensors aliasing
    // one data range, and an alignment that is not a power of two.
    #[cfg(feature = "gguf")]
    {
        use common::gguf::{GGUF_KV_U32, RawTensorInfo, raw_gguf};
        let dims = [4u64];
        let aliased: Vec<RawTensorInfo<'_>> = ["a", "b", "c"]
            .into_iter()
            .map(|name| RawTensorInfo {
                name,
                dims: &dims,
                ggml_type: 0,
                offset: 0,
            })
            .collect();
        let odd = 48u32.to_le_bytes();
        for (label, bytes) in [
            ("gguf-aliased", raw_gguf(&[], &aliased, 32, &[0u8; 16])),
            (
                "gguf-alignment-48",
                raw_gguf(
                    &[("general.alignment", GGUF_KV_U32, &odd)],
                    &aliased[..1],
                    48,
                    &[0u8; 16],
                ),
            ),
        ] {
            let len = bytes.len();
            inputs.push((
                format!("{label}@trunc{}", len / 2),
                bytes[..len / 2].to_vec(),
            ));
            let mut flipped = bytes.clone();
            flipped[len / 2] ^= 0xFF;
            inputs.push((format!("{label}@flip"), flipped));
            inputs.push((format!("{label}@whole"), bytes));
        }
    }

    // Crafted inputs for the formats Phase 7.10 added.
    for (label, bytes) in modelopt_inputs() {
        push_with_near_misses(&mut inputs, &label, bytes);
    }
    #[cfg(feature = "gguf")]
    for (label, bytes) in new_gguf_type_inputs() {
        push_with_near_misses(&mut inputs, &label, bytes);
    }

    // No `.gguf` file is committed, so without this nothing in the battery
    // parses as `GGUF` and every method called on a parsed `GGUF` result goes
    // unexercised. Generate one from the FP8 fixture, with its near-misses.
    #[cfg(feature = "gguf")]
    if let Ok(st) = std::fs::read(FIXTURES[0]) {
        let options = anamnesis::ConvertOptions::new();
        if let Ok((gguf, _)) =
            anamnesis::convert_bytes(&st, anamnesis::ConvertTarget::Gguf, &options)
        {
            let len = gguf.len();
            for cut in [16usize, len / 2, len.saturating_sub(1)] {
                inputs.push((
                    format!("{GGUF_FROM_FP8}@trunc{cut}"),
                    gguf[..cut.min(len)].to_vec(),
                ));
            }
            let mut flipped = gguf.clone();
            flipped[len / 2] ^= 0xFF;
            inputs.push((format!("{GGUF_FROM_FP8}@flip"), flipped));
            inputs.push((format!("{GGUF_FROM_FP8}@whole"), gguf));
        }
    }

    inputs
}

/// Label prefix of the generated `GGUF` inputs.
#[cfg(feature = "gguf")]
const GGUF_FROM_FP8: &str = "fp8-converted.gguf";

/// Guards the generated `GGUF` inputs the same way `battery_includes_every_fixture`
/// guards the committed ones: if the conversion ever stops producing a file that
/// parses, the `GGUF` half of `methods_on_parsed_results_never_panic` would go
/// silent again.
#[cfg(feature = "gguf")]
#[test]
fn battery_includes_a_parseable_gguf() {
    let parsed = adversarial_inputs()
        .into_iter()
        .filter(|(label, _)| label.starts_with(GGUF_FROM_FP8))
        .filter(|(_, bytes)| anamnesis::parse_gguf_bytes(bytes.clone()).is_ok())
        .count();
    assert!(parsed >= 1, "no generated GGUF input parses");
}

/// The same guard for the Phase 7.10 inputs: the well-formed ones must parse,
/// or the methods pass (dequantisation, the `NVFP4` scale fold) never reaches
/// them.
#[test]
fn battery_includes_parseable_phase_7_10_inputs() {
    let inputs = adversarial_inputs();
    let modelopt = inputs
        .iter()
        .filter(|(label, _)| label.starts_with(MODELOPT) && label.ends_with("@whole"))
        .filter_map(|(_, bytes)| anamnesis::parse_bytes(bytes.clone()).ok())
        .filter(|model| model.header.scheme == anamnesis::QuantScheme::Nvfp4)
        .count();
    assert!(
        modelopt >= 2,
        "only {modelopt} crafted ModelOpt inputs parse as NVFP4"
    );
    #[cfg(feature = "gguf")]
    {
        let gguf = inputs
            .iter()
            .filter(|(label, _)| label.starts_with(GGUF_NEW_TYPES) && label.ends_with("@whole"))
            .filter(|(_, bytes)| anamnesis::parse_gguf_bytes(bytes.clone()).is_ok())
            .count();
        assert!(gguf >= 10, "only {gguf} crafted GGUF 40/41/42 inputs parse");
    }
}

/// The label prefix used for inputs derived from a fixture path.
fn fixture_basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Guards against a silently-stale `FIXTURES` path: if a fixture moves, its
/// truncation/flip inputs vanish and the battery weakens *invisibly* — so assert
/// every fixture actually contributed inputs, loudly.
#[test]
fn battery_includes_every_fixture() {
    let inputs = adversarial_inputs();
    for fixture in FIXTURES {
        let base = fixture_basename(fixture);
        assert!(
            inputs.iter().any(|(label, _)| label.starts_with(base)),
            "no inputs derived from `{fixture}` — a FIXTURES path is stale, \
             silently weakening coverage"
        );
    }
}

// ---------------------------------------------------------------------------
// safetensors (always-on)
// ---------------------------------------------------------------------------

#[test]
fn safetensors_entry_points_never_panic() {
    use anamnesis::{
        parse, parse_bytes, parse_bytes_with_limits, parse_from_reader,
        parse_from_reader_with_limits, parse_safetensors_header,
        parse_safetensors_header_from_reader, parse_safetensors_header_from_reader_with_limits,
        parse_safetensors_header_with_limits, parse_with_limits,
    };
    use std::io::Cursor;

    let tmp = tempfile::NamedTempFile::new().expect("temp file");
    let path = tmp.path();
    let tight = tight();

    for (label, bytes) in adversarial_inputs() {
        // owned bytes
        assert_no_panic(&format!("st parse_bytes / {label}"), || {
            parse_bytes(bytes.clone())
        });
        assert_no_panic(
            &format!("st parse_bytes_with_limits[tight] / {label}"),
            || parse_bytes_with_limits(bytes.clone(), &tight),
        );
        // reader
        assert_no_panic(&format!("st parse_from_reader / {label}"), || {
            parse_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(
            &format!("st parse_from_reader_with_limits[tight] / {label}"),
            || parse_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        // header — bytes + reader
        assert_no_panic(&format!("st parse_safetensors_header / {label}"), || {
            parse_safetensors_header(&bytes)
        });
        assert_no_panic(
            &format!("st parse_safetensors_header_with_limits[tight] / {label}"),
            || parse_safetensors_header_with_limits(&bytes, &tight),
        );
        assert_no_panic(&format!("st header_from_reader / {label}"), || {
            parse_safetensors_header_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(
            &format!("st header_from_reader_with_limits[tight] / {label}"),
            || parse_safetensors_header_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        // path / mmap
        stage(path, &bytes);
        assert_no_panic(&format!("st parse(path) / {label}"), || parse(path));
        assert_no_panic(
            &format!("st parse_with_limits(path)[tight] / {label}"),
            || parse_with_limits(path, &tight),
        );
    }
}

// ---------------------------------------------------------------------------
// GGUF
// ---------------------------------------------------------------------------

#[cfg(feature = "gguf")]
#[test]
fn gguf_entry_points_never_panic() {
    use anamnesis::{
        inspect_gguf_from_reader, parse_gguf, parse_gguf_bytes, parse_gguf_bytes_with_limits,
        parse_gguf_from_reader, parse_gguf_from_reader_with_limits,
        parse_gguf_front_matter_from_reader, parse_gguf_front_matter_from_reader_with_limits,
        parse_gguf_with_limits,
    };
    use std::io::Cursor;

    let tmp = tempfile::NamedTempFile::new().expect("temp file");
    let path = tmp.path();
    let tight = tight();

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("gguf parse_gguf_bytes / {label}"), || {
            parse_gguf_bytes(bytes.clone())
        });
        assert_no_panic(
            &format!("gguf parse_gguf_bytes_with_limits[tight] / {label}"),
            || parse_gguf_bytes_with_limits(bytes.clone(), &tight),
        );
        assert_no_panic(&format!("gguf parse_gguf_from_reader / {label}"), || {
            parse_gguf_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(
            &format!("gguf parse_gguf_from_reader_with_limits[tight] / {label}"),
            || parse_gguf_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        assert_no_panic(&format!("gguf inspect_from_reader / {label}"), || {
            inspect_gguf_from_reader(Cursor::new(bytes.clone()))
        });
        // The full-detail sibling of `inspect_gguf_from_reader` (v0.7.1): same
        // core, but every parsed string and array is handed back to the caller
        // rather than reduced to counts, so it gets its own coverage here.
        assert_no_panic(
            &format!("gguf parse_gguf_front_matter_from_reader / {label}"),
            || parse_gguf_front_matter_from_reader(Cursor::new(bytes.clone())),
        );
        assert_no_panic(
            &format!("gguf parse_gguf_front_matter_from_reader_with_limits[tight] / {label}"),
            || parse_gguf_front_matter_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        stage(path, &bytes);
        assert_no_panic(&format!("gguf parse_gguf(path) / {label}"), || {
            parse_gguf(path)
        });
        assert_no_panic(
            &format!("gguf parse_gguf_with_limits(path)[tight] / {label}"),
            || parse_gguf_with_limits(path, &tight),
        );
    }
}

// ---------------------------------------------------------------------------
// PyTorch .pth
// ---------------------------------------------------------------------------

#[cfg(feature = "pth")]
#[test]
fn pth_entry_points_never_panic() {
    use anamnesis::{
        inspect_pth_from_reader, parse_pth, parse_pth_bytes, parse_pth_bytes_with_limits,
        parse_pth_from_reader, parse_pth_from_reader_with_limits,
        parse_pth_front_matter_from_reader, parse_pth_front_matter_from_reader_with_limits,
        parse_pth_with_limits,
    };
    use std::io::Cursor;

    let tmp = tempfile::NamedTempFile::new().expect("temp file");
    let path = tmp.path();
    let tight = tight();

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("pth parse_pth_bytes / {label}"), || {
            parse_pth_bytes(bytes.clone())
        });
        assert_no_panic(
            &format!("pth parse_pth_bytes_with_limits[tight] / {label}"),
            || parse_pth_bytes_with_limits(bytes.clone(), &tight),
        );
        assert_no_panic(&format!("pth parse_pth_from_reader / {label}"), || {
            parse_pth_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(
            &format!("pth parse_pth_from_reader_with_limits[tight] / {label}"),
            || parse_pth_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        assert_no_panic(&format!("pth inspect_from_reader / {label}"), || {
            inspect_pth_from_reader(Cursor::new(bytes.clone()))
        });
        // The full-detail sibling of `inspect_pth_from_reader` (v0.7.5): same
        // core, but every parsed tensor name and shape is handed back to the
        // caller rather than reduced to counts, so it gets its own coverage
        // here.
        assert_no_panic(
            &format!("pth parse_pth_front_matter_from_reader / {label}"),
            || parse_pth_front_matter_from_reader(Cursor::new(bytes.clone())),
        );
        assert_no_panic(
            &format!("pth parse_pth_front_matter_from_reader_with_limits[tight] / {label}"),
            || parse_pth_front_matter_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );
        stage(path, &bytes);
        assert_no_panic(&format!("pth parse_pth(path) / {label}"), || {
            parse_pth(path)
        });
        assert_no_panic(
            &format!("pth parse_pth_with_limits(path)[tight] / {label}"),
            || parse_pth_with_limits(path, &tight),
        );
    }
}

// ---------------------------------------------------------------------------
// NPZ (path-based parse + reader/path inspect)
// ---------------------------------------------------------------------------

#[cfg(feature = "npz")]
#[test]
fn npz_entry_points_never_panic() {
    use anamnesis::{
        InspectOptions, inspect_npz, inspect_npz_from_reader, inspect_npz_from_reader_with_options,
        inspect_npz_with_options, parse_npz, parse_npz_bytes, parse_npz_bytes_with_limits,
        parse_npz_from_reader, parse_npz_from_reader_with_limits, parse_npz_with_limits,
    };
    use std::io::Cursor;

    let tmp = tempfile::NamedTempFile::new().expect("temp file");
    let path = tmp.path();
    let default = ParseLimits::default();
    let tight = tight();

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("npz inspect_from_reader / {label}"), || {
            inspect_npz_from_reader(Cursor::new(bytes.clone()))
        });
        stage(path, &bytes);
        assert_no_panic(&format!("npz inspect_npz(path) / {label}"), || {
            inspect_npz(path)
        });
        assert_no_panic(&format!("npz parse_npz(path) / {label}"), || {
            parse_npz(path)
        });
        assert_no_panic(
            &format!("npz parse_npz_with_limits(path)[default] / {label}"),
            || parse_npz_with_limits(path, &default),
        );
        assert_no_panic(
            &format!("npz parse_npz_with_limits(path)[tight] / {label}"),
            || parse_npz_with_limits(path, &tight),
        );

        // The owned-bytes and reader family, new in v0.7.6. `NPZ` was the one
        // format without a copy-based parse, so these are the entry points this
        // suite's own header promised ("every re-exported parse/inspect entry
        // point of all four formats") before they existed.
        assert_no_panic(&format!("npz parse_npz_bytes / {label}"), || {
            parse_npz_bytes(bytes.clone())
        });
        assert_no_panic(
            &format!("npz parse_npz_bytes_with_limits[default] / {label}"),
            || parse_npz_bytes_with_limits(bytes.clone(), &default),
        );
        assert_no_panic(
            &format!("npz parse_npz_bytes_with_limits[tight] / {label}"),
            || parse_npz_bytes_with_limits(bytes.clone(), &tight),
        );
        assert_no_panic(&format!("npz parse_npz_from_reader / {label}"), || {
            parse_npz_from_reader(Cursor::new(bytes.clone()))
        });
        assert_no_panic(
            &format!("npz parse_npz_from_reader_with_limits[default] / {label}"),
            || parse_npz_from_reader_with_limits(Cursor::new(bytes.clone()), &default),
        );
        assert_no_panic(
            &format!("npz parse_npz_from_reader_with_limits[tight] / {label}"),
            || parse_npz_from_reader_with_limits(Cursor::new(bytes.clone()), &tight),
        );

        // The bounded inspect forms, also v0.7.6: a hostile budget puts the
        // rejection arithmetic on the hook, not just the parse.
        let opts_default = InspectOptions::new().with_limits(default.clone());
        let opts_tight = InspectOptions::new().with_limits(tight.clone());
        assert_no_panic(
            &format!("npz inspect_npz_with_options[default] / {label}"),
            || inspect_npz_with_options(path, &opts_default),
        );
        assert_no_panic(
            &format!("npz inspect_npz_with_options[tight] / {label}"),
            || inspect_npz_with_options(path, &opts_tight),
        );
        assert_no_panic(
            &format!("npz inspect_npz_from_reader_with_options[tight] / {label}"),
            || inspect_npz_from_reader_with_options(Cursor::new(bytes.clone()), &opts_tight),
        );
    }
}

/// The format-agnostic entry points added in v0.7.6: magic-byte detection and
/// the in-memory `convert`.
///
/// Both take raw caller bytes and are reachable before any other validation —
/// `detect_format_from_bytes` walks a `ZIP` central directory to tell an `NPZ`
/// from a `.pth`, which is real parsing on hostile input — so they belong in
/// this battery as much as the per-format parsers do.
#[test]
fn format_agnostic_entry_points_never_panic() {
    use anamnesis::{
        ConvertOptions, ConvertTarget, convert_bytes, detect_format_from_bytes,
        detect_format_from_bytes_with_limits,
    };

    let default = ParseLimits::default();
    let tight = tight();

    for (label, bytes) in adversarial_inputs() {
        assert_no_panic(&format!("detect_format_from_bytes / {label}"), || {
            detect_format_from_bytes(&bytes)
        });
        assert_no_panic(
            &format!("detect_format_from_bytes_with_limits[default] / {label}"),
            || detect_format_from_bytes_with_limits(&bytes, &default),
        );
        assert_no_panic(
            &format!("detect_format_from_bytes_with_limits[tight] / {label}"),
            || detect_format_from_bytes_with_limits(&bytes, &tight),
        );

        // Built as a `Vec` rather than an array literal with `cfg`-gated
        // elements: in a build with neither `gguf` nor `bnb` that literal
        // collapses to one element and the loop trips `single_element_loop`.
        // `mut` only when a feature adds a target, so the binding's mutability
        // is itself feature-conditional.
        #[cfg_attr(not(any(feature = "gguf", feature = "bnb")), allow(unused_mut))]
        let mut targets: Vec<(&str, ConvertTarget)> =
            vec![("safetensors", ConvertTarget::Safetensors)];
        #[cfg(feature = "gguf")]
        targets.push(("gguf", ConvertTarget::Gguf));
        #[cfg(feature = "bnb")]
        targets.push(("bnb-nf4", ConvertTarget::BnbNf4));

        for (target_label, target) in targets {
            assert_no_panic(
                &format!("convert_bytes[{target_label}, default] / {label}"),
                || convert_bytes(&bytes, target, &ConvertOptions::new()),
            );
            assert_no_panic(
                &format!("convert_bytes[{target_label}, tight] / {label}"),
                || {
                    convert_bytes(
                        &bytes,
                        target,
                        &ConvertOptions::new().with_limits(tight.clone()),
                    )
                },
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Methods on a successful result
// ---------------------------------------------------------------------------

/// Everything above stops at "the entry point returned". A binding goes on to
/// call methods on the result (inspect it at a chosen width, list and extract
/// tensors, dequantise, serialise), and those run over whatever the parser
/// accepted. This drives each of them on every input that parses, including
/// the intact fixtures and their near-misses. The v0.7.8 fuzz campaign found
/// two panics in exactly this territory (`tests/fuzz_regressions.rs`).
#[test]
fn methods_on_parsed_results_never_panic() {
    use anamnesis::{InspectInfo, InspectOptions, TargetDtype};

    let widths = [TargetDtype::BF16, TargetDtype::F32, TargetDtype::F16];
    for (label, bytes) in adversarial_inputs() {
        if let Ok(model) = anamnesis::parse_bytes(bytes.clone()) {
            assert_no_panic(&format!("st methods / {label}"), || {
                let _ = model.inspect();
                let _ = InspectInfo::from(&model.header);
                for w in widths {
                    let _ = model.inspect_with_options(&InspectOptions::new().with_output_dtype(w));
                    let _ = model.remember_to_bytes(w);
                }
            });
        }

        #[cfg(feature = "gguf")]
        {
            let options = InspectOptions::new().with_output_dtype(TargetDtype::F32);
            assert_no_panic(&format!("gguf inspect_with_options / {label}"), || {
                anamnesis::inspect_gguf_from_reader_with_options(
                    std::io::Cursor::new(bytes.clone()),
                    &options,
                )
            });
            if let Ok(front) =
                anamnesis::parse_gguf_front_matter_from_reader(std::io::Cursor::new(bytes.clone()))
            {
                assert_no_panic(&format!("gguf front matter inspect / {label}"), || {
                    front.inspect_with_options(&options)
                });
            }
            if let Ok(gguf) = anamnesis::parse_gguf_bytes(bytes.clone()) {
                assert_no_panic(&format!("gguf methods / {label}"), || {
                    let _ = gguf.inspect_with_options(&options);
                    let _ = gguf.tensors().count();
                    for info in gguf.tensor_info() {
                        let _ = gguf.dequantize_tensor(info);
                        let _ = gguf.dequantize_tensor_as::<anamnesis::F32Out>(info);
                    }
                    let _ = gguf.remember_to_bytes(TargetDtype::BF16);
                });
            }
        }

        #[cfg(feature = "pth")]
        {
            let options = InspectOptions::new().with_output_dtype(TargetDtype::F32);
            assert_no_panic(&format!("pth inspect_with_options / {label}"), || {
                anamnesis::inspect_pth_from_reader_with_options(
                    std::io::Cursor::new(bytes.clone()),
                    &options,
                )
            });
            if let Ok(front) =
                anamnesis::parse_pth_front_matter_from_reader(std::io::Cursor::new(bytes.clone()))
            {
                assert_no_panic(&format!("pth front matter inspect / {label}"), || {
                    front.inspect_with_options(&options)
                });
            }
            if let Ok(pth) = anamnesis::parse_pth_bytes(bytes.clone()) {
                assert_no_panic(&format!("pth methods / {label}"), || {
                    let _ = pth.inspect_with_options(&options);
                    let _ = pth.tensor_info();
                    let _ = pth.tensors();
                    let _ = pth.to_safetensors_bytes();
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kernels called directly (Phase 7.10)
// ---------------------------------------------------------------------------

/// The public kernels added in Phase 7.10, called the way a binding might:
/// with a data length and a declared element count that need not agree.
/// Every combination must be a clean `Ok`/`Err`, including counts at
/// `usize::MAX`, where the output size overflows.
#[cfg(feature = "gguf")]
#[test]
fn new_gguf_kernels_never_panic_on_mismatched_inputs() {
    use anamnesis::{Bf16Out, F16Out, F32Out, GgufType, dequantize_gguf, dequantize_gguf_blocks};

    let counts = [
        0usize,
        1,
        63,
        64,
        65,
        127,
        128,
        129,
        256,
        usize::MAX / 2,
        usize::MAX,
    ];
    for dtype in [GgufType::NVFP4, GgufType::Q1_0, GgufType::Q2_0] {
        for len in [0usize, 1, 17, 18, 35, 36, 37, 72, 144] {
            let data = patterned(len);
            for n in counts {
                let label = format!("{dtype} len {len} n {n}");
                assert_no_panic(&format!("dequantize_gguf<BF16> / {label}"), || {
                    dequantize_gguf::<Bf16Out>(&data, dtype, n)
                });
                assert_no_panic(&format!("dequantize_gguf<F32> / {label}"), || {
                    dequantize_gguf::<F32Out>(&data, dtype, n)
                });
                assert_no_panic(&format!("dequantize_gguf_blocks<F16> / {label}"), || {
                    dequantize_gguf_blocks::<F16Out, _>(&data, dtype, n, |_| Ok(()))
                });
            }
        }
    }
}

/// `dequantize_nvfp4` (the `ModelOpt` kernel) with shapes and slices that
/// disagree, products that overflow, and scales that are not finite.
#[cfg(feature = "nvfp4")]
#[test]
fn modelopt_nvfp4_kernel_never_panics_on_mismatched_inputs() {
    use anamnesis::{Bf16Out, F32Out, dequantize_nvfp4};

    let dims = [0usize, 1, 2, 4, 15, 16, 17, 32, usize::MAX / 2, usize::MAX];
    for weight_len in [0usize, 1, 7, 8, 64] {
        let weight = patterned(weight_len);
        for scales_len in [0usize, 1, 4, 8] {
            let scales = patterned(scales_len);
            for global in [0.01_f32, 0.0, f32::NAN, f32::INFINITY] {
                for rows in dims {
                    for cols in dims {
                        let label = format!(
                            "weight {weight_len} scales {scales_len} global {global} \
                             rows {rows} cols {cols}"
                        );
                        assert_no_panic(&format!("dequantize_nvfp4<BF16> / {label}"), || {
                            dequantize_nvfp4::<Bf16Out>(&weight, &scales, global, rows, cols)
                        });
                        assert_no_panic(&format!("dequantize_nvfp4<F32> / {label}"), || {
                            dequantize_nvfp4::<F32Out>(&weight, &scales, global, rows, cols)
                        });
                    }
                }
            }
        }
    }
}
