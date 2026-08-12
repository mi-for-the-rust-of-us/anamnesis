// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase 6.5 — dequantisation throughput benchmarks (decode side).
//!
//! One `criterion` benchmark group per kernel family
//! (`FP8`, `GPTQ` 4-bit, `AWQ` 4-bit, `BnB` `NF4`, `BnB` `INT8`,
//! `GGUF` `Q4_K`), measured on synthetic tensors sized like a real
//! transformer layer (`4096 × 11008`). Plus one real-world group on
//! the `Ollama`-distributed `llama3.2:1b` `Q8_0` slice the
//! [`cross_validation_ollama`](../tests/cross_validation_ollama.rs) test
//! validates against — same fixture, different harness.
//!
//! Synthesised inputs are deterministic (seeded by element index) so
//! `criterion`'s regression detection sees stable bit-patterns across
//! runs. The exact values matter little for throughput — every kernel
//! walks the same byte count regardless — but determinism keeps
//! cache-warm vs cold runs comparable.
//!
//! Run with:
//!
//! ```text
//! cargo bench --features gptq,awq,bnb,gguf --bench dequant
//! ```
//!
//! Reports land in `target/criterion/`; HTML index at
//! `target/criterion/report/index.html`.
//!
//! # Memory
//!
//! The synthetic-layer dequant benches allocate ~25 `MiB` of input
//! (packed weights + scales + zero-points) and ~90 `MiB` of `BF16`
//! output per iteration, with `criterion` dropping both between
//! iterations — peak resident is dominated by one snapshot of
//! `(input + output)` ≈ 115 `MiB`. The `Ollama`-fixture bench peaks
//! at ~200 `KiB` (the slice is 65 536 elements). Comfortable on any
//! machine with > 256 `MiB` free RAM.

#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

use anamnesis::{
    Dtype, F32Out, GgufType, dequantize_awq_to_bf16, dequantize_bnb_int8_to_bf16,
    dequantize_bnb4_to_bf16, dequantize_fp8_to_bf16, dequantize_gguf, dequantize_gguf_to_bf16,
    dequantize_gptq_to_bf16, dequantize_per_tensor_fp8_to_bf16,
};

// ---------------------------------------------------------------------------
// Synthetic layer dimensions
// ---------------------------------------------------------------------------

/// Rows for the synthetic transformer-layer-sized fixture. Matches a
/// `Llama`-class FFN gate/up/down matrix's smaller dimension.
const LAYER_ROWS: usize = 4096;
/// Cols for the synthetic transformer-layer-sized fixture. Matches the
/// `intermediate_size` of common 7B-class models (`Mistral`, `Llama-2-7B`).
const LAYER_COLS: usize = 11008;
/// Total elements per synthetic layer fixture (`4096 × 11008`).
const LAYER_ELEMENTS: usize = LAYER_ROWS * LAYER_COLS;

// ---------------------------------------------------------------------------
// Deterministic synthesis helpers
// ---------------------------------------------------------------------------

/// Fills a buffer with deterministic non-zero bytes via a Knuth
/// multiplicative hash on the index. Avoids the all-zero pathology
/// that some quantisation kernels short-circuit through their
/// fast-path branches.
fn fill_deterministic(buf: &mut [u8]) {
    for (i, b) in buf.iter_mut().enumerate() {
        *b = (i.wrapping_mul(2_654_435_761) & 0xFF) as u8;
    }
}

fn synth_bytes(n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    fill_deterministic(&mut v);
    v
}

// ---------------------------------------------------------------------------
// FP8 — per-tensor (single scale, full layer)
// ---------------------------------------------------------------------------

fn bench_fp8_per_tensor(c: &mut Criterion) {
    let weight = synth_bytes(LAYER_ELEMENTS);
    let scale: f32 = 0.125_f32;

    let mut group = c.benchmark_group("dequant_fp8_per_tensor");
    group.throughput(Throughput::Elements(LAYER_ELEMENTS as u64));
    group.bench_function("synthetic_4096x11008", |b| {
        b.iter(|| {
            let out = dequantize_per_tensor_fp8_to_bf16(black_box(&weight), black_box(scale))
                .expect("fp8 per-tensor dequant");
            black_box(out);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// FP8 — fine-grained (128x128 block scales)
// ---------------------------------------------------------------------------

fn bench_fp8_fine_grained(c: &mut Criterion) {
    let weight = synth_bytes(LAYER_ELEMENTS);
    // Fine-grained FP8 uses 128x128 block scales (BF16 LE).
    let block_rows = LAYER_ROWS / 128;
    let block_cols = LAYER_COLS / 128;
    // `0x3F80` = BF16 representation of 1.0.
    let scale_pair = [0x80u8, 0x3F];
    let scale_data: Vec<u8> = scale_pair.repeat(block_rows * block_cols);

    let mut group = c.benchmark_group("dequant_fp8_fine_grained");
    group.throughput(Throughput::Elements(LAYER_ELEMENTS as u64));
    group.bench_function("synthetic_4096x11008", |b| {
        b.iter(|| {
            let out = dequantize_fp8_to_bf16(
                black_box(&weight),
                black_box(&scale_data),
                LAYER_ROWS,
                LAYER_COLS,
                Dtype::BF16,
            )
            .expect("fp8 fine-grained dequant");
            black_box(out);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// GPTQ INT4 — group-wise scales + per-group zero-points, no g_idx
// ---------------------------------------------------------------------------

fn bench_gptq_int4(c: &mut Criterion) {
    let in_features: usize = LAYER_ROWS;
    let out_features: usize = LAYER_COLS;
    let bits: u8 = 4;
    let group_size: usize = 128;
    let pack_factor: usize = 32 / bits as usize;

    // qweight shape: [in_features / pack_factor, out_features], each element a u32
    let packed_rows = in_features / pack_factor;
    let qweight = synth_bytes(packed_rows * out_features * 4);
    // scales shape: [num_groups, out_features], BF16 LE (= 2 bytes / elem)
    let num_groups = in_features / group_size;
    let mut scales = vec![0u8; num_groups * out_features * 2];
    // `0x3F00` = BF16 representation of 0.5. Non-zero so dequant output is non-trivial.
    for pair in scales.chunks_exact_mut(2) {
        pair[0] = 0x00;
        pair[1] = 0x3F;
    }
    // qzeros shape: [num_groups, out_features / pack_factor], each element a u32
    let qzeros = synth_bytes(num_groups * (out_features / pack_factor) * 4);

    let mut group = c.benchmark_group("dequant_gptq_int4");
    group.throughput(Throughput::Elements(LAYER_ELEMENTS as u64));
    group.bench_function("synthetic_4096x11008_g128", |b| {
        b.iter(|| {
            let out = dequantize_gptq_to_bf16(
                black_box(&qweight),
                black_box(&scales),
                black_box(&qzeros),
                None,
                in_features,
                out_features,
                group_size,
                bits,
                Dtype::BF16,
            )
            .expect("gptq dequant");
            black_box(out);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// AWQ INT4 — column-packed (`qweight: [in_features, out_features / pack_factor]`)
// ---------------------------------------------------------------------------

fn bench_awq_int4(c: &mut Criterion) {
    let in_features: usize = LAYER_ROWS;
    let out_features: usize = LAYER_COLS;
    let bits: u8 = 4;
    let group_size: usize = 128;
    let pack_factor: usize = 32 / bits as usize;

    // AWQ qweight shape: [in_features, out_features / pack_factor], u32 packed.
    let qweight = synth_bytes(in_features * (out_features / pack_factor) * 4);
    let num_groups = in_features / group_size;
    let mut scales = vec![0u8; num_groups * out_features * 2];
    for pair in scales.chunks_exact_mut(2) {
        pair[0] = 0x00;
        pair[1] = 0x3F; // BF16 0.5
    }
    let qzeros = synth_bytes(num_groups * (out_features / pack_factor) * 4);

    let mut group = c.benchmark_group("dequant_awq_int4");
    group.throughput(Throughput::Elements(LAYER_ELEMENTS as u64));
    group.bench_function("synthetic_4096x11008_g128", |b| {
        b.iter(|| {
            let out = dequantize_awq_to_bf16(
                black_box(&qweight),
                black_box(&scales),
                black_box(&qzeros),
                in_features,
                out_features,
                group_size,
                bits,
                Dtype::BF16,
            )
            .expect("awq dequant");
            black_box(out);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// BnB NF4 — 16-entry codebook, per-block absmax (block_size = 64)
// ---------------------------------------------------------------------------

fn bench_bnb_nf4(c: &mut Criterion) {
    let total_elements: usize = LAYER_ELEMENTS;
    let block_size: usize = 64;

    // Packed 4-bit weight: total_elements / 2 bytes.
    let weight = synth_bytes(total_elements / 2);
    // Absmax: f32 per block, all 1.0 so dequant output is the codebook itself.
    let num_blocks = total_elements / block_size;
    let absmax: Vec<u8> = (0..num_blocks)
        .flat_map(|_| 1.0_f32.to_le_bytes())
        .collect();
    // Canonical NF4 codebook (same constant as anamnesis::NF4_CODEBOOK).
    let quant_map_floats: [f32; 16] = [
        -1.0,
        -0.696_192_8,
        -0.525_073_05,
        -0.394_917_5,
        -0.284_441_38,
        -0.184_773_43,
        -0.091_050_036,
        0.0,
        0.079_580_3,
        0.160_930_2,
        0.246_112_3,
        0.337_915_24,
        0.440_709_83,
        0.562_617,
        0.722_956_84,
        1.0,
    ];
    let quant_map: Vec<u8> = quant_map_floats
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let mut group = c.benchmark_group("dequant_bnb_nf4");
    group.throughput(Throughput::Elements(total_elements as u64));
    group.bench_function("synthetic_4096x11008_b64", |b| {
        b.iter(|| {
            let out = dequantize_bnb4_to_bf16(
                black_box(&weight),
                black_box(&absmax),
                black_box(&quant_map),
                total_elements,
                block_size,
            )
            .expect("bnb nf4 dequant");
            black_box(out);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// BnB INT8 — per-row absmax / 127 (LLM.int8())
// ---------------------------------------------------------------------------

fn bench_bnb_int8(c: &mut Criterion) {
    // BnB INT8 stores its weight as `[out_features, in_features]` — the
    // opposite of the FFN-gate ordering used by GPTQ/AWQ/Q4_K above.
    // Bench label below uses the same `<out>x<in>` form so the criterion
    // report's shape annotation matches the on-disk byte layout.
    let out_features: usize = LAYER_COLS;
    let in_features: usize = LAYER_ROWS;
    // INT8 weight: out_features * in_features int8 bytes.
    let weight = synth_bytes(out_features * in_features);
    // SCB: per-row f32 absmax (one f32 per `out_features` row).
    let scb: Vec<u8> = (0..out_features)
        .flat_map(|_| 1.0_f32.to_le_bytes())
        .collect();

    let mut group = c.benchmark_group("dequant_bnb_int8");
    group.throughput(Throughput::Elements((out_features * in_features) as u64));
    group.bench_function("synthetic_11008x4096", |b| {
        b.iter(|| {
            let out = dequantize_bnb_int8_to_bf16(
                black_box(&weight),
                black_box(&scb),
                out_features,
                in_features,
            )
            .expect("bnb int8 dequant");
            black_box(out);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// GGUF Q4_K — 256-element super-block, 144 bytes/block
// ---------------------------------------------------------------------------

fn bench_gguf_q4_k(c: &mut Criterion) {
    let n_elements: usize = LAYER_ELEMENTS;
    // Q4_K block layout: 144 bytes per 256-element block.
    let n_blocks = n_elements / 256;
    let raw = synth_bytes(n_blocks * 144);

    let mut group = c.benchmark_group("dequant_gguf_q4_k");
    group.throughput(Throughput::Elements(n_elements as u64));
    group.bench_function("synthetic_4096x11008", |b| {
        b.iter(|| {
            let out = dequantize_gguf_to_bf16(black_box(&raw), GgufType::Q4_K, n_elements)
                .expect("gguf q4_k dequant");
            black_box(out);
        });
    });
    // Phase 7.3: the cost curve for the new `F32` option, on the kernel that
    // drives `run_super_kernel`. This arm RECORDS rather than gates: `F32`
    // doubles the output bytes on a bandwidth-bound path, so it is *expected*
    // to be slower, and the phase claims exactness rather than speed. The
    // `BF16` id above is the regression guard, and it is unchanged.
    //
    // No `F16` arm: same width as `BF16`, so there is no bandwidth story to
    // tell, and its interesting properties (accuracy, exponent range) are
    // tests rather than benchmarks.
    group.bench_function("synthetic_4096x11008_f32", |b| {
        b.iter(|| {
            let out = dequantize_gguf::<F32Out>(black_box(&raw), GgufType::Q4_K, n_elements)
                .expect("gguf q4_k dequant f32");
            black_box(out);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// GGUF Q8_0 — real-world Ollama-distributed fixture (llama3.2:1b)
// ---------------------------------------------------------------------------

/// Reads the 16-byte fixture header and returns
/// `(n_elements, raw_data)` for the `Q8_0` slice from
/// `tests/fixtures/ollama_reference/llama3_2_1b_q8_0.bin`. Same parser
/// shape as `cross_validation_ollama::parse_ollama_fixture` but
/// duplicated locally so the bench has zero `tests/` cross-dependency.
fn load_ollama_q8_0_fixture() -> (usize, Vec<u8>) {
    const FIXTURE: &[u8] =
        include_bytes!("../tests/fixtures/ollama_reference/llama3_2_1b_q8_0.bin");
    let disc = u32::from_le_bytes([FIXTURE[0], FIXTURE[1], FIXTURE[2], FIXTURE[3]]);
    assert_eq!(disc, 8, "expected Q8_0 discriminant in Ollama fixture");
    let n_elements = u32::from_le_bytes([FIXTURE[4], FIXTURE[5], FIXTURE[6], FIXTURE[7]]) as usize;
    let raw_data_len =
        u32::from_le_bytes([FIXTURE[8], FIXTURE[9], FIXTURE[10], FIXTURE[11]]) as usize;
    let raw_start: usize = 16;
    let raw = FIXTURE[raw_start..raw_start + raw_data_len].to_vec();
    (n_elements, raw)
}

fn bench_gguf_q8_0_ollama(c: &mut Criterion) {
    let (n_elements, raw) = load_ollama_q8_0_fixture();

    let mut group = c.benchmark_group("dequant_gguf_q8_0_ollama");
    group.throughput(Throughput::Elements(n_elements as u64));
    group.bench_function("llama3.2_1b_blk0_attn_q_65536", |b| {
        b.iter(|| {
            let out = dequantize_gguf_to_bf16(black_box(&raw), GgufType::Q8_0, n_elements)
                .expect("gguf q8_0 ollama dequant");
            black_box(out);
        });
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion plumbing
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_fp8_per_tensor,
    bench_fp8_fine_grained,
    bench_gptq_int4,
    bench_awq_int4,
    bench_bnb_nf4,
    bench_bnb_int8,
    bench_gguf_q4_k,
    bench_gguf_q8_0_ollama,
);
criterion_main!(benches);
