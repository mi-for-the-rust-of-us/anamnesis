// SPDX-License-Identifier: MIT OR Apache-2.0

//! Command-line interface implementation, shared by the `anamnesis` and
//! `amn` binaries.
//!
//! Feature-gated behind `cli`; pulls in `clap` for argument parsing.
//! The two binary entry points (`src/bin/anamnesis.rs` and
//! `src/bin/amn.rs`) are 5-line wrappers that each delegate to `run`,
//! so the actual CLI code compiles exactly once and links into both
//! binaries instead of being compiled twice as two separate crate
//! roots (which is what the previous shared-`src/bin/main.rs` shape
//! did, producing the Cargo *"file found to be present in multiple
//! build targets"* warning on every invocation).

use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};

use crate::convert::{ConvertOptions, ConvertTarget, Format, detect_format};
use crate::{InspectInfo, InspectOptions, TargetDtype, format_bytes, parse};

/// Parse any format, recover any precision.
#[derive(Parser)]
#[command(name = "anamnesis", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse and summarize a model file.
    Parse {
        /// Path to the model file (`.safetensors`, `.pth`, `.pt`, `.bin`, `.gguf`).
        path: PathBuf,
    },
    /// Inspect format, tensor counts, and size estimates.
    #[command(alias = "info")]
    Inspect {
        /// Path to the model file.
        path: PathBuf,
        /// Output dtype the size estimate assumes: `bf16` (default), `f32`,
        /// or `f16`.
        ///
        /// The estimate feeds the inspect-before-parse decision, so it has to
        /// be sized at the width you actually intend to `remember` or
        /// `convert` at: an `F32` request against a `BF16` estimate
        /// under-reserves by exactly `2 ×`. `remember --to` fixed this same
        /// bug for its own summary line in v0.7.4; `inspect` had no flag at
        /// all until v0.7.6.
        ///
        /// Vacuous on `.pth` and `NPZ`, whose tensors are already full
        /// precision and pass through in their source dtype. Accepted there
        /// for the same reason `remember --to f32` is.
        #[arg(long, default_value = "bf16")]
        to: String,
    },
    /// Dequantize (recover precision) or convert to a target format.
    #[command(alias = "dequantize")]
    Remember {
        /// Path to the input model file.
        path: PathBuf,
        /// Output dtype for dequantised tensors: `bf16` (default), `f32`, or
        /// `f16`. `safetensors` is accepted as an alias for `bf16` on
        /// `.pth`/`.gguf` inputs, which always produce a safetensors file.
        ///
        /// `f32` emits the reference implementation's own `f32` with no
        /// narrowing step of anamnesis's, at double the output bytes. `f16`
        /// buys 3 significand bits over `bf16` and pays a far narrower exponent
        /// range (it saturates at 65504).
        ///
        /// Applies to **dequantised** tensors only; passthrough tensors keep
        /// their source dtype, so the output is legitimately mixed-dtype. On a
        /// `.pth` input nothing is dequantised, so the value is accepted and
        /// has no effect.
        #[arg(long, default_value = "bf16")]
        to: String,
        /// Output file path (derived from input if omitted).
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Dequantisation worker threads. Defaults to `min(cpu cores, 4)` — the
        /// measured scaling knee — so the rest of the machine stays free.
        /// Values below 1 are clamped to 1 (fully sequential). Output is
        /// byte-identical whatever you pass.
        #[arg(long, value_name = "N")]
        threads: Option<usize>,
    },
    /// Convert a model file to a different format.
    ///
    /// Targets available in this build (Phase 6):
    /// - `safetensors` (alias `bf16`) — dequantise any quantised input to a
    ///   BF16 safetensors file (passes through unquantised inputs losslessly).
    /// - `gguf` — write an unquantised GGUF file. Quantised GGUF emit
    ///   (`gguf-q4km`, …) is deferred to Phase 8.5 via the same dispatch.
    /// - `bnb-nf4` — encode the BF16 source into a BitsAndBytes-NF4
    ///   safetensors file (2-D tensors only; biases / norms / embeddings
    ///   pass through unchanged in BF16).
    Convert {
        /// Path to the input model file.
        path: PathBuf,
        /// Target format. Accepted values: `safetensors`/`bf16`, `gguf`,
        /// `bnb-nf4` (case-insensitive).
        #[arg(long)]
        to: String,
        /// Output file path (derived from input if omitted).
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// JSON file of `GGUF` metadata key/values to stamp on a `gguf` target.
        ///
        /// Values are typed: plain JSON is inferred (string, bool, integer →
        /// `u32`, float → `f32`, array from its first element), and an explicit
        /// `{"type": "i32", "value": 3}` — or `{"type": "array", "item_type":
        /// "i32", "value": [..]}` — pins an exact width. Merged over any KV
        /// inherited from a `GGUF` source; `--gguf-kv` wins over this file.
        #[arg(long, value_name = "FILE")]
        gguf_metadata: Option<PathBuf>,
        /// Repeatable `key=value` `GGUF` metadata. The value is always written as
        /// a string — use `--gguf-metadata` for typed or array values.
        #[arg(long, value_name = "KEY=VALUE")]
        gguf_kv: Vec<String>,
        /// Element type for tensors this conversion dequantises:
        /// `bf16` (default), `f32`, or `f16`.
        ///
        /// Named `--out-dtype` rather than reusing `--to`, because on `convert`
        /// `--to` already selects the output *format*. (On `remember`, `--to`
        /// selects a dtype, so that subcommand needs no new flag.)
        ///
        /// `f32` emits the reference implementation's own `f32` with no
        /// narrowing step of anamnesis's, at double the output bytes. `f16`
        /// buys 3 significand bits over `bf16` and pays a far narrower exponent
        /// range, saturating to infinity above 65504.
        ///
        /// Applies to **dequantised tensors only**. Passthrough tensors (norms,
        /// biases, anything not block-quantised) keep their source dtype, so
        /// this is not "rewrite every tensor as f32".
        ///
        /// Honoured for **every** input format that dequantises, since v0.7.4.
        /// (v0.7.3 accepted non-`bf16` values for `GGUF` input only and
        /// reported a clear error for quantised safetensors; that restriction
        /// is gone.) `NPZ` and `.pth` dequantise nothing, so the value is
        /// accepted and has no effect there.
        #[arg(long, value_name = "DTYPE", default_value = "bf16")]
        out_dtype: String,
        /// Dequantisation worker threads. Defaults to `min(cpu cores, 4)` — the
        /// measured scaling knee — so the rest of the machine stays free.
        /// Values below 1 are clamped to 1 (fully sequential). Output is
        /// byte-identical whatever you pass.
        #[arg(long, value_name = "N")]
        threads: Option<usize>,
    },
}

// ---------------------------------------------------------------------------
// Subcommand runners
// ---------------------------------------------------------------------------

/// Parses CLI arguments and dispatches to the appropriate subcommand
/// runner.
///
/// Entry point shared by the `anamnesis` and `amn` binaries; both thin
/// wrappers under `src/bin/` call this function and translate any
/// returned error into a `process::exit(1)`.
///
/// # Errors
///
/// Propagates any [`crate::AnamnesisError`] returned by the underlying
/// format parsers, dequantisation kernels, or output writers.
pub fn run() -> crate::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse { path } => {
            let resolved = resolve_input_path(path)?;
            run_parse(&resolved)
        }
        Commands::Inspect { path, to } => {
            let resolved = resolve_input_path(path)?;
            let target: TargetDtype = to.parse()?;
            run_inspect(&resolved, target)
        }
        Commands::Remember {
            path,
            to,
            output,
            threads,
        } => {
            let resolved = resolve_input_path(path)?;
            run_remember(&resolved, &to, output.as_deref(), threads)
        }
        Commands::Convert {
            path,
            to,
            output,
            gguf_metadata,
            gguf_kv,
            out_dtype,
            threads,
        } => {
            let resolved = resolve_input_path(path)?;
            run_convert(
                &resolved,
                &to,
                output.as_deref(),
                gguf_metadata.as_deref(),
                &gguf_kv,
                &out_dtype,
                threads,
            )
        }
    }
}

/// Resolves a CLI-supplied input path, expanding the `ollama:` URL
/// scheme to the on-disk `GGUF` blob path inside the local `Ollama`
/// model cache.
///
/// Recognised forms:
///
/// - `ollama:<model>:<tag>` (e.g., `ollama:llama3.2:1b`) — resolves
///   the manifest at `~/.ollama/models/manifests/registry.ollama.ai/library/<model>/<tag>`
///   to its model-layer blob.
/// - `ollama:<model>` — same as above with the tag defaulting to
///   `latest`.
/// - Any other input — returned unchanged; the existing format
///   detection pipeline handles regular file paths.
///
/// # Errors
///
/// Returns [`crate::AnamnesisError::Unsupported`] when the input uses
/// the `ollama:` scheme but the binary was built without the
/// `ollama` Cargo feature.
///
/// Returns the [`crate::AnamnesisError`] variants documented on
/// [`resolve_ollama_model`](crate::resolve_ollama_model) otherwise.
#[allow(clippy::unnecessary_wraps)]
fn resolve_input_path(raw: PathBuf) -> crate::Result<PathBuf> {
    let s = raw.to_string_lossy();
    if s.starts_with("ollama:") {
        #[cfg(feature = "ollama")]
        {
            return crate::resolve_ollama_model(&s);
        }
        #[cfg(not(feature = "ollama"))]
        {
            return Err(crate::AnamnesisError::Unsupported {
                format: "ollama:".into(),
                detail: "the `ollama:` URL scheme requires the `ollama` Cargo feature; \
                         rebuild with `cargo install anamnesis --features cli,ollama` \
                         (or `cargo build --features cli,ollama`) to add support"
                    .into(),
            });
        }
    }
    Ok(raw)
}

fn run_parse(path: &std::path::Path) -> crate::Result<()> {
    match detect_format(path)? {
        Format::Safetensors => run_parse_safetensors(path),
        #[cfg(feature = "pth")]
        Format::Pth => run_parse_pth(path),
        #[cfg(feature = "npz")]
        Format::Npz => run_parse_npz(path),
        #[cfg(feature = "gguf")]
        Format::Gguf => run_parse_gguf(path),
    }
}

fn run_parse_safetensors(path: &std::path::Path) -> crate::Result<()> {
    let model = parse(path)?;
    let info = InspectInfo::from(&model.header);
    let total = model.header.tensors.len();

    println!("{total} tensors parsed");

    let quantized = model.header.quantized_count();
    if quantized > 0 {
        println!("  {quantized:>3} quantized   {}", model.header.scheme);
    }

    let scales = model.header.scale_count();
    if scales > 0 {
        let mut dtypes: Vec<String> = Vec::new();
        for entry in model.header.scale_tensors() {
            let s = entry.dtype.to_string();
            if !dtypes.contains(&s) {
                dtypes.push(s);
            }
        }
        let dtype_list = dtypes.join(", ");
        println!("  {scales:>3} scale       {dtype_list}");
    }

    let zeropoints = model.header.zeropoint_count();
    if zeropoints > 0 {
        println!("  {zeropoints:>3} zero-point  I32 (packed)");
    }

    let group_indices = model.header.group_index_count();
    if group_indices > 0 {
        println!("  {group_indices:>3} g_idx       I32 (activation-order)");
    }

    let passthrough = model.header.passthrough_count();
    if passthrough > 0 {
        // Collect passthrough dtype summary.
        let mut dtypes: Vec<String> = Vec::new();
        for entry in model.header.passthrough_tensors() {
            let s = entry.dtype.to_string();
            if !dtypes.contains(&s) {
                dtypes.push(s);
            }
        }
        let dtype_list = dtypes.join(", ");
        println!("  {passthrough:>3} passthrough {dtype_list} (norms, embeddings, lm_head)");
    }

    println!("File: {}", format_bytes(info.current_size));
    Ok(())
}

#[cfg(feature = "pth")]
fn run_parse_pth(path: &std::path::Path) -> crate::Result<()> {
    let parsed = crate::parse_pth(path)?;
    let info = parsed.inspect();
    // Use tensor_info() (metadata only) instead of tensors() — avoids
    // materializing tensor data just for the display path.
    let tensor_info = parsed.tensor_info();

    println!(
        "Parsed {} (PyTorch state_dict)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)")
    );
    println!("  Tensors:    {}", info.tensor_count);
    println!("  Total size: {}", format_bytes(info.total_bytes));
    let dtype_list: String = info
        .dtypes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    println!("  Dtypes:     {dtype_list}");
    let endian = if info.big_endian {
        "big-endian"
    } else {
        "little-endian"
    };
    println!("  Byte order: {endian}");
    println!();

    for t in &tensor_info {
        let shape_str = format!("{:?}", t.shape);
        // CAST: usize → u64, tensor byte lengths fit in u64
        #[allow(clippy::as_conversions)]
        let byte_len = t.byte_len as u64;
        println!(
            "  {:<30} {:<6} {:<15} {}",
            t.name,
            t.dtype,
            shape_str,
            format_bytes(byte_len)
        );
    }
    Ok(())
}

#[cfg(feature = "npz")]
fn run_parse_npz(path: &std::path::Path) -> crate::Result<()> {
    // Use inspect_npz (header-only) instead of parse_npz — avoids loading
    // all tensor data into memory for a display-only operation.
    let info = crate::inspect_npz(path)?;

    println!(
        "Parsed {} (NPZ archive)",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)")
    );
    println!("  Tensors:    {}", info.tensors.len());
    println!("  Total size: {}", format_bytes(info.total_bytes));
    println!();

    for t in &info.tensors {
        let shape_str = format!("{:?}", t.shape);
        // CAST: usize → u64, tensor byte lengths fit
        #[allow(clippy::as_conversions)]
        let byte_len = t.byte_len as u64;
        println!(
            "  {:<30} {:<6} {:<15} {}",
            t.name,
            t.dtype,
            shape_str,
            format_bytes(byte_len)
        );
    }
    Ok(())
}

#[cfg(feature = "npz")]
fn run_inspect_npz(path: &std::path::Path, options: &InspectOptions) -> crate::Result<()> {
    // Header-only — no tensor data loaded.
    let info = crate::inspect_npz_with_options(path, options)?;
    println!("{info}");
    Ok(())
}

/// `inspect` for any supported format, sizing every size estimate at `target`.
///
/// The `target` parameter is v0.7.6's. Before it, this printed whatever the
/// `BF16` default produced, which made `amn inspect` unable to answer the one
/// question it exists to answer — *"how much memory will this become?"* — for a
/// caller who intended `--to f32`. That is the identical bug v0.7.4 fixed for
/// `remember`'s summary line and did not carry across.
fn run_inspect(path: &std::path::Path, target: TargetDtype) -> crate::Result<()> {
    let options = InspectOptions::new().with_output_dtype(target);
    match detect_format(path)? {
        Format::Safetensors => {
            let model = parse(path)?;
            let info = model.inspect_with_options(&options);
            println!("{info}");
        }
        #[cfg(feature = "pth")]
        Format::Pth => {
            let parsed = crate::parse_pth(path)?;
            let info = parsed.inspect_with_options(&options);
            println!("{info}");
        }
        #[cfg(feature = "npz")]
        Format::Npz => run_inspect_npz(path, &options)?,
        #[cfg(feature = "gguf")]
        Format::Gguf => {
            let parsed = crate::parse_gguf(path)?;
            let info = parsed.inspect_with_options(&options);
            println!("{info}");
        }
    }
    Ok(())
}

/// `threads` is the caller's `--threads` request, passed straight through to
/// [`RememberOptions`](crate::RememberOptions) (`None` = the built-in
/// `min(cores, 4)` default). Only the safetensors arm has dequantisation work to
/// spread; the `.pth` and `GGUF` arms below are whole-file conversions that go
/// through their own writers.
fn run_remember(
    path: &std::path::Path,
    to: &str,
    output: Option<&std::path::Path>,
    threads: Option<usize>,
) -> crate::Result<()> {
    match detect_format(path)? {
        Format::Safetensors => run_remember_safetensors(path, to, output, threads),
        #[cfg(feature = "pth")]
        Format::Pth => {
            // A `.pth` is already full precision, so `remember` copies its
            // tensors through in their source dtype and dequantises nothing.
            // `--to f32` / `--to f16` are therefore **vacuous rather than
            // wrong** here, and are accepted for the same reason `convert`
            // accepts `--out-dtype` on `NPZ` and `.pth`: refusing
            // `--to f32` on a file that is already `F32` would be hostile.
            // Anything that is not an output dtype or the format alias is
            // still rejected.
            let to_lower = to.to_ascii_lowercase();
            let recognised = to_lower == "safetensors" || to_lower.parse::<TargetDtype>().is_ok();
            if !recognised {
                return Err(crate::AnamnesisError::Unsupported {
                    format: "pth".into(),
                    detail: format!(
                        "unsupported --to value `{to}` for .pth files \
                         (supported: `safetensors`, `bf16`, `f32`, `f16` — .pth \
                         tensors are already full precision, so the dtype is \
                         accepted but nothing is narrowed or widened)"
                    ),
                });
            }
            run_remember_pth(path, output)
        }
        #[cfg(feature = "npz")]
        Format::Npz => {
            // Resolved the same way as the `.pth` arm above, and for the same
            // reason. An `NPZ` is already full precision, so `remember` copies
            // its arrays through in their source dtype and dequantises nothing;
            // `--to f32` is therefore vacuous rather than wrong, and refusing it
            // on a file that is already `F32` would be hostile.
            //
            // Until v0.7.6 this arm returned `Unsupported`, while
            // `npz_to_safetensors` sat exported in the library and
            // `amn convert file.npz --to safetensors` did the very same job.
            // One verb, one meaning, across all four formats.
            let to_lower = to.to_ascii_lowercase();
            let recognised = to_lower == "safetensors" || to_lower.parse::<TargetDtype>().is_ok();
            if !recognised {
                return Err(crate::AnamnesisError::Unsupported {
                    format: "NPZ".into(),
                    detail: format!(
                        "unsupported --to value `{to}` for .npz files \
                         (supported: `safetensors`, `bf16`, `f32`, `f16` — NPZ \
                         arrays are already full precision, so the dtype is \
                         accepted but nothing is narrowed or widened)"
                    ),
                });
            }
            run_remember_npz(path, output)
        }
        #[cfg(feature = "gguf")]
        Format::Gguf => {
            // Unlike `.pth`, a quantised `GGUF` really is dequantised here, so
            // the dtype is **honoured** rather than vacuous. The 24 `GGUF`
            // kernels became generic over `OutputElement` in v0.7.3; v0.7.4
            // only had to thread the choice through this arm.
            let to_lower = to.to_ascii_lowercase();
            let target = if to_lower == "safetensors" {
                TargetDtype::BF16
            } else {
                to_lower
                    .parse::<TargetDtype>()
                    .map_err(|_| crate::AnamnesisError::Unsupported {
                        format: "GGUF".into(),
                        detail: format!(
                            "unsupported --to value `{to}` for .gguf files \
                             (supported: `safetensors`, `bf16`, `f32`, `f16`)"
                        ),
                    })?
            };
            run_remember_gguf(path, output, target, threads)
        }
    }
}

fn run_remember_safetensors(
    path: &std::path::Path,
    to: &str,
    output: Option<&std::path::Path>,
    threads: Option<usize>,
) -> crate::Result<()> {
    let target: TargetDtype = to.parse()?;

    let model = parse(path)?;
    // Size the estimate at the width the caller actually asked for. `From` is
    // the `BF16` default, so using it here made `--to f32` report the `BF16`
    // figure: the summary line claimed 144 B beside a file holding 272 B of
    // payload. `InspectOptions` exists precisely so a size can be computed for
    // a stated width, and this is its most obvious consumer.
    let info = InspectInfo::with_options(
        &model.header,
        &InspectOptions::new().with_output_dtype(target),
    );

    let total = model.header.tensors.len();
    let quantized = model.header.quantized_count();
    println!("Parsing...  {total} tensors, {}", model.header.scheme);

    let output_path = match output {
        Some(p) => p.to_owned(),
        None => derive_output_path(path, target),
    };

    // `None` keeps the library's `min(cores, 4)` default; `Some(n)` is the
    // caller's `--threads`, clamped to at least 1 by the builder.
    let opts = match threads {
        Some(n) => crate::RememberOptions::new().with_threads(n),
        None => crate::RememberOptions::new(),
    };

    #[cfg(feature = "indicatif")]
    {
        use indicatif::{ProgressBar, ProgressStyle};

        // CAST: usize → u64, tensor count fits in u64
        #[allow(clippy::as_conversions)]
        let pb = ProgressBar::new(quantized as u64);
        let style = ProgressStyle::with_template("Recalling... {pos} tensors [{bar:20}] {elapsed}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=> ");
        pb.set_style(style);
        model.remember_with_progress_and_options(&output_path, target, opts, || pb.inc(1))?;
        pb.finish();
        println!();
    }

    #[cfg(not(feature = "indicatif"))]
    {
        println!("Recalling... {quantized} tensors");
        model.remember_with_options(&output_path, target, opts)?;
    }

    println!(
        "Output: {} ({})",
        output_path.display(),
        format_bytes(info.dequantized_size),
    );
    Ok(())
}

/// `remember` for an `NPZ` input: copy every array through into a safetensors
/// file, dequantising nothing.
///
/// New in v0.7.6. The library has exported `npz_to_safetensors` since Phase 3
/// and `amn convert file.npz --to safetensors` has always performed exactly
/// this conversion; only `remember` refused, which left one verb meaning
/// different things on different formats — something a Python binding would
/// have had to reproduce.
#[cfg(feature = "npz")]
fn run_remember_npz(path: &std::path::Path, output: Option<&std::path::Path>) -> crate::Result<()> {
    let tensors = crate::parse_npz(path)?;
    let info = crate::inspect_npz(path)?;

    let output_path = if let Some(p) = output {
        p.to_owned()
    } else {
        // Replace extension: weights.npz → weights.safetensors
        let mut out = path.to_owned();
        out.set_extension("safetensors");
        out
    };

    println!(
        "Converting {} → {}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );
    println!(
        "  {} tensors, {}",
        info.tensors.len(),
        format_bytes(info.total_bytes)
    );

    crate::npz_to_safetensors(&tensors, &output_path)?;
    println!("  Done.");
    Ok(())
}

#[cfg(feature = "pth")]
fn run_remember_pth(path: &std::path::Path, output: Option<&std::path::Path>) -> crate::Result<()> {
    let parsed = crate::parse_pth(path)?;
    let info = parsed.inspect();

    let output_path = if let Some(p) = output {
        p.to_owned()
    } else {
        // Replace extension: model.pth → model.safetensors
        let mut out = path.to_owned();
        out.set_extension("safetensors");
        out
    };

    println!(
        "Converting {} → {}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );
    println!(
        "  {} tensors, {}",
        info.tensor_count,
        format_bytes(info.total_bytes)
    );

    parsed.to_safetensors(&output_path)?;
    println!("  Done.");
    Ok(())
}

#[cfg(feature = "gguf")]
fn run_parse_gguf(path: &std::path::Path) -> crate::Result<()> {
    let parsed = crate::parse_gguf(path)?;
    let info = parsed.inspect();
    let tensor_info = parsed.tensor_info();

    println!(
        "Parsed {} (GGUF v{})",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)"),
        info.version
    );
    if let Some(arch) = info.architecture.as_deref() {
        println!("  Arch:       {arch}");
    }
    println!("  Tensors:    {}", info.tensor_count);
    println!("  Total size: {}", format_bytes(info.total_bytes));
    let dtype_list: String = info
        .dtypes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    println!("  Dtypes:     {dtype_list}");
    println!("  Alignment:  {} bytes", info.alignment);
    println!();

    for t in tensor_info {
        let shape_str = format!("{:?}", t.shape);
        let byte_len_str = t.byte_len.map_or_else(|| "?".into(), format_bytes);
        println!(
            "  {:<40} {:<8} {:<15} {}",
            t.name, t.dtype, shape_str, byte_len_str
        );
    }
    Ok(())
}

/// `remember` for a `GGUF` input: dequantise every block-quantised tensor into
/// `target` and pass scalar tensors through in their source dtype.
///
/// Gained the `target` parameter in v0.7.4. Before that it hard-coded `BF16`,
/// which made `amn remember model.gguf --to f32` a rejection even though the
/// 24 `GGUF` kernels had been generic over `OutputElement` since v0.7.3.
///
/// **v0.7.6 moved the work into the library.** This function used to carry 121
/// lines that transcribed `convert`'s `GGUF` reader — shape reversal, dtype
/// mapping, dequant dispatch, `TensorView` assembly, `serialize_to_file` — and
/// the transcription was sequential where the library path is threaded, took no
/// `threads` argument so `--threads` was silently ignored here, and left a
/// binding with nothing to call. It is now a call to
/// `ParsedGguf::remember_with_options`, which is the same code
/// `convert --to safetensors` runs.
#[cfg(feature = "gguf")]
fn run_remember_gguf(
    path: &std::path::Path,
    output: Option<&std::path::Path>,
    target: TargetDtype,
    threads: Option<usize>,
) -> crate::Result<()> {
    let parsed = crate::parse_gguf(path)?;
    let info = parsed.inspect();

    let output_path = if let Some(p) = output {
        p.to_owned()
    } else {
        let mut out = path.to_owned();
        out.set_extension("safetensors");
        out
    };

    println!(
        "Converting {} → {}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );
    println!("  {} tensors", info.tensor_count);

    // `None` keeps the library's `min(cores, 4)` default; `Some(n)` is the
    // caller's `--threads`, clamped to at least 1 by the builder. Before v0.7.6
    // this arm had no thread budget at all, so `--threads` was accepted and
    // discarded.
    let opts = match threads {
        Some(n) => crate::RememberOptions::new().with_threads(n),
        None => crate::RememberOptions::new(),
    };
    parsed.remember_with_options(&output_path, target, opts)?;

    println!("  Output: {}", output_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// `convert` subcommand — a thin wrapper over `crate::convert`
// ---------------------------------------------------------------------------

/// Builds [`ConvertOptions`] from the `GGUF` metadata flags: the JSON file first,
/// then each `--gguf-kv` merged over it (so a one-off flag beats the file).
///
/// # Errors
///
/// Returns [`crate::AnamnesisError::Io`] if the metadata file cannot be read, and
/// [`crate::AnamnesisError::Parse`] if the JSON or a `key=value` is malformed.
#[cfg(feature = "gguf")]
fn build_convert_options(
    gguf_metadata: Option<&std::path::Path>,
    gguf_kv: &[String],
) -> crate::Result<ConvertOptions> {
    let mut metadata = match gguf_metadata {
        Some(file) => {
            let json = std::fs::read_to_string(file).map_err(crate::AnamnesisError::Io)?;
            crate::convert::parse_gguf_metadata_json(&json)?
        }
        None => std::collections::HashMap::new(),
    };
    for arg in gguf_kv {
        let (key, value) = crate::convert::parse_gguf_kv_arg(arg)?;
        metadata.insert(key, value);
    }
    Ok(ConvertOptions::new().with_gguf_metadata(metadata))
}

/// The `gguf`-less counterpart: the metadata flags have nowhere to go, so using
/// them is a clear error rather than a silent no-op.
///
/// # Errors
///
/// Returns [`crate::AnamnesisError::Unsupported`] if either flag was supplied.
#[cfg(not(feature = "gguf"))]
fn build_convert_options(
    gguf_metadata: Option<&std::path::Path>,
    gguf_kv: &[String],
) -> crate::Result<ConvertOptions> {
    if gguf_metadata.is_some() || !gguf_kv.is_empty() {
        return Err(crate::AnamnesisError::Unsupported {
            format: "--gguf-metadata/--gguf-kv".into(),
            detail: "GGUF metadata pass-through requires the `gguf` Cargo feature; \
                     rebuild with `cargo install anamnesis --features cli,gguf`"
                .into(),
        });
    }
    Ok(ConvertOptions::new())
}

/// Runs the `convert` subcommand: parses the `--to` target, derives an output
/// path when `-o` is omitted, collects any caller-supplied `GGUF` metadata, and
/// delegates the whole `(input × target)` dispatch to [`crate::convert::convert`].
/// Parses `--out-dtype` into the element type the dequantised tensors get.
///
/// Accepts exactly the three dequantisation output widths, case-insensitively.
/// Deliberately **not** an `impl FromStr for Dtype`: `Dtype` names 15 element
/// types, and a `FromStr` on it would advertise that `--out-dtype i64` is a
/// meaningful request. The error string lists what is actually accepted, in the
/// same wording `TargetDtype`'s parser uses on the `remember` side.
///
/// # Errors
///
/// Returns [`AnamnesisError::Unsupported`] if `s` is not `bf16`, `f32` or
/// `f16`.
fn parse_out_dtype(s: &str) -> crate::Result<crate::Dtype> {
    match s.to_ascii_lowercase().as_str() {
        "bf16" => Ok(crate::Dtype::BF16),
        "f32" => Ok(crate::Dtype::F32),
        "f16" => Ok(crate::Dtype::F16),
        other => Err(crate::AnamnesisError::Unsupported {
            format: other.to_owned(),
            detail: "supported output dtypes: bf16, f32, f16".to_owned(),
        }),
    }
}

fn run_convert(
    path: &std::path::Path,
    to: &str,
    output: Option<&std::path::Path>,
    gguf_metadata: Option<&std::path::Path>,
    gguf_kv: &[String],
    out_dtype: &str,
    threads: Option<usize>,
) -> crate::Result<()> {
    let target = ConvertTarget::parse(to)?;
    let dequant_dtype = parse_out_dtype(out_dtype)?;
    let options = build_convert_options(gguf_metadata, gguf_kv)?.with_output_dtype(dequant_dtype);
    // `None` keeps the library's `min(cores, 4)` default; `Some(n)` is the
    // caller's `--threads`, clamped to at least 1 by the builder.
    let options = match threads {
        Some(n) => options.with_threads(n),
        None => options,
    };
    let output_path = output.map_or_else(
        // Name the file after the dtype it will actually hold: `--out-dtype
        // f32` must not derive `model-bf16.safetensors`.
        || crate::convert::derive_output_path_for_dtype(path, target, dequant_dtype),
        Path::to_owned,
    );

    println!(
        "Converting {} -> {}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(input)"),
        output_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(output)")
    );

    let stats = crate::convert::convert(path, target, &output_path, &options)?;

    if stats.dequantized > 0 {
        // Report the dtype actually written, not a hard-coded "BF16": since
        // v0.7.3 that is a caller choice, and a line claiming BF16 while the
        // file holds F32 would be worse than no line at all.
        println!("  {} dequantized to {dequant_dtype}", stats.dequantized);
    }
    if stats.quantized > 0 {
        println!(
            "  {} quantized to NF4, {} passed through as BF16",
            stats.quantized, stats.passthrough
        );
    }
    println!(
        "  Wrote {} tensors -> {}",
        stats.tensors,
        output_path.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Output path derivation
// ---------------------------------------------------------------------------

/// Derive an output path from the input path and target dtype.
///
/// `model-fp8.safetensors`  → `model-bf16.safetensors`
/// `model-GPTQ-Int4.safetensors` → `model-bf16.safetensors`
/// `weights.safetensors`    → `weights-bf16.safetensors`
///
/// Shares the quantisation-suffix table with `convert` via
/// [`crate::convert::strip_quant_suffix`], so the two derivations cannot drift.
fn derive_output_path(input: &std::path::Path, target: TargetDtype) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let suffix = target.to_string().to_lowercase();
    let new_name = format!(
        "{}-{suffix}.safetensors",
        crate::convert::strip_quant_suffix(stem)
    );
    input
        .parent()
        .map_or_else(|| PathBuf::from(&new_name), |p| p.join(&new_name))
}
