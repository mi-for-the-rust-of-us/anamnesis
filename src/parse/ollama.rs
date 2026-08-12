// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Ollama` model-cache path resolver.
//!
//! Turns an `Ollama`-style model reference like `llama3.2:1b` (or the
//! prefixed `ollama:llama3.2:1b`) into the on-disk `GGUF` blob path,
//! by reading the `Ollama` manifest at
//! `<root>/manifests/registry.ollama.ai/library/<name>/<tag>` and
//! following its `application/vnd.ollama.image.model` layer's `digest`
//! to `<root>/blobs/sha256-<hash>`.
//!
//! `<root>` is the `OLLAMA_MODELS` environment variable if set,
//! otherwise `$HOME/.ollama/models` (the default on macOS / Linux) or
//! `%USERPROFILE%\.ollama\models` (the default on Windows).
//!
//! The resolver is **pure path arithmetic plus a single `JSON` read**:
//! no network, no `Go` interop, no `ollama` CLI shell-out. Once
//! resolved, the returned `PathBuf` is a regular `GGUF` file ready to
//! be passed to `parse_gguf` / `inspect_gguf_from_reader` / the
//! `amn inspect` CLI / etc.
//!
//! Feature-gated behind `ollama` (which itself implies `gguf`).
//!
//! # Layout reference
//!
//! `Ollama` 0.24.0 caches models under (paths shown for Unix; Windows
//! uses `\` instead of `/` but the structure is identical):
//!
//! ```text
//! ~/.ollama/models/
//!   manifests/
//!     registry.ollama.ai/
//!       library/
//!         <model>/
//!           <tag>                   <-- single-line JSON manifest
//!   blobs/
//!     sha256-<hex64>                <-- the GGUF blob, plus templates / licenses
//! ```
//!
//! The manifest JSON has the Docker-style shape:
//!
//! ```json
//! {
//!   "schemaVersion": 2,
//!   "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
//!   "config": { "digest": "sha256:...", "size": ... },
//!   "layers": [
//!     {"mediaType": "application/vnd.ollama.image.model",    "digest": "sha256:...", "size": ...},
//!     {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:...", "size": ...},
//!     {"mediaType": "application/vnd.ollama.image.license",  "digest": "sha256:...", "size": ...}
//!   ]
//! }
//! ```
//!
//! Only the `application/vnd.ollama.image.model` layer's `digest`
//! matters for the path resolver; the rest is ignored.

use std::path::{Path, PathBuf};

use crate::error::AnamnesisError;

/// Registry host segment in `Ollama`'s manifest tree. Hard-coded here
/// because `Ollama` 0.x consistently caches under
/// `registry.ollama.ai/library/<model>/<tag>` regardless of how the
/// model was pulled — the host is part of the on-disk layout, not the
/// network URL.
const REGISTRY_HOST: &str = "registry.ollama.ai";

/// Registry namespace segment. Both first-party and community models on
/// `ollama.com` land under `library/` on disk.
const REGISTRY_NAMESPACE: &str = "library";

/// The `mediaType` discriminator that identifies the `GGUF` blob layer
/// inside a manifest. Other layers (`*.image.template`,
/// `*.image.license`, `*.image.params`) point at non-tensor blobs we
/// do not want.
const MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

/// Default `Ollama` tag when the caller passes a bare model name with
/// no tag. Matches `ollama pull <model>`'s default behaviour, which
/// resolves to the `latest` tag on the registry.
const DEFAULT_TAG: &str = "latest";

/// Optional URL scheme prefix that the `amn` CLI accepts: e.g.,
/// `amn inspect ollama:llama3.2:1b`. The resolver also accepts the
/// bare `<name>:<tag>` form (without the prefix) for callers using the
/// library API directly.
const URL_SCHEME_PREFIX: &str = "ollama:";

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Resolves an `Ollama` model spec to the local `GGUF` blob path.
///
/// `spec` accepts three forms:
///
/// - `"<model>:<tag>"` — explicit tag (e.g., `"llama3.2:1b"`).
/// - `"<model>"` — defaults to the `latest` tag, matching
///   `ollama pull <model>`'s behaviour.
/// - `"ollama:<model>:<tag>"` — the URL-scheme form used by the
///   `amn inspect ollama:llama3.2:1b` CLI surface. The
///   `ollama:` prefix is stripped before parsing the rest.
///
/// The model `name` may contain dots (e.g., `llama3.2`); the `tag` may
/// not — the **last** colon separates name from tag.
///
/// # Errors
///
/// Returns [`AnamnesisError::Parse`] when `spec` is empty, when the
/// manifest `JSON` is malformed, when the manifest lacks an
/// `application/vnd.ollama.image.model` layer, or when the layer's
/// `digest` is not in the expected `sha256:<hex>` form.
///
/// Returns [`AnamnesisError::Io`] when the manifest file or the
/// resolved blob file does not exist on disk (typically because the
/// model has not been `ollama pull`-ed yet) or cannot be read.
pub fn resolve_ollama_model(spec: &str) -> crate::Result<PathBuf> {
    let (model_name, tag) = parse_spec(spec)?;
    let root = ollama_models_root();
    let manifest_path = manifest_path_for(&root, model_name, tag);
    // EXHAUSTIVE: `io::ErrorKind` is foreign `#[non_exhaustive]` — we
    // only customise the `NotFound` message (the most common case
    // when the model has not been pulled) and pass every other kind
    // through unchanged.
    #[allow(clippy::wildcard_enum_match_arm)]
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => AnamnesisError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Ollama manifest not found at {} (run `ollama pull {model_name}:{tag}` first)",
                manifest_path.display()
            ),
        )),
        _ => AnamnesisError::Io(e),
    })?;
    let blob_hash = parse_model_digest(&manifest_bytes)?;
    let blob_path = blob_path_for(&root, &blob_hash);
    if !blob_path.exists() {
        return Err(AnamnesisError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Ollama blob not found at {} (manifest references it but the blob is absent — \
                 the cache may be partially populated)",
                blob_path.display()
            ),
        )));
    }
    Ok(blob_path)
}

// ---------------------------------------------------------------------------
// Path building (pure, testable in isolation)
// ---------------------------------------------------------------------------

/// Returns the root directory the resolver looks under.
///
/// Honours `OLLAMA_MODELS` (`Ollama`'s standard override), then falls
/// back to `$HOME/.ollama/models` on Unix or `%USERPROFILE%\.ollama\models`
/// on Windows. Anamnesis adds no `dirs`-crate dependency for this —
/// `std::env::home_dir` was un-deprecated in Rust 1.85 and is available
/// on the crate's MSRV (1.88).
fn ollama_models_root() -> PathBuf {
    if let Ok(explicit) = std::env::var("OLLAMA_MODELS")
        && !explicit.is_empty()
    {
        return PathBuf::from(explicit);
    }
    if let Some(home) = std::env::home_dir() {
        return home.join(".ollama").join("models");
    }
    // Last-ditch fallback. If neither `OLLAMA_MODELS` nor a home
    // directory is reachable, return a relative `.ollama/models` and
    // let the downstream file-not-found error surface a clear message.
    PathBuf::from(".ollama").join("models")
}

fn manifest_path_for(root: &Path, model_name: &str, tag: &str) -> PathBuf {
    root.join("manifests")
        .join(REGISTRY_HOST)
        .join(REGISTRY_NAMESPACE)
        .join(model_name)
        .join(tag)
}

fn blob_path_for(root: &Path, blob_hash: &str) -> PathBuf {
    root.join("blobs").join(format!("sha256-{blob_hash}"))
}

// ---------------------------------------------------------------------------
// Spec parsing (pure, testable in isolation)
// ---------------------------------------------------------------------------

/// Splits an `Ollama` model spec into `(model_name, tag)`.
///
/// Accepts `ollama:<rest>`, `<model>:<tag>`, or `<model>` (defaulting
/// the tag to `latest`). The **last** `:` separates name from tag —
/// model names containing dots (e.g., `llama3.2`) are preserved as the
/// name component.
fn parse_spec(spec: &str) -> crate::Result<(&str, &str)> {
    if spec.is_empty() {
        return Err(AnamnesisError::Parse {
            reason: "Ollama model spec is empty".into(),
        });
    }
    let cleaned = spec.strip_prefix(URL_SCHEME_PREFIX).unwrap_or(spec);
    if cleaned.is_empty() {
        return Err(AnamnesisError::Parse {
            reason: format!(
                "Ollama model spec {spec:?} has no model name after the `ollama:` prefix"
            ),
        });
    }
    // `rsplit_once(':')` returns the right-most split, preserving model
    // names that contain dots (e.g., `llama3.2:1b` splits into
    // `("llama3.2", "1b")`).
    match cleaned.rsplit_once(':') {
        Some(("", _)) => Err(AnamnesisError::Parse {
            reason: format!("Ollama model spec {spec:?} has an empty model name"),
        }),
        Some((_, "")) => Err(AnamnesisError::Parse {
            reason: format!("Ollama model spec {spec:?} has an empty tag"),
        }),
        Some((name, tag)) => Ok((name, tag)),
        None => Ok((cleaned, DEFAULT_TAG)),
    }
}

// ---------------------------------------------------------------------------
// Manifest JSON parsing (pure, testable in isolation)
// ---------------------------------------------------------------------------

/// Parses the manifest bytes, returns the hex digest (no `sha256:`
/// prefix) of the model-layer blob.
///
/// Looks for a layer whose `mediaType` equals
/// `application/vnd.ollama.image.model` and reads its `digest` field.
/// Other layers (`*.template`, `*.license`, `*.params`) are ignored.
fn parse_model_digest(manifest_bytes: &[u8]) -> crate::Result<String> {
    let manifest: serde_json::Value =
        serde_json::from_slice(manifest_bytes).map_err(|e| AnamnesisError::Parse {
            reason: format!("failed to parse Ollama manifest as JSON: {e}"),
        })?;
    let layers = manifest
        .get("layers")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "Ollama manifest has no `layers` array".into(),
        })?;
    let model_layer = layers
        .iter()
        .find(|layer| {
            layer.get("mediaType").and_then(serde_json::Value::as_str) == Some(MODEL_MEDIA_TYPE)
        })
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!("Ollama manifest has no layer with mediaType `{MODEL_MEDIA_TYPE}`"),
        })?;
    let digest = model_layer
        .get("digest")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| AnamnesisError::Parse {
            reason: "Ollama manifest model layer has no `digest` string".into(),
        })?;
    let hash = digest
        .strip_prefix("sha256:")
        .ok_or_else(|| AnamnesisError::Parse {
            reason: format!(
                "Ollama manifest model layer digest {digest:?} is not a `sha256:` digest"
            ),
        })?;
    if hash.is_empty() {
        return Err(AnamnesisError::Parse {
            reason: "Ollama manifest model layer digest is empty after `sha256:` prefix".into(),
        });
    }
    Ok(hash.to_owned())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::wildcard_enum_match_arm
)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // parse_spec — pure, no filesystem touch
    // -----------------------------------------------------------------------

    #[test]
    fn parse_spec_explicit_tag() {
        let (name, tag) = parse_spec("llama3.2:1b").unwrap();
        assert_eq!(name, "llama3.2");
        assert_eq!(tag, "1b");
    }

    #[test]
    fn parse_spec_url_scheme_prefix() {
        let (name, tag) = parse_spec("ollama:llama3.2:1b").unwrap();
        assert_eq!(name, "llama3.2");
        assert_eq!(tag, "1b");
    }

    #[test]
    fn parse_spec_no_tag_defaults_to_latest() {
        let (name, tag) = parse_spec("llama3.2").unwrap();
        assert_eq!(name, "llama3.2");
        assert_eq!(tag, "latest");
    }

    #[test]
    fn parse_spec_url_scheme_no_tag_defaults_to_latest() {
        let (name, tag) = parse_spec("ollama:qwen2.5-coder").unwrap();
        assert_eq!(name, "qwen2.5-coder");
        assert_eq!(tag, "latest");
    }

    #[test]
    fn parse_spec_empty_rejected() {
        let err = parse_spec("").unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert!(reason.contains("empty")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn parse_spec_only_prefix_rejected() {
        let err = parse_spec("ollama:").unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert!(reason.contains("no model name")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn parse_spec_empty_tag_rejected() {
        let err = parse_spec("llama3.2:").unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert!(reason.contains("empty tag")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn parse_spec_empty_name_rejected() {
        // `:1b` — empty model name before the colon.
        let err = parse_spec(":1b").unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert!(reason.contains("empty model name")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // parse_model_digest — pure, no filesystem touch
    // -----------------------------------------------------------------------

    #[test]
    fn parse_model_digest_typical_manifest() {
        // Real-world manifest shape, matches Ollama 0.24.0 llama3.2:1b.
        let manifest = br#"{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:4f659a1e86d7f5a33c389f7991e7224b7ee6ad0358b53437d54c02d2e1b1118d","size":485},"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45","size":1321082688},{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:966de95ca8a62200913e3f8bfbf84c8494536f1b94b49166851e76644e966396","size":1429},{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:fcc5a6bec9daf9b561a68827b67ab6088e1dba9d1fa2a50d7bbcc8384e0a265d","size":7711}]}"#;
        let hash = parse_model_digest(manifest).unwrap();
        assert_eq!(
            hash,
            "74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45"
        );
    }

    #[test]
    fn parse_model_digest_layer_order_irrelevant() {
        // Same manifest, but the model layer is third instead of first.
        let manifest = br#"{"layers":[{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:aaaa","size":1},{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:bbbb","size":1},{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:cccc","size":1}]}"#;
        let hash = parse_model_digest(manifest).unwrap();
        assert_eq!(hash, "cccc");
    }

    #[test]
    fn parse_model_digest_rejects_missing_layers_array() {
        let manifest = br#"{"schemaVersion":2}"#;
        let err = parse_model_digest(manifest).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert!(reason.contains("`layers` array")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn parse_model_digest_rejects_no_model_layer() {
        let manifest = br#"{"layers":[{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:aa","size":1}]}"#;
        let err = parse_model_digest(manifest).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert!(reason.contains("vnd.ollama.image.model")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn parse_model_digest_rejects_non_sha256_digest() {
        let manifest = br#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"md5:dead","size":1}]}"#;
        let err = parse_model_digest(manifest).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert!(reason.contains("not a `sha256:`")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn parse_model_digest_rejects_empty_digest_after_prefix() {
        let manifest = br#"{"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:","size":1}]}"#;
        let err = parse_model_digest(manifest).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert!(reason.contains("empty after")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn parse_model_digest_rejects_malformed_json() {
        let manifest = b"not json at all {";
        let err = parse_model_digest(manifest).unwrap_err();
        match err {
            AnamnesisError::Parse { reason } => assert!(reason.contains("failed to parse")),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // Path construction — pure
    // -----------------------------------------------------------------------

    #[test]
    fn manifest_path_layout() {
        let root = PathBuf::from("/x/ollama/models");
        let p = manifest_path_for(&root, "llama3.2", "1b");
        // Use the OS-portable join order; on Windows the separator is
        // `\` but the component sequence is the same.
        let expected = PathBuf::from("/x/ollama/models")
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join("llama3.2")
            .join("1b");
        assert_eq!(p, expected);
    }

    #[test]
    fn blob_path_layout() {
        let root = PathBuf::from("/x/ollama/models");
        let p = blob_path_for(&root, "deadbeef");
        let expected = PathBuf::from("/x/ollama/models")
            .join("blobs")
            .join("sha256-deadbeef");
        assert_eq!(p, expected);
    }

    // -----------------------------------------------------------------------
    // End-to-end smoke (gated on the local Ollama cache being populated)
    // -----------------------------------------------------------------------

    /// Resolves the cached `llama3.2:1b` blob if the local Ollama cache
    /// has it, asserts the resolved path exists and starts with the
    /// `GGUF` magic. Skipped silently if Ollama is not installed or
    /// the model has not been pulled — keeps the test usable on CI
    /// runners without an Ollama cache.
    #[test]
    fn resolve_llama_3_2_1b_smoke() {
        use std::io::Read as _;

        let blob = match resolve_ollama_model("llama3.2:1b") {
            Ok(p) => p,
            Err(AnamnesisError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("  skipping: Ollama cache does not contain llama3.2:1b ({e})");
                return;
            }
            Err(other) => panic!("unexpected resolver error: {other:?}"),
        };
        assert!(blob.exists(), "resolved blob {blob:?} does not exist");
        // Confirm it is a GGUF: first 4 bytes are the magic `b"GGUF"`.
        let mut buf = [0u8; 4];
        let mut file = std::fs::File::open(&blob).unwrap();
        file.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"GGUF", "resolved blob {blob:?} is not a GGUF file");
    }

    #[test]
    fn resolve_unknown_model_returns_not_found() {
        let err = resolve_ollama_model("definitely-not-a-real-model:nope").unwrap_err();
        match err {
            AnamnesisError::Io(e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
                let msg = e.to_string();
                assert!(msg.contains("manifest"), "unexpected message: {msg}");
            }
            other => panic!("expected Io NotFound, got {other:?}"),
        }
    }
}
