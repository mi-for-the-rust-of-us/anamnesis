// SPDX-License-Identifier: MIT OR Apache-2.0

//! Output files that a failure can never leave half-written, and the check
//! that an output is not the input.
//!
//! Until v0.7.9 the `GGUF` and `BnB-NF4` writers opened their destination with
//! `File::create`, truncating it before anything was validated or written, so
//! any later error left an empty or partial file where the old one had been.
//! `amn convert model.gguf --to gguf -o model.gguf` with a bad `--gguf-metadata`
//! alignment left the **input** at 0 bytes (Phase 7.9, audit finding M-8). The
//! `safetensors` writers were already atomic, through the upstream crate's
//! temp-file-and-rename; every writer now is.

#[cfg(any(feature = "gguf", feature = "bnb"))]
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::AnamnesisError;

/// Capacity of the buffer between a writer and its temporary file. 64 `KiB`
/// matches the buffer the `GGUF` parser reads through, so the syscall
/// amortisation is the same on both sides (the value the `GGUF` writer used
/// before it moved here).
#[cfg(any(feature = "gguf", feature = "bnb"))]
const WRITER_BUF_SIZE: usize = 64 * 1024;

/// Writes `path` by way of a temporary file in the same directory, which is
/// renamed over `path` only after `write` has succeeded and the buffer has
/// been flushed.
///
/// Until that rename, `path` is untouched: a validation error, a cancellation,
/// a full disk or a crash leaves the previous file (or no file) in place, never
/// a truncated one, and the temporary file is removed. The rename replaces a
/// symbolic link at `path` rather than writing through it, and gives a
/// hard-linked `path` a new file instead of rewriting the one its other names
/// share. On Unix the result is created with mode `0600`, as the
/// `safetensors` outputs already were.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] if the temporary file cannot be created,
/// written, flushed or renamed, and whatever error `write` returns.
// The writers that go through here (`GGUF`, `BnB-NF4`) exist only with their
// features; the `safetensors` writers are atomic upstream.
#[cfg(any(feature = "gguf", feature = "bnb"))]
pub(crate) fn write_atomically<F>(path: &Path, write: F) -> crate::Result<()>
where
    F: FnOnce(&mut BufWriter<&mut std::fs::File>) -> crate::Result<()>,
{
    // `Path::parent` is `Some("")` for a bare file name: the current directory.
    let parent = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let mut temp = tempfile::Builder::new()
        .prefix(".anamnesis-")
        .suffix(".tmp")
        .tempfile_in(parent)
        .map_err(AnamnesisError::Io)?;
    {
        let mut writer = BufWriter::with_capacity(WRITER_BUF_SIZE, temp.as_file_mut());
        write(&mut writer)?;
        writer.flush().map_err(AnamnesisError::Io)?;
    }
    temp.persist(path)
        .map_err(|e| AnamnesisError::Io(e.error))?;
    Ok(())
}

/// Refuses an `output` that is the same file as `input`.
///
/// Reading an input while its path is being replaced is at best a wasted run
/// and at worst, for a memory-mapped input on a platform that allows it, a
/// crash. The comparison follows symbolic links, and on Unix it compares
/// device and inode, so a hard link to the input is recognised too. An output
/// that does not exist yet is never the input.
///
/// # Errors
///
/// Returns [`AnamnesisError::Io`] with kind `InvalidInput` if the two are the
/// same file, or if `input`'s metadata cannot be read.
pub(crate) fn refuse_same_file(input: &Path, output: &Path) -> crate::Result<()> {
    if std::fs::metadata(output).is_err() {
        return Ok(());
    }
    if same_file(input, output).map_err(AnamnesisError::Io)? {
        return Err(AnamnesisError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "output `{}` is the input file; write to a different path",
                output.display()
            ),
        )));
    }
    Ok(())
}

/// Whether two existing paths name one file: same device and inode.
#[cfg(unix)]
fn same_file(a: &Path, b: &Path) -> std::io::Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let (a, b) = (std::fs::metadata(a)?, std::fs::metadata(b)?);
    Ok(a.dev() == b.dev() && a.ino() == b.ino())
}

/// Whether two existing paths name one file: same canonical path.
///
/// There is no stable file identity in `std` on this platform. Canonical paths
/// catch the same file reached through another spelling or a symbolic link; a
/// hard link is not caught, and the atomic write keeps it harmless, since the
/// input keeps its data.
#[cfg(not(unix))]
fn same_file(a: &Path, b: &Path) -> std::io::Result<bool> {
    Ok(std::fs::canonicalize(a)? == std::fs::canonicalize(b)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[cfg(any(feature = "gguf", feature = "bnb"))]
    #[test]
    fn a_failed_write_leaves_the_previous_file_and_no_temporary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.bin");
        std::fs::write(&path, b"previous").unwrap();
        let result = write_atomically(&path, |w| {
            w.write_all(b"partial").map_err(AnamnesisError::Io)?;
            Err(AnamnesisError::Parse {
                reason: "validation failed".into(),
            })
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"previous");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[cfg(any(feature = "gguf", feature = "bnb"))]
    #[test]
    fn a_successful_write_replaces_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.bin");
        std::fs::write(&path, b"previous").unwrap();
        write_atomically(&path, |w| w.write_all(b"new").map_err(AnamnesisError::Io)).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn the_input_is_refused_as_output_under_any_spelling() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("model.bin");
        std::fs::write(&input, b"x").unwrap();
        let other = dir.path().join("other.bin");
        let dotted = dir.path().join(".").join("model.bin");
        assert!(refuse_same_file(&input, &input).is_err());
        assert!(refuse_same_file(&input, &dotted).is_err());
        assert!(refuse_same_file(&input, &other).is_ok());
        std::fs::write(&other, b"y").unwrap();
        assert!(refuse_same_file(&input, &other).is_ok());
    }
}
