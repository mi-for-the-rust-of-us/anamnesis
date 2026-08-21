# Claude Code Instructions

## Coding Conventions

Always apply the rules in `CONVENTIONS.md` to all code changes. Every annotation pattern, doc-comment rule, and style rule in that file is mandatory.

Every `.rs` file must start with `// SPDX-License-Identifier: MIT OR Apache-2.0` as its first line.

## Pre-commit Checks

Before every commit, run and fix any issues from:
1. `cargo build --features cli` (ensures CLI binary is current before integration tests)
2. `cargo fmt`
3. `cargo clippy --all-targets --all-features -- -D warnings`
4. `cargo test`
5. **If the commit touches any `///` or `//!` comment**, run the rustdoc sweep — see [Documentation Checks](#documentation-checks). `--all-features` alone cannot see a link that breaks under an intermediate feature combination, and a public-items run cannot see a broken link on a `pub(crate)` item at all.
6. Update `CHANGELOG.md` — add a bullet under the `[Unreleased]` section for any user-visible change (new feature, fix, breaking change). Follow [Keep a Changelog](https://keepachangelog.com/) categories: Added, Changed, Fixed, Removed.

## Documentation Checks

Rustdoc is checked in CI by the `docs` job in `.github/workflows/ci.yml`, which
runs **twelve feature combinations** with `RUSTDOCFLAGS="-D warnings"` and
`--document-private-items`. Both halves are load-bearing, and each was added
because the missing half had already let breakage through:

- **Feature combinations.** An intra-doc link naming a feature-gated item
  resolves under `--all-features` and fails under a combination that gates the
  item out. `--all-features` cannot see this class by construction.
- **`--document-private-items`.** A broken link on a `pub(crate)` item is
  invisible to a public-items run. Phase 7.6 left three behind in its own
  refactor and found five more that predated it.

To reproduce the CI job locally, run the same loop it runs:

```powershell
$env:RUSTDOCFLAGS = "-D warnings"
cargo doc --no-deps --document-private-items
foreach ($c in @(
    "--all-features",
    "--no-default-features",
    "--no-default-features --features npz",
    "--no-default-features --features pth",
    "--no-default-features --features gguf",
    "--no-default-features --features bnb",
    "--no-default-features --features gptq",
    "--no-default-features --features awq",
    "--no-default-features --features cli",
    "--no-default-features --features ollama",
    "--features cli,gguf,npz,pth")) {
    Write-Host "cargo doc $c"
    cargo doc --no-deps --document-private-items $c.Split(" ")
    if ($LASTEXITCODE -ne 0) { throw "rustdoc failed for: $c" }
}
$env:RUSTDOCFLAGS = $null
```

**When a link genuinely cannot resolve in every configuration** — an optional
dependency such as `clap`, a feature-gated item referenced from an always-on
one, or anything in a `pub(crate)` module's own `//!` header, which `rustdoc`
will not resolve even fully qualified — use a plain code span rather than link
syntax, per `CONVENTIONS.md` § *Intra-Doc Link Safety*, and say why in a
comment.

## Performance Changes

If a commit claims a perf win (faster, less memory, fewer allocations, fewer branches), it must include a measurement, not just an analysis:

1. **Best-of-5 release-mode median**, with `target-cpu=native`, on a real fixture the claim is about. Templates: `tests/bench_npz_adhoc.rs` and `tests/bench_pth_adhoc.rs` — each is gated `#[ignore]` and run with `cargo test --release --features <flag> --test <name> <test_fn> -- --nocapture --ignored`.
2. **Record both before and after numbers in the commit message** — median + range (min/max), and the bench command used. This is what makes a regression reversible: the next reviewer (or the next person to read `git log`) can re-run the same bench against the parent commit and know the answer.
3. **If the measurement does not show a win in the expected direction, do not commit.** Estimates and asymptotic arguments are hypotheses, not data — see `5f2632b` ("Revert NPZ memset elimination") for the cautionary case where a confidently estimated `~30 %` saving turned out to be a measured `~33 %` regression.

This rule applies to perf-claim commits only. Correctness fixes, refactors, doc changes, and feature additions do not need a measurement to ship.

Before proposing a perf-claim change, **read [`docs/perf-experiments.md`](docs/perf-experiments.md)** — it catalogs hypotheses already tested and their measured outcomes (some confirmed, some rejected, some contradicting their original CHANGELOG claims). This avoids re-litigating the same ideas. When an experiment is shipped or attempted, add a row to that file's index plus a section with method + numbers, even if the result is "no change" or a regression.

## Release Checklist

Before tagging a release (`v*`), complete these steps in order:
1. Bump `version` in `Cargo.toml` to match the tag (e.g., `"0.4.0"` for `v0.4.0`)
2. Run `cargo check` to update `Cargo.lock`
3. Rename `## [Unreleased]` in `CHANGELOG.md` to `## [X.Y.Z] - YYYY-MM-DD`
4. **Dry-run the publish workflow locally** — runs the same gauntlet `.github/workflows/publish.yml` runs, plus `cargo publish --dry-run`. The dry-run catches packaging issues that the regular CI does not exercise: missing `Cargo.toml` metadata (`license`, `description`, `repository`, `readme`, `keywords`, `categories`), files referenced by `include`/`exclude` that don't exist, the 10 MiB published-tarball cap, or version-already-on-registry conflicts. Every step must succeed before committing the version bump:
   ```powershell
   cargo fmt --check;
   cargo clippy --all-targets -- -D warnings;
   cargo clippy --all-targets --all-features -- -D warnings;
   cargo test --all-features;
   $env:RUSTDOCFLAGS = "-D warnings";
   cargo doc --all-features --no-deps --document-private-items;
   $env:RUSTDOCFLAGS = $null;
   cargo publish --dry-run --allow-dirty
   ```
   `--allow-dirty` is required because step 4 runs **before** the bump commit (steps 1–3 leave `Cargo.toml`/`Cargo.lock`/`CHANGELOG.md` uncommitted by design). The real publish workflow runs against a tagged commit and never sees a dirty tree. If `cargo publish --dry-run` flags issues, fix them in-place before creating the bump commit.

   **Also run the gauntlet on the MSRV toolchain, not just stable.** The gauntlet above uses whatever `cargo` is on `PATH`, but CI has a separate MSRV job, and the two toolchains do not lint identically:
   ```powershell
   rustup run 1.88 cargo clippy --all-targets -- -D warnings;
   rustup run 1.88 cargo clippy --all-targets --all-features -- -D warnings;
   rustup run 1.88 cargo clippy --all-targets --no-default-features -- -D warnings;
   rustup run 1.88 cargo test --all-features
   ```
   This is not hypothetical: v0.7.3 pushed a green-on-stable `main` that failed CI on MSRV, because rustc 1.88's dead-code analysis does not count a reference from a `const _: () = { assert!(…) }` block as a use, while current stable does. Note the **`--no-default-features` and default-features runs specifically** — a `pub(crate)` item consumed only from a feature-gated module is live under `--all-features` and dead without it, so an all-features-only sweep cannot see it.
5. Commit as `bump version to vX.Y.Z, update changelog date`
6. Push the commit, wait for CI to go GREEN
7. `git tag vX.Y.Z && git push origin vX.Y.Z`
8. Wait for the publish workflow to go GREEN. Since v0.7.3 it does **two**
   things: `cargo publish`, then `gh release create` for the tag. The Release
   is what carries the test corpus, because `Cargo.toml`'s `exclude` keeps
   `tests/` out of the published crate (0.60 MiB instead of 4.8 MiB), and
   GitHub's per-tag source tarball ships it verbatim.
9. Check the Release actually appeared and its notes are the right section.
   If the job failed *after* `cargo publish` succeeded, **do not re-run the
   workflow** — the publish step would fail on the already-taken version and
   mask the real error. Fix the workflow for next time, then create the missing
   Release by hand from a clean checkout at the tag:
   ```powershell
   $v = "0.7.3"
   awk -v hdr="## [$v]" 'index($0,hdr)==1{f=1;next} f&&/^## \[/{exit} f{print}' CHANGELOG.md > release-body.md
   # append the "Verifying the correctness claims" footer, then:
   gh release create "v$v" --title "v$v" --notes-file release-body.md --verify-tag
   ```
   The workflow slices them out of `CHANGELOG.md` by matching
   `## [X.Y.Z]`, and **fails the job** if no section matches — which is the
   intended alarm for "step 3 was skipped and `## [Unreleased]` was never
   renamed". Release creation runs *after* `cargo publish` on purpose, so a
   Release can never advertise a version whose publish failed.

**Two checks specific to the packaging split**, worth running before the tag:

- `cargo package --list | grep '^tests/'` must print **nothing**. If it does,
  the `exclude` regressed and the crate is about to grow ~8×.
- `scripts/verify-claims.sh` (or `.ps1`) should pass from a clean checkout.
  That is the path the README points a consumer at for verifying the
  correctness claims, so it needs to work at the tag, not just on your
  machine.

**Never tag before bumping `Cargo.toml`** — `cargo publish` will reject the crate if the version in the registry already exists.

## Shell Environment

The user runs PowerShell on Windows. Use PowerShell syntax for all suggested commands:
- Use `$env:VAR="value";` instead of `VAR=value` for environment variables
- Use semicolons to chain commands, not `&&`
- Use forward slashes in paths when running Rust/cargo commands
