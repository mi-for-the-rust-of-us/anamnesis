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
4. Update `CHANGELOG.md` — add a bullet under the `[Unreleased]` section for any user-visible change (new feature, fix, breaking change). Follow [Keep a Changelog](https://keepachangelog.com/) categories: Added, Changed, Fixed, Removed.

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
   cargo doc --all-features --no-deps;
   $env:RUSTDOCFLAGS = $null;
   cargo publish --dry-run --allow-dirty
   ```
   `--allow-dirty` is required because step 4 runs **before** the bump commit (steps 1–3 leave `Cargo.toml`/`Cargo.lock`/`CHANGELOG.md` uncommitted by design). The real publish workflow runs against a tagged commit and never sees a dirty tree. If `cargo publish --dry-run` flags issues, fix them in-place before creating the bump commit.
5. Commit as `bump version to vX.Y.Z, update changelog date`
6. Push the commit, wait for CI to go GREEN
7. `git tag vX.Y.Z && git push origin vX.Y.Z`
8. Wait for the publish workflow to go GREEN. Since v0.7.3 it does **two**
   things: `cargo publish`, then `gh release create` for the tag. The Release
   is what carries the test corpus, because `Cargo.toml`'s `exclude` keeps
   `tests/` out of the published crate (0.19 MiB instead of 4.8 MiB), and
   GitHub's per-tag source tarball ships it verbatim.
9. Check the Release actually appeared and its notes are the right section.
   The workflow slices them out of `CHANGELOG.md` by matching
   `## [X.Y.Z]`, and **fails the job** if no section matches — which is the
   intended alarm for "step 3 was skipped and `## [Unreleased]` was never
   renamed". Release creation runs *after* `cargo publish` on purpose, so a
   Release can never advertise a version whose publish failed.

**Two checks specific to the packaging split**, worth running before the tag:

- `cargo package --list | grep '^tests/'` must print **nothing**. If it does,
  the `exclude` regressed and the crate is about to grow ~25×.
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
