# Claude Code Instructions

## Coding Conventions

Always apply the rules in `CONVENTIONS.md` to all code changes. Every annotation pattern, doc-comment rule, and style rule in that file is mandatory.

Every `.rs` file must start with `// SPDX-License-Identifier: MIT OR Apache-2.0` as its first line.

## Pre-commit Checks

Before every commit, run and fix any issues from:
1. `cargo fmt`
2. `cargo clippy --all-targets --all-features -- -D warnings`
3. `cargo test`, plus `cargo test --all-features` (what CI runs) when the change can reach the CLI. `tests/cli.rs` and `tests/cli_convert.rs` are gated on the `cli` feature, so a default-feature run skips them. No separate `cargo build` step is needed: Cargo builds the `anamnesis` / `amn` binaries for those tests and hands them over through `CARGO_BIN_EXE_*`, so they can never run against a stale binary.
4. **If the commit touches any `///` or `//!` comment**, run the rustdoc sweep — see [Documentation Checks](#documentation-checks). `--all-features` alone cannot see a link that breaks under an intermediate feature combination, and a public-items run cannot see a broken link on a `pub(crate)` item at all.
5. Update `CHANGELOG.md` — add a bullet under the `[Unreleased]` section for any user-visible change (new feature, fix, breaking change). Follow [Keep a Changelog](https://keepachangelog.com/) categories: Added, Changed, Fixed, Removed.

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

## Fuzzing

`cargo-fuzz` (libFuzzer) does not build on Windows-MSVC, but it **is installed in WSL2 Ubuntu on this machine** (nightly + `cargo-fuzz` 0.13.1), which is where every recorded campaign ran. Run it from WSL, with a Linux-side target dir so it neither crawls through `/mnt/c` nor mixes with the Windows build:

```bash
cd "/mnt/c/Users/Eric JACOPIN/Documents/Code/Source/anamnesis"
CARGO_TARGET_DIR=~/anamnesis-fuzz-target cargo +nightly fuzz run fuzz_convert_bytes -- -max_total_time=180 -rss_limit_mb=2048
```

On Windows, `cargo +nightly check --manifest-path fuzz/Cargo.toml --bins` confirms the targets compile but runs nothing. When a fuzz target is added or its entry point changes, run a campaign and record it in `fuzz/README.md` § Status; a compile check is not coverage.

## Performance Changes

anamnesis is meant to be fast: a change may make it faster or leave it as fast, never slower.

If a commit claims a perf win (faster, less memory, fewer allocations, fewer branches), it must include a measurement, not just an analysis:

1. **Decide with the paired harness, on x86-64.** [`benches/ab.rs`](benches/ab.rs) (tango) loads the baseline and the candidate together and interleaves them sample by sample, so drift cancels; its floor is ~2 %. Export the baseline from the parent commit, then compare:
   ```powershell
   cargo export target/benchmarks -- bench --bench=ab --features gptq,awq,bnb,gguf   # on the baseline
   cargo bench --bench=ab --features gptq,awq,bnb,gguf -- compare target/benchmarks/ab --filter 'gptq_*' --noise-threshold 2.5
   ```
   **Run it about 10 times per arm** (these arms take milliseconds) and judge the median, with min and max. Filter each run to the family under test plus one untouched family as a control (e.g. `'{gptq,fp8_tensor}_*'`): on this desktop a full-suite run can be contended part-way through and swing untouched kernels by ±100 %, and the control is how a noisy run is recognised. See `CONVENTIONS.md` § *Benchmark evidence* for which instrument may decide what, and each one's measured floor.
2. **For an absolute magnitude on a real fixture** (a model file, not the synthetic layer), use a best-of-5 release-mode median with `target-cpu=native`. Templates: the `tests/bench_*_adhoc.rs` files, each gated `#[ignore]` and run with `cargo test --release --features <flag> --test <name> <test_fn> -- --nocapture --ignored`.
3. **Record both before and after numbers in the commit message** — median + range (min/max), and the bench command used. This is what makes a regression reversible: the next reviewer (or the next person to read `git log`) can re-run the same bench against the parent commit and know the answer.
4. **If the measurement does not show a win in the expected direction, do not commit.** Estimates and asymptotic arguments are hypotheses, not data — see `5f2632b` ("Revert NPZ memset elimination") for the cautionary case where a confidently estimated `~30 %` saving turned out to be a measured `~33 %` regression.

These rules apply to perf-claim commits. Correctness fixes, doc changes, and feature additions do not need a measurement to ship, **with one exception: any change inside a dequant kernel's module (`src/remember/*`, `src/lethe/*`) gets the paired no-regression check (step 1) even when it claims nothing.** Code that never runs per element can still change how the hot loop compiles: in the v0.7.8 close-out, moving `GPTQ`'s entry validation into a shared helper cost `gptq_int4_bf16` +69 %, and a `checked_add` in a per-block reader cost `NF4` ~5 % (`docs/perf-experiments.md` Experiment 19).

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
8. **Approve the release, then wait for the publish workflow to go GREEN.**
   Since v0.7.9 it runs three jobs:
   - `verify` builds, tests and packages the crate (`cargo package --locked`),
     with a read-only token and no `id-token` permission, and checks that
     `CHANGELOG.md` has a section for the tag;
   - `publish` waits for your approval on the `release` environment (Actions
     tab → the run → **Review deployments**), then runs
     `cargo publish --locked --no-verify` through Trusted Publishing. It
     compiles nothing: no dependency build script ever runs in the job that
     can mint a publish token;
   - `release` runs `gh release create` for the tag. The Release is what
     carries the test corpus, because `Cargo.toml`'s `exclude` keeps `tests/`
     out of the published crate (under 1 MiB instead of 4.8 MiB), and GitHub's
     per-tag source tarball ships it verbatim.
9. Check the Release actually appeared and its notes are the right section.
   If `release` failed *after* `publish` succeeded, **do not re-run the
   workflow**: the publish step would fail on the already-taken version and
   mask the real error. Fix the workflow for next time, then create the missing
   Release by hand from a clean checkout at the tag:
   ```bash
   v=0.7.9
   scripts/release-notes.sh "$v" > release-body.md
   # append the "Verifying the correctness claims" footer, then:
   gh release create "v$v" --title "v$v" --notes-file release-body.md --verify-tag
   ```
   `scripts/release-notes.sh` slices the notes out of `CHANGELOG.md` by
   matching `## [X.Y.Z]`, and **fails** if no section matches, which is the
   intended alarm for "step 3 was skipped and `## [Unreleased]` was never
   renamed". `verify` runs it first, so that alarm fires before anything is
   published. Release creation runs *after* `publish` on purpose, so a Release
   can never advertise a version whose publish failed.

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
