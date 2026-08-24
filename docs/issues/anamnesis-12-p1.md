# anamnesis #12, reply 1 (Posted)

- **Target issue:** https://github.com/mi-for-the-rust-of-us/anamnesis/pull/12
- **Status:** Posted (2026-08-24),
  [comment 5393198623](https://github.com/mi-for-the-rust-of-us/anamnesis/pull/12#issuecomment-5393198623).
  Posted **after** CI went green and **before** the merge, so the answer to the
  contributor's question was on the thread before the decision was executed
  rather than implied by it. The body below matches what is live verbatim.
- **Context:** [PR #12](https://github.com/mi-for-the-rust-of-us/anamnesis/pull/12)
  is the measurement [issue #11](https://github.com/mi-for-the-rust-of-us/anamnesis/issues/11)
  asked for: an external contributor (@LittleCoinCoin) ran the `BnB` `INT8` `F16`
  A/B on an Apple M3 Pro, hardware nobody here owns, and opened a report-only PR
  with the study, charts and raw data under `__reports__/`. The contributor asked
  **twice** whether that folder should be relocated under `docs/` or dropped. It
  was never answered on the thread, while `main` had already accepted it in
  `Cargo.toml` and `CHANGELOG.md`. This reply closes that gap and accepts the PR.
- **Outcome:** PR #12 merged as `3390572` (merge commit, deliberately not a
  squash, so the contributor's own commit `1f6ac98` stays in the graph under
  their authorship). All four checks green: Stable, MSRV 1.88, ThreadSanitizer,
  Docs. Issue #11 closed against it. `__reports__/README.md` and a provenance
  pointer from Experiment 18 landed in the follow-up commit on `main`.
- **Lesson / Leverage angle:** The archive's first entry about **someone else's
  contribution to us**, rather than our reply to an upstream. The transferable
  point is procedural: the acceptance decision was made on `main` (`Cargo.toml`
  `exclude`, plus a CHANGELOG line) *before* the contributor was told, so from
  their side an answered question still looked open. Staging for a merge is not
  the same as replying to the person who asked. Say it on the thread.
- **Accuracy flags:**
  1. **"Already accounted for on `main`" is verified, not asserted.**
     `Cargo.toml:42-45` carries both the comment (`/__reports__` holds contributed
     measurement studies) and `exclude = ["/fuzz", "/tests", "/__reports__"]`;
     `CHANGELOG.md:27` records it. Checked at `9534b63`.
  2. **The "unlikely to reverse" quotation is verified against the diff.**
     `73bc144` removes the line `makes the direction unlikely to reverse` from
     `src/remember/output.rs` and replaces it with text naming the break
     explicitly. The reply quotes our own prior wording, not a paraphrase.
  3. **"Four of seven families" is verified**, from the same diff:
     `F32` is strictly faster than `BF16` in four of the seven families on M3.
  4. **Scope is one machine, and the reply says so.** `output.rs` as amended
     states the scope is "one M3 Pro, macOS 26.5.2, rustc 1.92.0, so read it as
     one machine rather than as 'Apple Silicon'". The reply does not generalise
     past that, and deliberately avoids the phrase "Apple Silicon does X".
  5. **The mechanism remains unexplained.** Issue #11 flagged that static
     instruction counts point the wrong way (435 → 433 while wall clock rose
     21 %). Nothing in PR #12 explains *why*; it establishes *whether*. The reply
     does not claim a cause.
  6. **The two platforms' figures come from different instruments** (paired
     harness on x86-64 and M3, CodSpeed walltime on server `aarch64`). Stated in
     issue #11; not restated in this reply because the thread already carries it.
  7. **`__reports__/README.md` is promised, not yet written** at the time of
     posting. It lands in the follow-up commit on `main` immediately after merge.

---

Merging as-is, `__reports__/` included. No relocation needed, and thank you for
offering twice.

The folder is already accounted for on `main`: `Cargo.toml` excludes
`/__reports__` from the published crate, with a comment describing it as holding
contributed measurement studies, and the `CHANGELOG` records it. That was staged
in anticipation of this PR but never said here, which is my omission rather than
an open question on your side.

Keeping it separate from `docs/` is deliberate. Everything under `docs/` is
maintainer-authored prose; this is submitted primary data with its own charts and
a reproducible `plots.py`, and those are worth distinguishing. I will add a
`__reports__/README.md` describing the convention so it is documented rather than
inferred by the next contributor.

On the substance, the verdict landed in `73bc144`. The change stays, per issue
#11's pre-registered decision rule. `F16Out` and `F32Out` are now
platform-qualified, and Experiment 18 in `docs/perf-experiments.md` records that
this measurement **broke a prediction this project had made in writing**.
`F16Out`'s cost section said the direction was "unlikely to reverse", on the
grounds that the two platforms then measured agreed. That agreement was not
evidence: x86-64 and server `aarch64` were both doing *software* `f16`
conversion, so they agreed about the mechanism they shared and said nothing about
a part that does it in hardware. Generalising from a sample that shares the
mechanism you are generalising about is not evidence, and that lesson is recorded
rather than quietly patched.

The `F32` result you flagged as a bonus is the part I did not anticipate at all.
`F32Out` running faster than `BF16` in four of the seven families on that machine
made our `F32` documentation x86-specific without it saying so, which is a
documentation bug the issue never thought to ask about.

Two things I have kept deliberately narrow in how this is written up, both of
which follow your own framing:

- The result is scoped to **one M3 Pro** (macOS 26.5.2, rustc 1.92.0), not to
  "Apple Silicon". `output.rs` says so in those words.
- The **mechanism is still unexplained**. Static instruction counts moved the
  wrong way on `aarch64` (435 → 433 while wall clock rose 21 %). Your data
  settles *whether* the regression reproduces, not *why* it happens, and nothing
  here claims otherwise.

The three-run design with the F16 and F32 arms moving in opposite directions is
what makes the thermal-drift objection answerable, and calibrating the noise
threshold against the machine's own self-check floor before trusting it is
exactly the protocol the issue asked for. Thank you for doing it properly, and
for the raw output rather than a summary.

Closing #11 with this.
