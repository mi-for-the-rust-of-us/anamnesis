# Issue replies archive

This folder archives comments drafted or posted on upstream issues where
anamnesis's own work surfaced or diagnosed a problem. Currently
[rust-lang/cargo](https://github.com/rust-lang/cargo/issues) and
[rust-lang/rust](https://github.com/rust-lang/rust/issues); previously
[huggingface/candle](https://github.com/huggingface/candle/issues), whose thread
is archived in the sibling project (see below).

It follows the conventions established by
[`hf-fetch-model/docs/issues/README.md`](https://github.com/mi-for-the-rust-of-us/hf-fetch-model),
deliberately, so the two archives read alike and a reply can move between them
without reformatting. **That file is the origin of the convention; this one is a
copy adapted to anamnesis.** If the two ever disagree, prefer the hf-fetch-model
version and fix this one.

## Which archive does a reply belong in?

By **which project's work produced the evidence**, not by which upstream repo is
being written to. The candle `.pth` pickle-VM reports came out of anamnesis's
hardening but are archived in hf-fetch-model because that is where the issue
archive lived at the time; anything new from anamnesis belongs here. Cross-link
rather than duplicate.

## File naming

```
<upstream-repo>-<issue-number>-p<N>.md
```

- `<upstream-repo>`: short identifier (`cargo`, `rust`, `candle`, …)
- `<issue-number>`: the upstream issue/PR number (no `#`)
- `p<N>`: post/reply index within that thread (`p1` = our first comment)

**Not-yet-filed issues** have no number. Use `<repo>-new-<slug>-p1.md` and
**rename the file once it is filed**, updating the `Target issue:` line and this
README's table. Example: `cargo-new-rustflags-build-std-p1.md`.

## File structure

A lightweight metadata header, then the reply body verbatim:

```markdown
# <upstream-repo> #<issue-number>, reply <N> (<status>)

- **Target issue:** <URL>
- **Status:** <Posted | Draft | Superseded>  (plus date or superseding file)
- **Context:** one or two sentences: who asked, what was stuck, what we showed
- **Outcome:** (for posted) what the OP/maintainer said in response
- **Lesson / Leverage angle:** (optional) anything worth noting for future replies
- **Accuracy flags:** anything not 100% faithful or not 100% certain

---

<reply body exactly as posted, or as drafted>
```

The `---` marks the boundary, so the body is pastable into GitHub as-is.

## Status taxonomy

| Status | Meaning |
|--------|---------|
| **Posted** | Sent upstream; matches what is live there verbatim |
| **Draft** | Written but not yet posted (pending review, verification, release timing) |
| **Superseded** | Was posted, but later investigation gave a better diagnosis. Link the superseding file. **Do not delete**, because superseded files document the learning curve and stop the same mistake being made twice |

## Flagging practice

**Flag anything that is not 100% faithful or not 100% correct.** This is the part
of the convention that does the most work, and the part most easily skipped:

- **Verified vs guessed.** If a claim was checked against a specific reference
  (upstream source, a doc page, live tool output), name the reference. If it was
  a reasoned inference, say so in those words.
- **Confounded evidence.** If two variables changed between the failing and
  working configurations, the conclusion does not follow from the comparison.
  Say so, and either isolate it or mark the claim conditional. An upstream bug
  report resting on a confounded comparison wastes a maintainer's time and is
  worse than not filing.
- **Posted content later found inaccurate.** Note it in the metadata, link the
  correction, and say what the live thread now says.
- **Scope of verification.** Name the exact environment a recipe was verified on
  (OS, toolchain, target triple, dependency tree). "Works" on one CI runner is
  not "works".
- **Solution vs workaround.** If a thread asks for X and we achieved X' by
  avoiding the thing that made X hard, it is a workaround. Say that in the reply
  itself, not only here.

## Workflow

1. **Before posting**: draft as `<repo>-<issue>-p<N>.md` with status `Draft`.
   Review offline. Verify the accuracy flags actually hold. Post.
2. **After posting**: set status to `Posted`, add the date, leave `Outcome:`
   blank until someone replies.
3. **If superseded**: mark `Superseded`, add `See: <newer-file>`, keep the
   original intact.

## Current archive

| File | Target | Status |
|------|--------|--------|
| [cargo-13146-p1.md](cargo-13146-p1.md) | cargo #13146: `-Zbuild-std` + `cargo test`; a `--bin` workaround for the sanitizer use case | Posted |

A second issue, `target.<triple>.rustflags` not reaching `-Zbuild-std` units,
was drafted and **abandoned before filing**. The evidence was confounded (two
variables changed between the failing and working runs), and a single-variable
rerun disproved it: both flag sources work. It is recorded here rather than in a
file because nothing was ever written to send, and because "we nearly filed this
and were wrong" is the more useful note. See the `.github/workflows/tsan.yml`
header for the corrected constraint list.

Related, archived in the sibling project: the candle `.pth` pickle-VM DoS thread
(`candle-3617-p1..p3`), which came out of anamnesis's Phase 6.11 hardening.
