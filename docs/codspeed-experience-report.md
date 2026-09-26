# CodSpeed: a retour d'expérience

*Written 2026-08-23, after four weeks of use on `anamnesis` (first run
2026-07-26) and one
phase (7.7) that leaned on it hard. Scope: one Rust crate, one organisation on
the free plan, ~61 runs. This is an engineering report, not a verdict: several
items below are properties of benchmarking rather than of CodSpeed, and they are
marked as such.*

*Status, 2026-09-26: the `paths:` filter this report describes as added was
staged inside a commented-out trigger, because the August allowance was already
spent. The job therefore did not run at all from 2026-08-23 until the v0.7.8
close-out re-enabled it with the filter live. The report below is unchanged.*

## What we used it for

Continuous performance tracking on a quantised-tensor library whose whole value
proposition is speed and bit-exactness. The dequant suite grew from nine
benchmark ids to twenty-two during the period, across seven kernel families and
three output widths, plus whole-model convert and parse paths.

## What worked, and worked well

**The low-noise claim is real, and we verified it rather than trusting it.**
Comparing two `main` commits whose diff under `src/` was comments and attributes
only, so the kernels were provably codegen-identical, six benchmarks moved by at
most **0.90 %**, mean 0.59 %. Our own development desktop, measured the same way,
sits at 10 to 23 %. That is a ten to twenty-five fold improvement in run-to-run
stability, and it is the product's core promise delivered.

**Setup was genuinely small.** A workflow file, an OIDC permission, and
`cargo codspeed build` / `run`. No token to manage. PR comments and a status
check arrive without further work.

**It measures an architecture we cannot otherwise reach.** Our releases are
x86-64; the macro runners are `aarch64`. That turned out to be a constraint (see
below) but it also gave us three findings we would never have had, including a
21 % regression on ARM that our x86 measurements called a 5 % improvement.

## Properties of benchmarking, not of CodSpeed

Listed separately because it would be unfair to charge them to the product.

**Code layout dominates small effects.** Two different binaries have two
different layouts, and the measured difference contains that. On CodSpeed we saw
an **untouched** kernel move **4.16 %** consistently across two runs. So while
run-to-run noise is 0.9 %, the *effective* resolution for a real code change is
nearer 4 %. Nothing removes this; layout randomisation is a research tool, not a
product feature. The consequence for us was practical: we had to stop quoting the
0.9 % figure as though it were the experiment's precision.

**Our own benchmarks were partly at fault.** Two of them wrote output files. On a
CI runner that puts storage in the numerator. They reported **1.00x across thread
budgets** (167.45 ms at one thread, 168.47 ms at four) and, at `F32`, **315.39 ms
at both**. That is our design error, not CodSpeed's.

## What we actually ran into

**1. Walltime is `aarch64`-only, with no self-hosted option.** Confirmed in the
documentation: the instrument requires CodSpeed's own macro runners, and those
are Linux ARM64. There is no x86-64 walltime path at any tier we could find. For
a project developed, released and mostly consumed on x86-64, the continuous
instrument therefore measures a platform we do not ship from. We now treat it as
an ARM drift watch and decide changes locally, which is a reasonable outcome, but
it was not the one we expected when we adopted it.

**2. The feedback loop is about 18 minutes, including a fixed build tax.**
Measured on one run: **3 m 51 s from job start to the first benchmark**, paid on
every question regardless of size.
One question we spent 18 minutes on was later answered in **30 seconds** locally
with a paired harness. Loop length changes which questions you are willing to
ask, which is a subtler cost than the minutes.

**3. Two measurement bases disagreed by about 2x, and we never resolved it.**
For `remember_gguf_whole_model/threads_4`, criterion's own median printed in the
run log was stable at **168.32 to 168.47 ms across five consecutive `main` runs**.
CodSpeed's reported `BASE` for the same benchmark id was **89.9 ms**. Every other
group agreed between the two within 1 %. We could not determine which number to
trust for those groups, and the divergence was confined to the file-writing ones,
which suggests it is related to their I/O rather than arbitrary. Still: when
two numbers for the same identifier differ by 2x, the reader needs to know which
is the number.

**4. A false regression on a pull request that changed no library code.** The
report read *"Merging this PR will degrade performance by 34.31 %"* on a change
that touched only `benches/` and a roadmap file, so no library code at all.
Driven by the two file-writing groups. **The same run also produced two checks
that disagreed**: `Run benchmarks` passed while `CodSpeed Performance Analysis`
failed, which is coherent once you know one is execution and the other is the
verdict, and confusing before you do. The practical harm is not the wrong number,
it is that a red check people learn to click past stops working as a check.

**5. There is no path to bring your own measurements.** The action's inputs are
`run`, `mode`, `token`, `working-directory`, `config`, `instruments`,
`upload-url` and instrument knobs. There is no upload-only mode and no
results-file input, so the measurement and the analysis are coupled: you take
CodSpeed's numbers or none. This is a legitimate product decision, and it is the
reason we could not keep the dashboard while measuring on x86-64. By contrast,
tools whose product is the *analysis* accept externally produced results.

**6. The free-tier arithmetic is tight, and the default trigger does not help.**
600 macro-runner minutes per organisation per month, against an ~18-minute suite,
is about **33 runs a month**. We spent 22 in three days and exhausted August.
Sixteen of those were pushes to `main`, most of them documentation commits that
could not possibly move a benchmark. The fix is a `paths:` filter on the trigger,
which we have now added; a default or a documented recommendation would have
saved us the lesson. As an illustration of how easily this happens: **the commit
that added this very report triggered another run**, because it landed on `main`
before the filter did.

**Suite growth compounds it silently.** Adding thirteen benchmark ids, which was
the right call for coverage, took a run from ~13 to ~20 minutes. That is a
permanent tax on every future run, and nothing in the product surfaces the
trade between coverage and remaining allowance at the moment you make it.

**7. There is no quota visibility, and an exhausted allowance queues rather than
fails.** We learned we were out by email, mid-session, after the fact: no warning
in the workflow at 80 %, and no counter we could read from CI. Worse for
diagnosis, runs submitted after exhaustion sit in `queued` rather than failing
fast, so a workflow neither succeeds nor tells you why. One is queued as this is
written.

**8. The x86-64 fallback is a different instrument, not a cheaper one.** The
simulation mode runs anywhere, including free x86 runners, but it derives its
figures from instruction-level simulation. Phase 7.7's central finding was that
**instruction counts do not predict wall clock**: one kernel's counts fell at
every output width while its wall clock rose about 30 % on the same architecture.
Our own Experiment 10 had separately recorded simulation scoring a null SIMD
change as a win. So the free x86 path cannot answer the questions the paid ARM
path can.

## Where we landed

| Question | Instrument |
|---|---|
| Is this change faster? | Local paired harness (tango), x86-64, ~30 s, ~2 % floor |
| Has ARM drifted since last release? | CodSpeed, `main` pushes only, with a `paths:` filter |
| Does it still produce identical bytes? | Cross-validation suites, unchanged |

We kept CodSpeed. `aarch64` is a first-class target for a GGUF library, since
Apple Silicon and Graviton carry a real share of that workload, and nothing else
we have watches it continuously. What changed is that it no longer renders
verdicts on pull requests: it watches for drift, and decisions are made on the
architecture we release from.

## The one thing we would tell a new user

Calibrate the floor before quoting a delta, and calibrate it against
**byte-identical source** rather than against intuition. Every number in this
report that mattered came from doing that, and the two mistakes we made in the
period both came from not doing it soon enough.
