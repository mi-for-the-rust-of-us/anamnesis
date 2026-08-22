# anamnesis #new, reply 1 (Draft)

- **Target issue:** not yet filed. To be opened on
  https://github.com/mi-for-the-rust-of-us/anamnesis/issues **after `v0.7.7` is
  tagged and published**, because the reproduction says `git checkout v0.7.7`
  and the `CHANGELOG` advice to pin `0.7.6` only parses once `0.7.7` exists.
  Rename this file to `anamnesis-<number>-p1.md` once filed.
- **Status:** Draft (2026-08-22).
- **Context:** The first entry in this archive that is **not** an upstream reply.
  Phase 7.7 migrated `dequantize_bnb_int8`'s tile loop and measured it faster on
  x86-64 and slower on `aarch64`. v0.7.7 ships it knowingly, disclosed in
  `CHANGELOG.md` and in `src/remember/bnb.rs`. This issue asks a contributor with
  Apple Silicon to measure the one platform neither of our instruments covers,
  because the `aarch64` hardware we *can* reach is Linux server-class ARM without
  hardware `FP16`, and M-series has `ARMv8.2 FEAT_FP16`.
- **Outcome:** (none yet; unfiled)
- **Lesson / Leverage angle:** The archive's convention assumed replies onto
  other people's threads. This is our own issue, asking for hardware we lack.
  Kept here anyway rather than inventing a second folder, because the value is
  the same: a drafted, reviewed body that can be pasted verbatim.
- **Accuracy flags:**
  1. **The mechanism is unexplained and the issue says so.** Static counts point
     the wrong way: instruction count fell 435 → 433 and out-of-line calls 4 → 3
     while wall clock rose 21 %. The software-`f16` reading is a hypothesis
     consistent with the counts, not a confirmed cause.
  2. **The x86-64 and `aarch64` figures come from different instruments** — a
     paired harness and CodSpeed walltime respectively — because no single
     instrument covers both. Each is reproduced within its own instrument (three
     runs, and two independent baselines), but they are not like-for-like.
  3. **`~2 %` floor and `--noise-threshold 2.5` are ours, not universal.** The
     body tells the reader to calibrate on their own machine before trusting
     either, and the thermal caveat for laptops is a real confound at this
     effect size, not boilerplate.
  4. **Apple Silicon behaviour is entirely unmeasured**, including whether the
     separate `F16`-costs-2–3×-`BF16` finding holds there. Both are stated as
     open questions rather than predictions.

---

## Summary

v0.7.7 ships a change to the `BnB` `INT8` dequantisation kernel that is **faster on x86-64 and slower on `aarch64`**, at `F16` output only. We shipped it knowingly, because both readings are real and we cannot measure the architecture most likely to care.

**What we need:** a measurement on **Apple Silicon**. Neither of our instruments covers it, and it is plausibly the platform where this matters most — GGUF on an M-series Mac is one of the most common local-LLM setups there is.

## The numbers

The change is `dequantize_bnb_int8`'s tile loop moving from `chunks_exact` + `OutputElement::write_scratch` to `as_chunks` + `OutputElement::write_tile` (fixed-size arrays).

| Arm | x86-64 (3 runs, paired harness) | `aarch64` (CodSpeed walltime) |
|---|---|---|
| **`bnb_int8` @ `F16`** | **−5.22 % / −2.87 % / −5.96 %** | **+21.35 % / +20.82 %** |
| `bnb_int8` @ `F32` | not significant | +5.56 % / +1.92 % |
| `bnb_int8` @ `BF16` | not significant | +0.74 % / −1.26 % |

The `aarch64` figures come from two independent baselines: one isolating the merge that introduced the change (`033e763` → `7218723`), one comparing the release candidate against pre-merge `main`. Untouched kernel families moved by at most **3.42 %** on the same runs, so the effect is roughly 6× the control spread and reproduces.

The `aarch64` hardware is **Linux server-class ARM bare metal** (CodSpeed macro runners). **Not Apple Silicon.** That distinction is the whole reason for this issue.

## What we know about the mechanism

Reading the cross-compiled `aarch64` disassembly:

- It is **`F16`-only**. `BF16` and `F32` are flat.
- On `aarch64`, the `F16` narrowing has **no hardware fast path in play**: `F16Out::write_scratch` compiles to 91 instructions with **zero `fcvt`** and zero NEON lane operations. It is a software conversion.
- `F16Out::write_tile` **is** inlined there (no standalone symbol), so the change inlined that software conversion into the hot loop where it previously went out of line.
- The migrated version does **less** vector memory work: 128-bit load/stores fell 11 → 6, NEON lane-ops 60 → 52.

## What we do not know

**Why it is slower.** Static counts point the *wrong way*: instruction count went 435 → 433 and out-of-line calls 4 → 3, both slightly better, while wall clock rose 21 %. This is the second time in this phase that instruction counts contradicted the stopwatch (see `docs/perf-experiments.md` Experiment 17).

**Whether it reproduces on Apple Silicon.** M-series parts have `ARMv8.2` `FEAT_FP16` — *hardware* half-precision arithmetic — and a materially different microarchitecture (wider decode, much larger caches). If Apple's hardware `f16` path is selected there, the regression may shrink, vanish, or invert. Or it may not. We have no data.

## How to reproduce (Apple Silicon)

Either method works; the second needs no extra tooling.

**Method A — paired A/B (most sensitive, ~2 % floor):**

```sh
git clone https://github.com/mi-for-the-rust-of-us/anamnesis
cd anamnesis
git checkout v0.7.7
cargo install cargo-export

# baseline = v0.7.7 as shipped (the migrated kernel)
cargo export target/benchmarks -- bench --bench=ab --features gptq,awq,bnb,gguf

# candidate = the pre-migration kernel
git checkout 033e763 -- src/remember/bnb.rs

# compare. A large NEGATIVE delta on bnb_int8_f16 means reverting is faster,
# i.e. the regression reproduces on your hardware.
cargo bench --bench=ab --features gptq,awq,bnb,gguf -- \
    compare target/benchmarks/ab --noise-threshold 2.5
```

**Calibrate the threshold on your own machine first**, rather than inheriting ours. `2.5` comes from a floor we measured on *this* x86-64 box; yours will differ. To find it, compare the harness against itself with nothing changed — any delta it reports is the floor:

```sh
cargo export target/benchmarks -- bench --bench=ab --features gptq,awq,bnb,gguf
cargo bench --bench=ab --features gptq,awq,bnb,gguf -- compare target/benchmarks/ab
```

Everything it prints there is noise by construction. Set `--noise-threshold` a little above the largest value, then run the real comparison. On our box that self-check reported up to +2.20 %, and starred three arms as significant while nothing had changed — **a significance marker means "significant given the sampling", not "real"**.

**Do not quote its absolute millisecond figures** — it samples adaptively for the *difference*, and its magnitudes swing 50–100 % between invocations. See `benches/ab.rs`.

**If you are on a laptop, this matters:** these benchmarks run for minutes at full tilt on a 4096 × 11008 fixture, and a MacBook will thermally throttle well inside the window. Sustained throttling can move numbers by more than the 21 % effect we are chasing, and it drifts *downward over time*, which biases whichever side you measure second. Plug in, keep it cool, and prefer a Mac Studio / Mac mini if one is to hand. Method A's interleaved sampling is far more robust to this than Method B, so use A on a laptop.

**Method B — criterion medians (simpler, gives absolute times too):**

```sh
git checkout v0.7.7
cargo bench --bench dequant --features gptq,awq,bnb,gguf -- dequant_bnb_int8
git checkout 033e763 -- src/remember/bnb.rs
cargo bench --bench dequant --features gptq,awq,bnb,gguf -- dequant_bnb_int8
```

criterion saves a baseline on the first run and prints the change on the second.

To restore the shipped kernel afterwards: `git checkout v0.7.7 -- src/remember/bnb.rs`.

The revert is bit-exact — `cross_validation_bnb` and `cross_validation_bnb_encode` both pass either way — so nothing about correctness is at stake in flipping between them.

## What would help most

1. **Does `bnb_int8` at `F16` regress on M-series, and by how much?** This is the question. The `BF16` and `F32` arms in the same run act as controls.
2. **Bonus, and independently useful:** is `F16` output 2–3× slower than `BF16` there at all? We measured **2.02× to 3.11×** on x86-64 and **2.10× to 2.93×** on Linux `aarch64`, across all seven dequant families, and documented it on `F16Out`. Apple Silicon's hardware `FP16` may not show it, and if so the doc needs a caveat. `cargo bench --bench dequant --features gptq,awq,bnb,gguf` prints every width.
3. **The raw output rather than a summary**, please — the untouched kernel arms in the same run are what let us tell signal from noise, and a "bnb_int8_f16 was X %" on its own cannot be judged.
4. **Environment**, so the numbers can be read later:
   ```sh
   sysctl -n machdep.cpu.brand_string
   sw_vers
   rustc --version
   ```

## What the answer decides

- **Regression reproduces on M-series** → revert the migration in a patch release. The x86 gain (~5 %) does not pay for it.
- **Regression is absent on M-series** → it is a property of server-class ARM without hardware `FP16`, which narrows who is affected and lets us keep the change with a documented caveat.
- Either way, the `F16` cost table on `F16Out` gains its missing third platform.

## Background

- `docs/perf-experiments.md` Experiment 17 — the whole `as_chunks` investigation: six sites, six separate answers, and why no site may be migrated on a sibling's number
- `docs/perf-experiments.md` Experiment 18 — what `F16` output costs and why the inlining hypothesis failed
- `src/remember/output.rs` — `F16Out`'s `# Cost` section
- `ROADMAP.md` Phase 7.7

Happy to run anything else from this side; we just cannot run it on the right silicon.
