# __reports__ / issue11_bnb_int8_f16_m3

Findings from investigating [issue #11](https://github.com/mi-for-the-rust-of-us/anamnesis/issues/11):
does the `BnB` `INT8` F16 dequant migration (shipped v0.7.7) regress on Apple Silicon?

## Status

Open question answered: **the `aarch64` ~21 % F16 regression does not reproduce on
Apple M3 Pro.** The migrated (shipped) kernel is ~5 % faster at F16, matching
x86-64's direction. Bonus: on M3, `bnb_int8` F16 (13.5 ms) is *faster than its own
BF16* (14.4 ms) — an inversion of the other two platforms. Single-machine (M3 Pro)
evidence; awaiting PI confirmation before it becomes a doc claim.

## Documents

### Round 00
- `00-findings_v0.md` — **latest** — findings briefing (data, charts, steering questions)
- `00-findings_v0-f16_vs_bf16_ratio.png` — F16/BF16 ratio across families
- `00-findings_v0-absolute_medians.png` — absolute medians per width
- `00-findings_v0-paired_ab_delta.png` — paired A/B deltas
- `dequant_medians.json` — raw medians extracted from `target/criterion/`
- `plots.py` — chart generation source (regenerate with `uv run --with matplotlib python3 plots.py`)

## Revisions
- v0 — 2026-08-23 — initial findings
