# `__reports__` — contributed measurement studies

This folder holds **measurement studies contributed from outside the core
maintainers**, kept as submitted: raw data, charts, and the scripts that produced
them. One subfolder per study.

It exists because some questions this project needs answered require hardware
nobody here owns. When someone runs the measurement and sends the data, the data
itself belongs in the repository, not just a maintainer's summary of it.

## Why this is not under `docs/`

`docs/` is maintainer-authored prose: it argues, concludes, and is edited freely
as understanding changes. This folder is **primary evidence**, and it is
deliberately *not* edited after the fact. A claim in `docs/perf-experiments.md`
can be checked against the numbers here instead of taken on trust, which only
works if the two are kept apart and the source is left alone.

## Conventions

- **One subfolder per study**, named `issue<N>_<short_slug>` where `<N>` is the
  issue or PR the study answers.
- **Each subfolder carries its own `README.md`** stating the question, the
  verdict, and an index of its files.
- **Content is merged verbatim.** Maintainers do not rewrite a contributed study
  to match house style or to reflect later conclusions. If a study's framing is
  later superseded, that is recorded in `docs/`, with a link back here. The study
  stands as the record of what was measured and when.
- **Raw data is required, not optional.** A study is a chart plus the numbers
  behind it plus the script that drew it. Charts alone are not reproducible.
- **The linking direction is `docs/` → here.** The `docs/` claim carries the
  pointer to its provenance; a study does not need to track every place it is
  cited.

## Packaging

`/__reports__` is in `Cargo.toml`'s `exclude`, so nothing here ships in the
published crate. It is carried in the git repository and in GitHub's per-tag
source tarball, which is where a reader verifying a claim will look.

Keep studies small. Charts belong here; multi-megabyte artefacts do not.

## Current studies

| Study | Question | Verdict |
|---|---|---|
| [`issue11_bnb_int8_f16_m3/`](issue11_bnb_int8_f16_m3/) | Does the v0.7.7 `BnB` `INT8` `F16` migration regress on Apple Silicon the way it does on server `aarch64` (+21 %)? | **No.** ~5 % *faster* on an M3 Pro, matching x86-64. The change stays, per [issue #11](https://github.com/mi-for-the-rust-of-us/anamnesis/issues/11)'s pre-registered rule. Read in [`docs/perf-experiments.md`](../docs/perf-experiments.md) Experiment 18. |
