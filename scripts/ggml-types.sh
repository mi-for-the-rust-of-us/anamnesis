#!/usr/bin/env bash
# Print the live `ggml_type` entries of a llama.cpp `ggml.h`, one per line, as
# `GGML_TYPE_<NAME> = <discriminant>`, sorted by discriminant.
#
# Usage: scripts/ggml-types.sh [ggml.h]      (reads stdin when no file is given)
#
# "Live" means uncommented: upstream keeps removed types in the enum as
# comments (`// GGML_TYPE_Q4_2 = 4, support has been removed`), and those must
# stay unknown to anamnesis. `GGML_TYPE_COUNT` is the enum's sentinel, not a
# type, so it is dropped too.
#
# Used to produce `tests/fixtures/ggml_types.txt`, which
# `tests/gguf_type_snapshot.rs` checks the parser against, and by
# `.github/workflows/ggml-drift.yml`, which compares that snapshot with
# upstream every week. Issue #15 is what happens without it: upstream added
# three types and nothing noticed until a user's file was refused.
set -euo pipefail

# A live entry starts the line (after indentation) with its name; a removed one
# starts with `//`, so anchoring at the start of the line is what tells them
# apart.
grep -E '^[[:space:]]*GGML_TYPE_[A-Z0-9_]+[[:space:]]*=[[:space:]]*[0-9]+' "${1:--}" |
  sed -E 's/^[[:space:]]*(GGML_TYPE_[A-Z0-9_]+)[[:space:]]*=[[:space:]]*([0-9]+).*/\1 = \2/' |
  grep -v '^GGML_TYPE_COUNT ' |
  sort -t= -k2 -n
