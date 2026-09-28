#!/usr/bin/env bash
# Print the CHANGELOG.md section for one version, or fail if there is none.
#
# Usage: scripts/release-notes.sh <version> > release-notes.md
#   <version> without the leading `v`, e.g. `0.7.9`.
#
# Used twice by .github/workflows/publish.yml: by the `verify` job, so that a
# tag whose `## [Unreleased]` was never renamed fails BEFORE anything is
# published, and by the `release` job, which turns the section into the GitHub
# Release body.
set -euo pipefail

version="${1:?usage: release-notes.sh <version>}"

# Print the body between this version's heading and the next `## [`.
#
# The version is matched with `index(...) == 1`, a LITERAL prefix test, not a
# regex. An earlier draft used a dynamic regex (`$0 ~ "^## \\[" v "\\]"`) and
# silently matched nothing: inside the shell-quoted awk program `\\[` collapses
# to `\[`, which gawk treats as a plain `[`, so the pattern became
# `^## [0.7.3]` where `[0.7.3]` is a CHARACTER CLASS matching one of `0 . 7 3`.
# It could never match a literal bracket. Dots in a version string are regex
# metacharacters too, so literal matching is the right tool here regardless.
notes="$(awk -v hdr="## [$version]" '
  index($0, hdr) == 1 { found = 1; next }
  found && /^## \[/   { exit }
  found               { print }
' CHANGELOG.md)"

if [ -z "${notes//[[:space:]]/}" ]; then
  echo "::error::No CHANGELOG section found for [$version]. Rename '## [Unreleased]' to '## [$version] - YYYY-MM-DD' before tagging." >&2
  exit 1
fi

printf '%s\n' "$notes"
