# Security policy

anamnesis parses files that are often untrusted: model checkpoints downloaded
from a hub, or uploaded to a server. A crash, an unbounded allocation or
super-linear work triggered by a crafted file is a security bug here, not just a
bug. So is any way to make the `.pth` pickle interpreter reach something outside
its allowlist.

## Reporting a vulnerability

**Please do not open a public issue.** Report it privately through GitHub:

> [Report a vulnerability](https://github.com/mi-for-the-rust-of-us/anamnesis/security/advisories/new)
> (repository **Security** tab → **Report a vulnerability**)

A useful report says which entry point you called (`parse_*`, `inspect_*`,
`convert_bytes`, the `amn` CLI, …), with which `ParseLimits`, and includes the
smallest input that shows the problem, or the code that builds it.

You can expect an acknowledgement within a week. Fixes for confirmed issues ship
as a patch release, with credit to the reporter unless you prefer otherwise.

## Supported versions

| Version | Supported |
|---|---|
| latest `0.7.x` | ✅ |
| older | ❌ upgrade to the latest release |

## What counts as a vulnerability

The contract for untrusted input is described in the README,
[Parsing untrusted input](README.md#parsing-untrusted-input): every input yields
a clean `Err`, never a panic, an abort, memory beyond the caller's `ParseLimits`,
or work that grows faster than the input. Reports of any of these are in scope,
as is any pickle global that gets past the allowlist.

Out of scope: verifying that a model file is authentic (signing), process
isolation, work that is linear in the size of the input, and the memory-mapped
entry points (`parse`, `parse_pth`, `parse_gguf` on a path), which are meant for
trusted local files; use the copy-based `*_bytes` / `*_from_reader` entry points
for untrusted input.

## How fixes are published

A confirmed issue gets a [GitHub security advisory](https://github.com/mi-for-the-rust-of-us/anamnesis/security/advisories),
a patch release on crates.io, and an entry in the
[RustSec advisory database](https://github.com/rustsec/advisory-db) so that
`cargo audit` and `cargo deny` report it. Once the Python package exists, its
advisory is published on PyPI as well.
