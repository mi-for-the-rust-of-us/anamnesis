# PLAN — Upstreaming anamnesis's pickle-VM hardening to candle (#3617)

> **HISTORICAL (archived at the v0.7.8 close-out).** A plan written on 2026-06-17, when anamnesis was at v0.6.8. It records the upstreaming proposal as it stood then; for the current status of candle#3617, see the issue itself.

**Status:** Proposed (issue posted, no PR yet)
**Date:** 2026-06-17
**Upstream issue:** [huggingface/candle#3617](https://github.com/huggingface/candle/issues/3617) — *"Unbounded pickle-VM working set in `candle-core/src/pickle.rs` (DoS via crafted `.pth`)"* (opened by PCfVW, 2026-06-13)
**Reference implementation:** `anamnesis ≥ 0.6.6` (current 0.6.8), [`src/parse/pth.rs`](../../src/parse/pth.rs)
**Relation to other docs:** continues the the private `docs/security-audit-brief.md` / the private `docs/security-audit-findings.md` theme; this one targets *upstream* (candle) rather than anamnesis itself.

---

## Why this doc lives in anamnesis

The fix being upstreamed is anamnesis's own pickle-VM hardening (Phase 6.11, shipped v0.6.6, `cargo-fuzz`-ed). candle's `pickle.rs` has the same structural vulnerability anamnesis already closed, so this is "port our hardening to the upstream that inspired the dogfooding." The issue was surfaced via the hf-fetch-model candle-engagement work ([`hf-fetch-model/docs/issues/`](../../../hf-fetch-model/docs/issues/) archives the reply trail); a one-line cross-link from there back to this plan keeps the two projects' records consistent.

---

## Current state

### The vulnerability (confirmed against candle `main` source, 841 lines — byte-identical to the cached 0.9.2)

`candle-core/src/pickle.rs` has **no resource caps anywhere**. `read_pth_tensor_info` → `Stack::read_loop` → `finalize` materializes the entire pickle object tree before any tensor metadata is extracted, so a crafted `.pth` drives the VM directly on the normal load path. Two unbounded vectors:

1. **CWE-1325 — unbounded working set.**
   - `Stack { stack: Vec<Object>, memo: HashMap<u32, Object> }` (current `pickle.rs:296-299`) grows one `Object` per opcode with no cap (`Vec::with_capacity(512)` is only a hint).
   - The amplifier is the **memo clone**: `memo_get` (`pickle.rs:390-398`) does `obj.clone()` — and the code already flags it: *"Maybe we should use refcounting rather than doing potential large clones here."* `BinGet`/`LongBinGet` replay a memoized `List`/`Dict`, turning a few KB of opcodes into multi-GB of heap.

2. **CWE-674 — unbounded recursion + derived `Drop`.**
   - `Object` nests through `Box<Object>` (`Reduce`/`Build`/`PersistentLoad`, `pickle.rs:104-129`) and derives `Drop`.
   - `BinPersId` (`pickle.rs:489-493`) wraps one box deeper per opcode; a `Q`-chain builds an arbitrarily deep value whose **derived recursive `Drop`** overflows the call stack on teardown (an uncatchable process abort).

Scope is **availability only** — `reduce` (`pickle.rs:355-381`) builds an `Object::Reduce {…}` without invoking the callable, so there is no code-execution path. DoS only.

> **Line-number drift:** issue #3617 cites a **673-line** `pickle.rs` (e.g. `Stack` at L266-271, `memo_get` at L312-318). Current `main` is **841 lines** (those land at 296-299 / 390-398). The cited *structures* are all present and verified; the PR must re-anchor line references to current `main`.

### The precedent that makes this convertible

- [#3533](https://github.com/huggingface/candle/issues/3533) (GGUF DoS, opened by UNILESS) → PR [#3556](https://github.com/huggingface/candle/pull/3556) (HueCodes, merged) capped GGUF allocations, **including the `GGUF_MAX_VALUE_DEPTH` nesting cap** (currently **64**, verified at `gguf_file.rs:19`, enforced at `:326`); → PR [#3585](https://github.com/huggingface/candle/pull/3585) (ivarflakstad, **merged 2026-06-10**) finished it with the *tensor-size* check (*"Verify gguf tensor size before allocating… finishing the conversation in #3533"*). The depth precedent is therefore #3556's, not #3585's.
- On #3533, **PCfVW's comment** identified the residual gap (citing anamnesis's caps); ivarflakstad replied *"Thanks… @PCfVW 🙌 — all bases should be covered once #3585 is merged."*
- **#3585 deliberately does not touch `pickle.rs`.** So the pickle path is the open sibling, on a threat-class candle maintainers have demonstrably merged twice in ~3 weeks. The conversion template is proven: report (#3533) → PR (#3556/#3585).

### Status of #3617

Open, **0 maintainer comments** as of 2026-06-17. Body includes 3 minimal PoCs and a "how anamnesis fixes this (v0.6.6)" reference.

---

## The design to port (anamnesis `pth.rs`, verified)

anamnesis's `PickleVm` closes exactly these vectors. The working-set + depth governance below is **Phase 6.11 (anamnesis v0.6.6)**; the `MAX_PKL_SIZE` (100 MiB opcode-stream) and `MAX_PICKLE_PAYLOAD` (64 MiB per-item) caps predate it (v0.6.1–v0.6.3). candle's `pickle.rs` currently has **none** of these, so the PR adds the full stack. The mechanisms (all in [`src/parse/pth.rs`](../../src/parse/pth.rs)):

| Mechanism | anamnesis | Purpose |
|---|---|---|
| `MAX_PICKLE_WORKING_SET = 512 MiB` | permanent floor, `charge()` on every push + memo clone | bounds flat-stack flood **and** memo-replay together |
| `MAX_PICKLE_VM_DEPTH = 256` | enforced in `push_value` / `bump_top_depth`; `SlotMeta { depth }` tracked **incrementally** | construction-depth cap → recursive `Drop`/`Clone`/walks stay shallow |
| `MAX_PICKLE_PAYLOAD = 64 MiB` | `enforce_pickle_payload_cap` before `BINUNICODE`/`BINSTRING`/`BINBYTES` alloc | per-item string/bytes cap (defence in depth) |
| `deep_size` charged **before** the clone | `memoize_top` / `memo_get_clone` | reject an over-budget memo without allocating the duplicate |
| O(1) per opcode | depth tracked in `meta_stack`; deep walk only on memo-clone opcodes; memo stores `(value, SlotMeta)` | avoids an O(n²) CPU-DoS **in the guard itself** (a naive per-push deep walk would be the new bug) |

`cargo-fuzz`-ed — per anamnesis [`fuzz/README.md`](../../fuzz/README.md), the pickle/PTH targets ran clean: latest seeded campaign `fuzz_pth` **381k** runs, `fuzz_pth_limits` **193k** (earlier phases: 213k / 224k). The only crash any campaign surfaced was in the **NPZ** NPY-descriptor parser (`parse_descr`), unrelated to pickle and since fixed. **Accuracy note:** #3617 originally posted `fuzz_pth` 385k / `fuzz_pth_limits` 508k pointing at the CHANGELOG — both wrong (508k matched no PTH target; nearest is `fuzz_npz_parse` 503k). **Corrected in the live issue 2026-06-18** to the `fuzz/README.md` figures above.

---

## The adaptation gap (this is NOT a copy-paste)

candle's and anamnesis's pickle VMs differ structurally — the *accounting model* ports, the code does not:

| | candle `pickle.rs` | anamnesis `pth.rs` |
|---|---|---|
| Value type | `enum Object` | `enum PickleValue` |
| VM state | `Stack { stack: Vec<Object>, memo: HashMap<u32,Object> }` | `PickleVm { stack, meta_stack, mark_stack, memo, working_set, budget }` |
| Input | streaming `R: BufRead` (zip entry reader) | in-memory `&[u8]` + `pos` cursor |
| Dispatch | `Stack::read()` one opcode/call, `read_loop` to STOP | `execute()` loop |
| Errors | `crate::bail!` / `E::Msg` | `AnamnesisError::Parse` |
| Depth tracking | **none** | `SlotMeta` parallel stack |

**Implication:** add the accounting to candle's `Stack`, mirroring anamnesis's *model* in candle's idioms.

---

## What needs to be done (PR checklist)

1. **Re-anchor to current `main`** (841 lines; confirmed identical to cached 0.9.2). Branch from a fresh `main`.
2. **Add constants** matching candle's existing `GGUF_MAX_*` naming: `PICKLE_MAX_WORKING_SET`, `PICKLE_MAX_DEPTH`, `PICKLE_MAX_PAYLOAD`. **Depth value is a real choice, not a copy:** candle's `GGUF_MAX_VALUE_DEPTH` is **64**, anamnesis's `MAX_PICKLE_VM_DEPTH` is **256** — see Open decisions. Working-set default: anamnesis uses 512 MiB.
3. **Extend `Stack`** with `working_set: u64` and depth bookkeeping. Since candle's `Stack` has no per-slot depth, add a parallel `meta_stack: Vec<u32>` kept length-synced (or compute child-depth at each construction site). Mirror anamnesis's `SlotMeta` incremental approach to stay **O(1) per opcode**.
4. **Charge + cap at the choke points:**
   - Wrap `push` (or every push site) → charge shallow size + enforce depth (mirror `push_value`).
   - `memo_put` (`pickle.rs:400-404`) and **`memo_get`** (`pickle.rs:390-398`) → charge `deep_size` **before** the clone.
   - `BinUnicode` (`pickle.rs:482-488`) → cap `len` before `vec![0u8; len]`.
5. **Depth cap at construction** for the box/vec-nesting opcodes: `BinPersId` (489-493), `Reduce` (355-381), `Build` (338-353), `NewObj` (598-603), `Tuple*`, `Append(s)`/`SetItem(s)`/`Dict`.
6. **Map errors** to candle's `bail!`/`Error`.
7. **Port the 3 PoCs as `#[cfg(test)]` unit tests** (`Stack`/`read_loop` are `pub`): memo-replay amplification, `Q`-chain recursion, `N`-flood — each asserts a clean `Err`, not OOM/overflow. The recursion test must rely on the depth cap *preventing construction* so `Drop` never deep-walks (a real overflow can't be caught in-test).
8. **Keep it minimal and mirror #3585's framing.** PR title in the same register (e.g. *"Cap pickle VM working set and nesting depth"*); body links #3617, credits the `GGUF_MAX_VALUE_DEPTH` precedent (#3533/#3585), states scope = availability-only.
9. **Gate:** `cargo fmt`, `cargo clippy -D warnings`, `cargo test` (candle CI).

---

## Open decisions (settle before coding)

- **Working-set cap value.** anamnesis uses 512 MiB (permanent floor). candle is a library embedded in many contexts; maintainers may prefer a different default or a configurable knob. Lead with a fixed const matching their #3585 "always-on" choice; flag configurability as a follow-up.
- **Always-on vs parameterized.** #3585 added always-on GGUF caps → precedent is **always-on**. Match it (no API change to `read_all` / `PthTensors::new`).
- **Depth cap value.** candle's `GGUF_MAX_VALUE_DEPTH` = **64**; anamnesis's `MAX_PICKLE_VM_DEPTH` = **256**. Both are far above real `.pth` nesting (torch `state_dict` nests ~6 deep; anamnesis's post-execution cap is 32). Use a dedicated `PICKLE_MAX_DEPTH` const (separate subsystem). **Recommend 64** — it matches the maintainers' just-merged GGUF choice (cross-subsystem consistency) and is still ~10× real depth; fall back to 256 only if a legitimate deep `state_dict` surfaces.
- **`data.pkl` stream cap.** anamnesis also bounds the declared `data.pkl` entry size (`MAX_PKL_SIZE`, 100 MiB) before interpreting. candle reads the entry via a zip reader with no such cap — decide whether to add the stream-size guard too, or rely solely on the working-set charge (the working-set cap alone already bounds the heap; the entry cap is cheap defence-in-depth).
- **Fuzzing.** anamnesis ships `fuzz_pth` targets; #3585 shipped with unit tests only. Unit tests (the 3 PoCs) are likely sufficient for PR acceptance; offer a fuzz target as a follow-up if maintainers want it.

---

## Risks / notes

- **Streaming vs slice.** candle reads `R: BufRead` (no full in-memory buffer), but the working-set accounting maps cleanly — charge per push regardless of source. No need to buffer the whole stream.
- **No custom `Drop` required.** Capping *construction* depth at ≤256 means the derived recursive `Drop` never deep-walks — same resolution anamnesis chose. (An iterative `Drop` is the heavier alternative; depth-cap is simpler and matches #3585's spirit.)
- **Guard must stay O(1)/opcode.** A naive "deep-walk the value on every push" guard is itself an O(n²) CPU-DoS on a `Q`-chain. Port anamnesis's incremental `SlotMeta` depth + memo-only deep-size, not a per-push walk.
- **Maintainer awareness.** candle's `memo_get` already carries a refcounting TODO — the charge-before-clone is the minimal, welcome fix rather than a redesign.

---

## Cross-links to add when this lands

- `hf-fetch-model/docs/issues/candle-3617-p1.md` (archive the posted issue body; status **Posted**) → link to this plan.
- `anamnesis/ROADMAP.md` and the `### Security` CHANGELOG entry → note the upstream port.
- This plan → update **Status** to *PR opened (#NNNN)* once submitted.
