# ADR-0008: Disposable source-stage mutation testing with stage-scoped Seatbelt writes and Linux confinement gate

- Status: accepted
- Date: 2026-11-15
- Deciders: Sein, Hermes Agent
- Supersedes: none
- Superseded by: none

## Context and problem

Mutation testing is the first TestBench capability whose *subject under test mutates the source
tree it is pointed at*. Both credible engines — `cargo-mutants` and `mutmut` — work by writing a
mutant into the project, running the suite, and recording whether the suite noticed. That is
fundamentally unlike every earlier adapter (pytest, cargo, FileCheck, GnuCOBOL, tapes): they only
read the checkout.

Three forces collide in Phase 3:

1. **`allow_repo_write=False` is the platform's strongest promise.** ADR-0005 made Seatbelt
   deny-by-default for writes, with the repository writable only when a suite explicitly asks.
   A mutation run that needs to write mutants appears to *require* relaxing that promise.
2. **Confined hosts and unconfined hosts.** `sandbox-exec` exists on macOS; on Linux there is no
   per-stage Seatbelt. Since mutants are arbitrary source edits, running the engine unconfined on
   a CI box means an untrusted mutation tool can rewrite anything the runner can reach.
3. **"Score" is the easiest number to lie with.** A naive `caught / tested` where `tested`
   counts every mutant the tool emitted is inflated by compile failures, timeouts and skipped
   mutants — a suite that kills nothing can score 90% on a target where most mutants never build.

Additionally the engine is a *dependency*, not something TestBench builds: mutant operators,
timeouts, baseline collection and outcome vocabularies differ per language and are maintained
upstream.

## Constraints and requirements

- The real checkout must never be mutated, and that must be *verifiable after the fact*, not only
  asserted: `git status --porcelain` is unchanged and no mutant artefact (`mutants`,
  `mutants.out`, `mutants.out.old`, `.mutmut-cache`) appears in the project root.
- `allow_repo_write=False` for the real repository stays the default for mutation suites; the
  relaxation may exist only for a TestBench-owned disposable directory.
- A mutation suite must never silently pass because the tool was missing, the baseline failed, or
  the platform could not confine the run. "Cannot measure" is not "clean".
- No environment variable may assert confinement. A variable is inherited evidence about *some*
  outer sandbox, never proof that this process can install its own per-stage profile.
- The gate that can fail a build must come from an operator or a trusted manifest — never from a
  file the agent under test can write.
- Outcomes must fit `SCHEMA_VERSION = 1` (`SuiteResult.metrics`, existing status vocabulary) with
  no DDL change (ADR-0001).
- `mutmut` reports per-mutant verdicts only through its cache, and `cargo-mutants` only through
  `mutants.out/outcomes.json`; both are tool-specific and must be normalised.

## Alternatives considered

| Alternative | Why it was weighed | Why it lost |
|---|---|---|
| Mutate the real checkout in place | Zero staging cost; the suite cwd already exists; `cargo-mutants` defaults to it | Breaks the "checkout is read-only" promise (ADR-0005) for every mutation run, and a `SIGKILL`/timeout mid-run leaves a mutated working tree behind. `git status` at the end cannot repair an operator's uncommitted work that a mutant overwrote. Also requires `allow_repo_write=True`, i.e. handing the sandbox the entire repository. |
| `allow_repo_write=True` on the project root | One-line change: the tool just works | Grants write to everything under the root — `~/.ssh`-style escalations are out of reach, but the operator's `.env`, fixtures and `tests/` become writable by a third-party binary. Rejected: the write surface can be exactly one directory, so it should be. |
| Write a custom mutation engine (AST rewriting in `hermes_testbench`) | Full control over operators, buckets and score; no external binaries | Re-implements per-language parsers and operator sets that upstream projects already maintain, and a home-grown engine's *unviable mutant* rate is an unknown quantity — exactly the number the score depends on. Out of scope for a harness whose value is evidence quality, not mutant generation. |
| Trust `TESTBENCH_SEATBELT_ACTIVE` (or similar) as proof of confinement | Lets CI run mutation suites on Linux by declaring an outer sandbox | The variable proves nothing about *this* child: the child is spawned by `run_isolated_command`, which builds the profile. On a host without `sandbox-exec` the variable is trivially set by anyone (including the agent), so it converts a hard gate into a self-certification. Rejected outright; the capability probe (`can_apply_seatbelt()`) is the only accepted evidence. |
| Score as `(caught + timeout) / tested` (count timeouts as kills) | Timeouts often mean "the mutant hung the suite", which sounds like detection | A timeout is a harness symptom (unbounded loop, build hang, per-mutant budget too small), not evidence the assertions caught the change. Counting it as `caught` lets a suite with no assertions score highly on a target whose mutants hang. Timeouts stay in their own bucket and are excluded from the score. |
| Single "mutants killed" counter, gate on the raw count | Simplest metric to display | Not comparable across targets or commits: the mutant count depends on source size and operator set. A fraction over *viable and completed* mutants is the only gate that means the same thing twice. |

## Decision

1. **Mutate a disposable copy, never the checkout.** `stage_project_tree(project_root, stage_dir)`
   copies the project sources into `scratch_root / f"tbmut-{nonce}"` (skipping `.git`, `.venv`,
   `target`, `node_modules`, caches and prior mutant artefacts), and the mutation tool's cwd is
   resolved *inside* that stage. The real checkout is compared before/after — `git status` diff
   plus sentinel scan — and any difference ends the suite `ERROR`/`harness_error`
   (`_untouched_violations`). The stage is removed in a `finally` via
   `_safe_rmtree_no_symlinks`.
2. **The write relaxation is internal and stage-scoped.** `run_isolated_command(...,
   allow_repo_write=False, extra_writable_dirs=(stage_dir,))`. `extra_writable_dirs` is marked
   *internal only* in `isolation.py` — it is never populated from manifest input — and its entries
   are added to the Seatbelt `file-write*` allow list in addition to the sandbox dir. Net effect:
   the tool can write its mutants, the repository stays read-only under the same profile.
3. **Confinement is a hard gate, not a hint.** `can_apply_seatbelt()` is checked before any tool
   resolution or staging; when false the suite returns `BLOCKED` with
   `FailureKind.SANDBOX_DENIED` and "confinement unavailable". No env-var bypass exists. On Linux
   CI this is the documented, visible outcome — mutation testing is unavailable there until a
   Linux confinement backend exists, rather than silently unconfined.
4. **Exactly five outcome buckets.** `caught`, `survived`, `invalid` (unviable/compile failure),
   `timeout`, `inconclusive` (baseline failure, no tests, skipped, interrupted, not checked,
   harness error). `parsers.parse_cargo_mutants` (from `mutants.out/outcomes.json`, falling back to
   console tallies) and `parsers.parse_mutmut` (from `mutmut results --all true`, falling back to
   progress tallies) both normalise into `MUTATION_BUCKETS`, and `metrics["mutation"]` exposes them
   with `tested = caught + survived`.
5. **Score is `caught / (caught + survived)`.** `_score()` returns `None` when nothing viable was
   tested — which the gate reports as "cannot be evaluated" and fails, never as a pass.
6. **Gates must be trusted.** `--mutation-gate` (validated as a fraction in `[0, 1]`, `nan`
   rejected both in `cli.py` and `_apply_mutation_gate`) is stamped `mutation_gate_from_cli=True`.
   A manifest `mutation_score_gate` only counts when the manifest is trusted; from an untrusted
   manifest the suite is `BLOCKED` with a message telling the operator to `testbench trust` or pass
   the flag. Surviving mutants without a gate are findings in evidence, and the suite passes.
7. **Defense in depth on the staged cwd.** The declared suite cwd is rejected lexically before
   staging (absolute paths, `..` traversal), and — because `stage_project_tree` copies symlinks
   verbatim — re-verified *after* staging: the staged entry must not be a symlink and its
   `realpath` must remain inside the `realpath` of the stage, else the stage is removed and the
   suite ends `ERROR`/`harness_error`.

## Rationale

Staging is the only option that keeps ADR-0005 intact while letting a genuinely destructive tool
run: the write surface shrinks from "the repository" to "one throwaway directory that TestBench
created and deletes". Delegating mutant generation keeps TestBench's evidence trustworthy — it
never has to defend its own operator set — while normalising the two tools' vocabularies into five
buckets makes `cargo-mutants` and `mutmut` results comparable and makes "we could not measure this"
a first-class, visible outcome instead of a silent pass. A capability probe as the gate means the
unsupported platform fails loudly and identically for every caller, which is the property a CI
gate needs.

## Tradeoffs and consequences

- **Mutation testing does not run on Linux CI.** Mutation suites there report `BLOCKED` /
  `sandbox_denied`. That is deliberate and visible; a Linux confinement backend (namespaces,
  bubblewrap, Landlock) is future work, and this ADR does not pre-approve one.
- Staging costs a full source copy per run and doubles peak disk for large trees (`target/` and
  `.venv` are skipped, which mitigates but does not eliminate this). Rust targets recompile from
  scratch inside the stage unless a shared `CARGO_TARGET_DIR` is arranged.
- The score excludes `timeout` and `invalid`, so a target whose mutants mostly fail to build
  reports a score over a small `tested` denominator; `tested`, `invalid` and `timeout` are exposed
  alongside it precisely so a reader can see that.
- Symlinked project layouts are constrained: a cwd that resolves through a symlink is rejected
  rather than followed, which is a usability cost paid for determinism.
- The tool vocabulary is upstream's. A `mutmut` status TestBench does not recognise lands in
  `inconclusive` (never `caught`), which can only understate a score.
- `mutmut results --all true` is a second confined spawn per run; it costs a short extra process
  and is bounded to 120 s.

## Supporting evidence

- `hermes_testbench/adapters/mutation.py` — `MUTATION_ADAPTERS`, `MUTATION_BUCKETS`,
  `SENTINEL_NAMES`, `stage_project_tree`, `_stage_ignore`, `resolve_tool`, `_score`, `_metrics`,
  `run_mutation_suite` (gates A/B/C, post-staging symlink re-check, `_untouched_violations`,
  `finally: _safe_rmtree_no_symlinks`).
- `hermes_testbench/isolation.py` — `can_apply_seatbelt()`, `build_seatbelt_profile(...,
  extra_writable_dirs=...)` joined into the `file-write*` allow list, and the "Internal only"
  comment on `run_isolated_command(extra_writable_dirs=...)`.
- `hermes_testbench/adapters/parsers.py` — `_MUTMUT_STATUS_BUCKET`, `_MUTMUT_EMOJI_BUCKET`,
  `parse_cargo_mutants`, `parse_mutmut`, `_sync_mutation_counts`, `_mark_baseline_failed`.
- `hermes_testbench/engine.py` — `_apply_mutation_gate` (fraction validation, CLI gate is trusted
  by construction) and `run_sync(mutation_gate=...)`.
- `hermes_testbench/schema.py` — `mutation_score_gate`, `mutation_gate_from_cli`.
- `tests/test_mutation_testing.py` — real `mutmut` and `cargo-mutants` caught+survived runs on a
  disposable copy, the CLI gate failing on a known survivor, checkout-pristine assertions,
  `test_suite_cwd_cannot_escape_the_staged_copy`,
  `test_symlinked_staged_cwd_is_rejected_after_staging`,
  `test_spoofed_seatbelt_env_var_still_blocks_when_seatbelt_unavailable`,
  `test_stage_dir_is_writable_while_real_project_root_is_read_only`.

## Verification status

verified — `tests.test_mutation_testing`: 28 tests, `skip=0`, on Python 3.14 and 3.11 with
`mutmut 3.8.1` and `cargo-mutants 27.1.0` on `PATH` (real caught+survived end-to-end runs, real
Seatbelt staging). Simulated Linux host (`sandbox-exec` hidden from `shutil.which`): same 28 tests,
0 failures, 0 errors, 6 capability-gated skips; every mutation suite reports
`BLOCKED`/`sandbox_denied` and nothing is spawned. Local profiles: `unit` (216 tests), `security`
(14), `terminal` (151), `regression` (420) all PASS — see the Phase 3 PR for the run IDs.

## Related decisions

- ADR-0001 — single stdlib core engine and WAL SQLite store (schema stayed at `SCHEMA_VERSION = 1`).
- ADR-0005 — deny-by-default Seatbelt writes; this ADR adds one internal, stage-scoped exception
  rather than a suite-visible `allow_repo_write`.
- ADR-0007 — per-attempt RunRecords; mutation suites reuse the same one-suite-one-`SuiteResult`
  shape, so a mutation run is a normal run with extra metrics.
