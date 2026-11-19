"""Contract tests for ``hermes_testbench.schema``: import purity and dict round-trip fidelity.

Schema imports none of ``sanitize``, ``reporting`` or ``store``; the arrows point *into*
schema — ``reporting`` and ``store`` import it, then ``isolation``, then
``adapters``/``engine``, then ``cli``/``mcp_server``/``tui``. Because every other module
depends on schema, two properties are worth a test rather than a review comment:

* it stays importable and side-effect free — no regex engine, XML writer, SQLite, or
  process/terminal plumbing;
* the record survives ``run_to_dict``/``run_from_dict`` unchanged, because those two
  functions *are* the ``payload_json`` codec and the store reads nothing else back.

There is no I/O and no subprocess here: both tests are pure, so they run everywhere and
cannot fail for environmental reasons.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from hermes_testbench import schema
from hermes_testbench.schema import (
    CaseResult,
    FailureRecord,
    RunRecord,
    RunStatus,
    SuiteResult,
    new_run_id,
    run_from_dict,
    run_to_dict,
    utc_now_iso,
)


class SchemaContractTests(unittest.TestCase):
    """Two properties of the contract module: it imports nothing impure, and its payload codec is lossless.

    Nothing here touches the store: the point is that the schema can be imported and exercised
    with neither a filesystem nor a process, which is what makes it usable from every layer.
    """
    def test_schema_module_has_zero_regex_xml_sqlite_or_subprocess_imports(self) -> None:
        """``schema.py`` imports none of the modules that would make it non-pure.

        The forbidden set is not stylistic: ``re``/``xml`` pull compilation in at import time,
        and ``sqlite3``/``subprocess``/``pty``/``termios``/``select``/``fcntl`` would give the
        data layer the ability to touch files, processes or a terminal — which is how a "pure"
        module quietly acquires ambient state and stops being importable in a stripped runtime.

        Walking ``ast`` rather than grepping text means only real imports count: a module named
        in a docstring or an attribute called ``re`` cannot trip it, and an ``import`` cannot hide
        behind formatting. The failure message reports the intersection, so it names the
        offending module.
        """
        src = Path(schema.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"re", "xml", "sqlite3", "subprocess", "pty", "termios", "select", "fcntl"}
        self.assertEqual(imported & forbidden, set(), f"schema.py imported forbidden modules: {imported & forbidden}")

    def test_run_record_dict_roundtrip_preserves_none_model_usage_and_nested_records(self) -> None:
        """``run_to_dict`` → ``run_from_dict`` preserves nested records and a ``None`` usage.

        Two discriminators, each aimed at a specific way a codec breaks:

        * the nested failures/cases are read back as *attributes*
          (``restored.suites[0].failures[0].case_name``). ``asdict`` produces plain dicts, so a
          ``run_from_dict`` that skipped rehydration would raise ``AttributeError`` here rather
          than quietly returning dicts — this asserts real ``FailureRecord``/``CaseResult``
          construction, not just equal values.
        * ``model_usage`` is ``None`` on both sides. ``None`` is the store's "usage was never
          measured" marker and what the UI reads to decide whether to show a cost; coercing it to
          ``{}`` or ``0`` en route would turn "unknown" into "free". ``run_from_dict`` re-sets the
          key explicitly for that reason.

        ``rid``/``ts`` also pin the two generator contracts the payload depends on: ids start
        ``tb_`` and timestamps end in ``Z``, the form the lexicographic ordering keys assume.
        """
        rid = new_run_id()
        self.assertTrue(rid.startswith("tb_"))
        ts = utc_now_iso()
        self.assertTrue(ts.endswith("Z"))
        run = RunRecord(
            run_id=rid,
            project="hxmail",
            project_root="/Users/sein/Code/hermes-terminal-env",
            revision="8234029",
            profile="unit",
            status=RunStatus.FAIL,
            created_at=ts,
            total_count=2,
            pass_count=1,
            fail_count=1,
            suites=[
                SuiteResult(
                    name="core",
                    adapter="cargo",
                    profile="unit",
                    command=["cargo", "test"],
                    cwd="/Users/sein/Code/hermes-terminal-env",
                    status=RunStatus.FAIL,
                    exit_code=101,
                    total_count=2,
                    pass_count=1,
                    fail_count=1,
                    failures=[FailureRecord(suite="core", case_name="t_bad", message="boom")],
                    cases=[CaseResult(name="t_bad", suite="core", status=RunStatus.FAIL, duration_ms=3.5)],
                )
            ],
        )
        as_dict = run_to_dict(run)
        self.assertIsNone(as_dict["model_usage"])
        restored = run_from_dict(as_dict)
        self.assertEqual(restored.run_id, rid)
        self.assertIsNone(restored.model_usage)
        self.assertEqual(restored.suites[0].failures[0].case_name, "t_bad")
        self.assertEqual(restored.suites[0].cases[0].status, RunStatus.FAIL)


if __name__ == "__main__":
    unittest.main()
