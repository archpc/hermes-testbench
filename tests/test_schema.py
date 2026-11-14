"""TDD tests for pure contract behavior in hermes_testbench.schema."""

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
    def test_schema_module_has_zero_regex_xml_sqlite_or_subprocess_imports(self) -> None:
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
