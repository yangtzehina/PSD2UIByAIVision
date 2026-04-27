import json
import tempfile
import unittest
import uuid
from pathlib import Path

from uiir.closeout import CloseoutOptions, run_closeout


class CloseoutTests(unittest.TestCase):
    def test_dry_run_writes_planned_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "closeout"
            pattern = f"UNLIKELY_CLOSEOUT_PATTERN_{uuid.uuid4().hex}"
            report = run_closeout(
                CloseoutOptions(
                    output_dir=out,
                    dry_run=True,
                    skip_inspector_build=True,
                    sensitive_patterns=(pattern,),
                )
            )

            self.assertEqual(report["status"], "planned")
            self.assertTrue((out / "closeout_report.json").exists())
            self.assertTrue((out / "closeout_report.md").exists())
            self.assertTrue(all(item["status"] == "planned" for item in report["commands"]))
            self.assertEqual(report["sensitive_scan"]["pattern_count"], 1)
            self.assertEqual(report["sensitive_scan"]["match_count"], 0)

    def test_dry_run_report_redacts_sensitive_pattern_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "closeout"
            pattern = f"VERY_SECRET_LOCAL_PATTERN_{uuid.uuid4().hex}"
            run_closeout(
                CloseoutOptions(
                    output_dir=out,
                    dry_run=True,
                    skip_inspector_build=True,
                    sensitive_patterns=(pattern,),
                )
            )

            raw = (out / "closeout_report.json").read_text(encoding="utf-8")
            report = json.loads(raw)
            self.assertNotIn(pattern, raw)
            self.assertEqual(report["sensitive_scan"]["pattern_count"], 1)


if __name__ == "__main__":
    unittest.main()
