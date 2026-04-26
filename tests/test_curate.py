import json
import tempfile
import unittest
from pathlib import Path

from uiir.curate import curate_run


class CurateTests(unittest.TestCase):
    def test_curate_prioritizes_quarantine_and_render_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "run" / "openai" / "sample"
            sample.mkdir(parents=True)
            (sample / "vision_quarantined.json").write_text(json.dumps([{"id": "p1"}, {"id": "p2"}]), encoding="utf-8")
            (sample / "render_review.json").write_text(json.dumps({"status": "ok", "issue_count": 1, "issues": [{"type": "missing"}]}), encoding="utf-8")
            (sample / "ui_graph.json").write_text(json.dumps({"stats": {"edge_count": 8, "edge_type_counts": {"contains": 3}}}), encoding="utf-8")
            (root / "run" / "comparison.json").write_text(
                json.dumps(
                    {
                        "prompt_version": "semantic_v3",
                        "vision_policy": "strict",
                        "items": [
                            {
                                "name": "sample",
                                "openai_output": sample.as_posix(),
                                "document_kind": "screen",
                                "quarantined_proposals": 2,
                                "type_changes": [{"candidate_id": "c1"}],
                                "openai_golden": {"relation_f1": 0.5, "type_f1": 0.7},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = curate_run(root / "run", output_dir=root / "curation")

            self.assertEqual(report["count"], 1)
            self.assertGreater(report["queue"][0]["curation_value_score"], 0)
            self.assertIn("quarantined_proposals=2", report["queue"][0]["reasons"])
            self.assertTrue((root / "curation" / "curation_queue.json").exists())
            self.assertTrue((root / "curation" / "curation_queue.md").exists())


if __name__ == "__main__":
    unittest.main()
