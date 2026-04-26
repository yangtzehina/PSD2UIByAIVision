import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from uiir.evaluate import evaluate_outputs


class EvaluateTests(unittest.TestCase):
    def test_evaluates_single_extract_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(out / "composite.png")
            (out / "uiir.json").write_text(
                json.dumps(
                    {
                        "version": "0.1",
                        "source": "sample.psd",
                        "width": 20,
                        "height": 20,
                        "assetsRoot": "assets/",
                        "root": {
                            "id": "n1",
                            "type": "Screen",
                            "bbox": {"x": 0, "y": 0, "w": 20, "h": 20},
                            "confidence": 1,
                            "sourceRefs": ["document"],
                            "children": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_outputs(out)

            self.assertEqual(report["count"], 1)
            self.assertEqual(report["schema_ok"], 1)
            self.assertTrue((out / "preview.png").exists())
            self.assertTrue((out / "replay_preview.png").exists())
            self.assertTrue((out / "diagnostic_overlay.png").exists())
            self.assertTrue((out / "metrics.json").exists())

    def test_replay_preview_skips_unrenderable_openai_proposal_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(out / "composite.png")
            (out / "uiir.json").write_text(
                json.dumps(
                    {
                        "version": "0.1",
                        "source": "sample.psd",
                        "width": 20,
                        "height": 20,
                        "assetsRoot": "assets/",
                        "metadata": {"documentKind": "screen"},
                        "root": {
                            "id": "n1",
                            "type": "Screen",
                            "bbox": {"x": 0, "y": 0, "w": 20, "h": 20},
                            "confidence": 1,
                            "sourceRefs": ["document"],
                            "children": [
                                {
                                    "id": "n2",
                                    "type": "Button",
                                    "bbox": {"x": 2, "y": 2, "w": 10, "h": 10},
                                    "confidence": 0.55,
                                    "sourceRefs": ["openai-vision:p1"],
                                    "metadata": {"source": "openai-vision-proposal"},
                                    "children": [],
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_outputs(out)

            self.assertEqual(report["items"][0]["visual"]["render_pixel_similarity"], 1.0)
            self.assertTrue((out / "diagnostic_overlay.png").exists())

    def test_asset_sheet_pixel_similarity_is_not_primary_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(out / "composite.png")
            (out / "uiir.json").write_text(
                json.dumps(
                    {
                        "version": "0.1",
                        "source": "ui.psd",
                        "width": 20,
                        "height": 20,
                        "assetsRoot": "assets/",
                        "metadata": {"documentKind": "asset_sheet"},
                        "root": {
                            "id": "n1",
                            "type": "Screen",
                            "bbox": {"x": 0, "y": 0, "w": 20, "h": 20},
                            "confidence": 1,
                            "sourceRefs": ["document"],
                            "children": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = evaluate_outputs(out)

            self.assertIsNone(report["avg_pixel_similarity"])
            self.assertFalse(report["items"][0]["visual"]["pixel_similarity_applicable"])
            self.assertIn("asset_sheet_render_pixel_similarity", report["items"][0]["visual"])

    def test_golden_metrics_include_proposals_and_decision_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out" / "sample"
            golden = root / "goldens" / "sample"
            out.mkdir(parents=True)
            golden.mkdir(parents=True)
            Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(out / "composite.png")
            uiir = {
                "version": "0.1",
                "source": "sample.psd",
                "width": 20,
                "height": 20,
                "assetsRoot": "assets/",
                "root": {
                    "id": "n1",
                    "type": "Screen",
                    "bbox": {"x": 0, "y": 0, "w": 20, "h": 20},
                    "confidence": 1,
                    "sourceRefs": ["document"],
                    "children": [
                        {
                            "id": "n2",
                            "type": "Button",
                            "bbox": {"x": 1, "y": 1, "w": 10, "h": 10},
                            "confidence": 0.9,
                            "sourceRefs": ["openai-vision:p1"],
                            "metadata": {"openaiComponentGroupId": "g1", "candidateId": "c1"},
                            "children": [],
                        }
                    ],
                },
            }
            (out / "uiir.json").write_text(json.dumps(uiir), encoding="utf-8")
            golden_uiir = {
                **uiir,
                "metadata": {"golden": {"decisions": {"loaded": 2, "accepted": 1, "edited": 0, "proposal_accepted": 1, "proposal_rejected": 1}}},
            }
            (golden / "uiir.json").write_text(json.dumps(golden_uiir), encoding="utf-8")

            report = evaluate_outputs(root / "out", golden_root=root / "goldens")

            metrics = report["items"][0]["golden"]
            self.assertEqual(metrics["proposal_precision"], 1.0)
            self.assertEqual(metrics["proposal_recall"], 1.0)
            self.assertEqual(metrics["relation_f1"], 1.0)
            self.assertEqual(metrics["human_accept_rate"], 0.5)
            self.assertEqual(metrics["quarantine_usefulness"], 0.5)


if __name__ == "__main__":
    unittest.main()
