import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from uiir.focus import build_focus_tiles


class FocusTileTests(unittest.TestCase):
    def test_crop_bbox_padding_clamps_to_image_and_records_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGBA", (100, 80), (255, 255, 255, 255)).save(root / "composite.png")
            (root / "render_review.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "issues": [
                            {
                                "id": "rr:1",
                                "type": "missing",
                                "bbox": {"x": 85, "y": 70, "w": 20, "h": 20},
                                "overlapping_nodes": ["n1", "n2"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "ui_graph.json").write_text(json.dumps({"nodes": []}), encoding="utf-8")

            report = build_focus_tiles(root, padding=10)

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["tile_count"], 1)
            self.assertEqual(report["tiles"][0]["bbox"], {"x": 75, "y": 60, "w": 25, "h": 20})
            self.assertEqual(report["tiles"][0]["source_issue"]["id"], "rr:1")
            self.assertEqual(report["tiles"][0]["source_issue"]["type"], "missing")
            self.assertEqual(report["tiles"][0]["related_nodes"], ["n1", "n2"])
            graph_path = (root / "ui_graph.json").resolve().as_posix()
            self.assertEqual(report["graph_metadata_path"], graph_path)
            self.assertEqual(report["tiles"][0]["graph_metadata_path"], graph_path)

            tile_path = Path(report["tiles"][0]["path"])
            self.assertTrue(tile_path.exists())
            with Image.open(tile_path) as tile:
                self.assertEqual(tile.size, (25, 20))
            self.assertTrue((root / "focus_tiles.json").exists())

    def test_no_eligible_issues_writes_empty_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGBA", (50, 40), (255, 255, 255, 255)).save(root / "composite.png")
            (root / "render_review.json").write_text(
                json.dumps({"status": "ok", "issues": [{"id": "rr2", "type": "bad_parent", "bbox": {"x": 0, "y": 0, "w": 50, "h": 40}}]}),
                encoding="utf-8",
            )

            report = build_focus_tiles(root)

            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["eligible_issue_count"], 0)
            self.assertEqual(report["tile_count"], 0)
            self.assertEqual(report["tiles"], [])
            self.assertTrue((root / "focus_tiles.json").exists())

    def test_max_tile_cap_limits_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGBA", (120, 80), (255, 255, 255, 255)).save(root / "composite.png")
            issues = [
                {"id": f"rr{i}", "type": "extra", "bbox": {"x": i * 3, "y": i * 2, "w": 5, "h": 5}, "overlapping_nodes": []}
                for i in range(8)
            ]
            (root / "render_review.json").write_text(json.dumps({"status": "ok", "issues": issues}), encoding="utf-8")

            report = build_focus_tiles(root, padding=0, max_tiles=3)

            self.assertEqual(report["eligible_issue_count"], 8)
            self.assertEqual(report["tile_count"], 3)
            self.assertTrue(report["truncated"])
            self.assertEqual([tile["source_issue"]["id"] for tile in report["tiles"]], ["rr0", "rr1", "rr2"])
            self.assertEqual(len(list((root / "focus_tiles").glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()
