import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from uiir.graph import build_ui_graph


class GraphTests(unittest.TestCase):
    def test_builds_psd_aware_graph_edges_and_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGBA", (120, 80), (255, 255, 255, 255)).save(root / "composite.png")
            (root / "uiir.json").write_text(
                json.dumps(
                    {
                        "version": "0.1",
                        "source": "sample.psd",
                        "width": 120,
                        "height": 80,
                        "assetsRoot": "assets/",
                        "root": {"id": "n1", "type": "Screen", "bbox": {"x": 0, "y": 0, "w": 120, "h": 80}, "confidence": 1, "sourceRefs": ["document"], "children": []},
                    }
                ),
                encoding="utf-8",
            )
            (root / "layer_metadata.json").write_text(
                json.dumps(
                    [
                        {"id": "layer:group", "parent_id": None},
                        {"id": "layer:bg", "parent_id": "layer:group"},
                        {"id": "layer:text", "parent_id": "layer:group"},
                    ]
                ),
                encoding="utf-8",
            )
            (root / "candidates.json").write_text(
                json.dumps(
                    [
                        {"id": "c1", "bbox": {"x": 10, "y": 10, "w": 80, "h": 30}, "source": "psd-layer", "type_hint": "Image", "confidence": 0.9, "source_refs": ["layer:bg"]},
                        {"id": "c2", "bbox": {"x": 30, "y": 18, "w": 32, "h": 12}, "source": "psd-layer", "type_hint": "Text", "confidence": 0.9, "source_refs": ["layer:text"], "text": "OK"},
                        {"id": "c3", "bbox": {"x": 10, "y": 50, "w": 80, "h": 20}, "source": "visual-contour", "type_hint": "Unknown", "confidence": 0.4, "source_refs": ["visual:1"]},
                    ]
                ),
                encoding="utf-8",
            )

            result = build_ui_graph(root)

            self.assertTrue(result.graph_json.exists())
            self.assertTrue(result.graph_overlay.exists())
            edge_types = {edge["type"] for edge in result.graph["edges"]}
            self.assertIn("contains", edge_types)
            self.assertIn("text_on_image", edge_types)
            self.assertIn("component_group_candidate", edge_types)


if __name__ == "__main__":
    unittest.main()
