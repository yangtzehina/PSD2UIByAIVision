import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uiir.fidelity import build_parser_fidelity_report


class ParserFidelityTests(unittest.TestCase):
    def test_counts_parser_fidelity_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sample_extract(root)

            report = build_parser_fidelity_report(root)

            self.assertEqual(report["counts"]["layers"], 5)
            self.assertEqual(report["counts"]["groups"], 1)
            self.assertEqual(report["counts"]["text_layers"], 2)
            self.assertEqual(report["counts"]["styled_text_layers"], 1)
            self.assertEqual(report["counts"]["smart_object_ish_layers"], 1)
            self.assertEqual(report["counts"]["assets"], 2)
            self.assertEqual(report["counts"]["layer_effects"], 2)
            self.assertEqual(report["counts"]["warnings"], 2)
            self.assertEqual(report["counts"]["candidates"], 3)
            self.assertEqual(report["counts"]["uiir_nodes"], 4)
            self.assertEqual(report["warnings"]["top_level"], 1)
            self.assertEqual(report["warnings"]["layer_level"], 1)
            self.assertEqual(report["psd_tools_coverage"]["candidate_known_layer_refs"], 3)
            self.assertEqual(report["psd_tools_coverage"]["uiir_known_layer_refs"], 3)
            self.assertEqual(report["psd_tools_coverage"]["candidate_layer_coverage"], 0.6)
            self.assertEqual(report["psd_tools_coverage"]["uiir_layer_coverage"], 0.6)
            self.assertEqual(report["psd_tools_coverage"]["asset_coverage"], 0.66667)
            self.assertEqual(report["psd_tools_coverage"]["text_style_coverage"], 0.5)

            written = json.loads((root / "parser_fidelity.json").read_text(encoding="utf-8"))
            self.assertEqual(written["counts"], report["counts"])

    def test_optional_photoshopapi_probe_is_redacted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_sample_extract(root)

            with patch("importlib.util.find_spec", side_effect=RuntimeError("secret-token /tmp/private/plugin.py")):
                report = build_parser_fidelity_report(root, probe_photoshopapi=True)

            probe = report["photoshopapi_probe"]
            self.assertEqual(probe["status"], "not_available")
            self.assertFalse(probe["available"])
            serialized_probe = json.dumps(probe)
            self.assertNotIn("secret-token", serialized_probe)
            self.assertNotIn("/tmp/private", serialized_probe)
            self.assertNotIn("plugin.py", serialized_probe)


def _write_sample_extract(root: Path) -> None:
    (root / "layer_metadata.json").write_text(
        json.dumps(
            {
                "source": "sample.psd",
                "width": 200,
                "height": 100,
                "warnings": ["top warning"],
                "layers": [
                    {
                        "id": "layer:1",
                        "name": "Group",
                        "kind": "group",
                        "bbox": {"x": 0, "y": 0, "w": 200, "h": 100},
                        "visible": True,
                        "is_group": True,
                    },
                    {
                        "id": "layer:2",
                        "name": "Title",
                        "kind": "type",
                        "bbox": {"x": 10, "y": 10, "w": 80, "h": 20},
                        "visible": True,
                        "is_group": False,
                        "text": "Hello",
                        "style": {"fontSize": 18, "color": "#ffffff"},
                        "asset": "assets/layers/title.png",
                        "warnings": ["text style partial"],
                    },
                    {
                        "id": "layer:3",
                        "name": "Caption",
                        "kind": "type",
                        "bbox": {"x": 10, "y": 40, "w": 60, "h": 16},
                        "visible": True,
                        "is_group": False,
                        "text": "World",
                        "style": {},
                    },
                    {
                        "id": "layer:4",
                        "name": "Hero Smart Object",
                        "kind": "smartobject",
                        "bbox": {"x": 100, "y": 10, "w": 50, "h": 50},
                        "visible": True,
                        "is_group": False,
                        "asset": "assets/layers/hero.png",
                        "effects": [{"type": "dropShadow"}, {"type": "stroke"}],
                    },
                    {
                        "id": "layer:5",
                        "name": "Hidden",
                        "kind": "pixel",
                        "bbox": {"x": 0, "y": 0, "w": 20, "h": 20},
                        "visible": False,
                        "is_group": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "candidates.json").write_text(
        json.dumps(
            [
                {"id": "c1", "source_refs": ["layer:2"], "asset": "assets/layers/title.png"},
                {"id": "c2", "source_refs": ["layer:3"]},
                {"id": "c3", "source_refs": ["layer:4"]},
            ]
        ),
        encoding="utf-8",
    )
    (root / "uiir.json").write_text(
        json.dumps(
            {
                "version": "0.1",
                "source": "sample.psd",
                "width": 200,
                "height": 100,
                "assetsRoot": "assets/",
                "root": {
                    "id": "n1",
                    "type": "Screen",
                    "bbox": {"x": 0, "y": 0, "w": 200, "h": 100},
                    "confidence": 1,
                    "sourceRefs": ["document"],
                    "children": [
                        {"id": "n2", "type": "Text", "sourceRefs": ["layer:2"], "children": []},
                        {"id": "n3", "type": "Text", "sourceRefs": ["layer:3"], "children": []},
                        {
                            "id": "n4",
                            "type": "Image",
                            "sourceRefs": ["layer:4"],
                            "asset": "assets/layers/hero.png",
                            "children": [],
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
