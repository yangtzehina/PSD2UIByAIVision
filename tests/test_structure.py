import unittest
from pathlib import Path

from uiir.detect import infer_uiir_document
from uiir.models import BBox, Candidate
from uiir.psd import PSDExtractResult


class StructureMappingTests(unittest.TestCase):
    def test_psd_siblings_are_lifted_into_button_components(self):
        document = infer_uiir_document(
            _extract(),
            [
                _candidate("c1", "Background", BBox(0, 0, 500, 400), "layer:1", name="背景"),
                _candidate("c2", "Button", BBox(40, 40, 140, 48), "layer:2", parent="layer:1", name="按钮背景"),
                _candidate("c3", "Text", BBox(78, 54, 52, 18), "layer:3", parent="layer:1", name="按钮文字", text="确定"),
                _candidate("c4", "Button", BBox(40, 110, 140, 48), "layer:4", parent="layer:1", name="按钮2背景"),
                _candidate("c5", "Text", BBox(260, 40, 100, 24), "layer:5", parent="layer:1", name="文本", text="说明"),
            ],
        )

        background = document.root.children[0]
        buttons = [child for child in background.children if child.type == "Button"]
        plain_text = [child for child in background.children if child.type == "Text"]

        self.assertEqual(background.type, "Background")
        self.assertEqual(len(buttons), 2)
        self.assertEqual(len(plain_text), 1)
        self.assertEqual(buttons[0].bbox, BBox(40, 40, 140, 48))
        self.assertEqual(buttons[0].metadata["groupingReason"], "background_text_component")
        self.assertEqual(set(buttons[0].source_refs), {"layer:2", "layer:3"})
        self.assertEqual([child.type for child in buttons[0].children], ["Background", "Text"])
        self.assertEqual(set(buttons[1].source_refs), {"layer:4"})
        self.assertEqual(buttons[1].metadata["groupingReason"], "single_component_layer")
        self.assertEqual(buttons[1].children[0].source_refs, ["layer:4"])


def _candidate(candidate_id, node_type, bbox, layer_id, parent=None, name="", text=None):
    return Candidate(
        id=candidate_id,
        bbox=bbox,
        source="psd-layer",
        type_hint=node_type,
        confidence=0.9,
        source_refs=[layer_id],
        name=name,
        text=text,
        parent_hint=parent,
        metadata={
            "name": name,
            "psdParentId": parent,
            "psdPath": f"背景/{name}" if parent else name,
            "psdDepth": 1 if parent else 0,
        },
    )


def _extract():
    return PSDExtractResult(
        source=Path("mock.psd"),
        width=500,
        height=400,
        composite_path=Path("composite.png"),
        assets_root=Path("assets"),
        layers=[],
    )


if __name__ == "__main__":
    unittest.main()
