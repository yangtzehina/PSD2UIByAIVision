import unittest
from pathlib import Path

from uiir.detect import infer_uiir_document
from uiir.models import BBox, Candidate
from uiir.psd import PSDExtractResult


class InferDocumentTests(unittest.TestCase):
    def test_uses_candidate_parent_hint(self):
        document = infer_uiir_document(
            _extract(),
            [
                Candidate(id="c1", bbox=BBox(0, 0, 300, 300), source="psd-layer", type_hint="Container"),
                Candidate(id="c2", bbox=BBox(20, 20, 80, 40), source="psd-layer", type_hint="Button", parent_hint="c1"),
            ],
        )

        parent = document.root.children[0]
        self.assertEqual(parent.type, "Container")
        self.assertEqual(parent.children[0].type, "Button")

    def test_promotes_repeated_children_to_vertical_list(self):
        document = infer_uiir_document(
            _extract(),
            [
                Candidate(id="c1", bbox=BBox(0, 0, 300, 300), source="psd-layer", type_hint="Container"),
                Candidate(id="c2", bbox=BBox(20, 20, 80, 30), source="visual", type_hint="Image", parent_hint="c1"),
                Candidate(id="c3", bbox=BBox(20, 60, 82, 30), source="visual", type_hint="Image", parent_hint="c1"),
                Candidate(id="c4", bbox=BBox(20, 100, 78, 30), source="visual", type_hint="Image", parent_hint="c1"),
            ],
        )

        parent = document.root.children[0]
        self.assertEqual(parent.type, "List")
        self.assertEqual(parent.layout, "vertical")


def _extract() -> PSDExtractResult:
    return PSDExtractResult(
        source=Path("screen.psd"),
        width=400,
        height=400,
        composite_path=Path("composite.png"),
        assets_root=Path("assets"),
        layers=[],
    )


if __name__ == "__main__":
    unittest.main()
