import unittest
from pathlib import Path

from uiir.document_kind import classify_document_kind, resolve_document_kind
from uiir.models import BBox, LayerRecord


class DocumentKindTests(unittest.TestCase):
    def test_classifies_named_ui_asset_sheet(self):
        layers = [
            _layer("UI", "UI", is_group=True),
            _layer("Common", "UI/Common", is_group=True),
            _layer("Buttons", "UI/Common/Buttons", is_group=True),
            _layer("Icons", "UI/Common/Icons", is_group=True),
            _layer("Slider", "UI/Common/Slider", is_group=True),
        ]

        self.assertEqual(classify_document_kind(Path("ui.psd"), layers), "asset_sheet")

    def test_classifies_screen_when_no_asset_palette_signals(self):
        layers = [
            _layer("background", "background"),
            _layer("dialog", "dialog", is_group=True),
            _layer("confirm text", "dialog/confirm text", text="OK"),
        ]

        self.assertEqual(classify_document_kind(Path("interface.psd"), layers), "screen")

    def test_explicit_kind_wins_over_auto(self):
        self.assertEqual(resolve_document_kind("screen", Path("ui.psd"), []), "screen")


def _layer(name, path, is_group=False, text=None):
    return LayerRecord(
        id=f"layer:{name}",
        name=name,
        path=path,
        kind="group" if is_group else "pixel",
        bbox=BBox(0, 0, 10, 10),
        is_group=is_group,
        text=text,
    )


if __name__ == "__main__":
    unittest.main()
