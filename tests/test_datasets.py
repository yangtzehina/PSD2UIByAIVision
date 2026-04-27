import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from uiir.datasets import import_rico_dataset, map_android_class_to_uiir_type


class RicoDatasetImportTests(unittest.TestCase):
    def test_imports_tiny_hierarchy_and_screenshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            Image.new("RGB", (120, 80), (255, 255, 255)).save(input_dir / "screen1.png")
            (input_dir / "screen1.json").write_text(
                json.dumps(
                    {
                        "class": "android.widget.FrameLayout",
                        "bounds": [0, 0, 120, 80],
                        "children": [
                            {
                                "class": "android.widget.TextView",
                                "bounds": [8, 6, 70, 24],
                                "text": "Hello",
                            },
                            {
                                "class": "android.widget.Button",
                                "bounds": [10, 40, 92, 70],
                                "text": "OK",
                                "clickable": True,
                            },
                            {
                                "class": "android.widget.EditText",
                                "bounds": [95, 40, 118, 70],
                                "hint": "Name",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = import_rico_dataset(input_dir, output_dir)

            self.assertEqual(manifest["count"], 1)
            uiir_path = output_dir / "goldens" / "screen1" / "uiir.json"
            self.assertTrue(uiir_path.exists())
            self.assertTrue((output_dir / "goldens" / "screen1" / "screenshot.png").exists())
            uiir = json.loads(uiir_path.read_text(encoding="utf-8"))
            self.assertEqual(uiir["width"], 120)
            self.assertEqual(uiir["height"], 80)
            self.assertEqual(uiir["root"]["type"], "Screen")
            frame = uiir["root"]["children"][0]
            self.assertEqual(frame["type"], "Container")
            self.assertEqual([child["type"] for child in frame["children"]], ["Text", "Button", "Input"])
            self.assertEqual(frame["children"][0]["text"], "Hello")
            self.assertEqual(frame["children"][1]["interaction"], "tap")
            self.assertEqual(frame["children"][2]["interaction"], "input")
            self.assertEqual(json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["count"], 1)

    def test_limit_counts_successful_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            for index in range(3):
                stem = f"screen{index}"
                Image.new("RGB", (10, 10), (255, 255, 255)).save(input_dir / f"{stem}.jpg")
                (input_dir / f"{stem}.json").write_text(
                    json.dumps({"class": "android.widget.TextView", "bounds": [0, 0, 10, 10], "text": stem}),
                    encoding="utf-8",
                )

            manifest = import_rico_dataset(input_dir, output_dir, limit=2)

            self.assertEqual(manifest["count"], 2)
            self.assertEqual(len(manifest["samples"]), 2)
            self.assertTrue((output_dir / "goldens" / "screen0" / "uiir.json").exists())
            self.assertTrue((output_dir / "goldens" / "screen1" / "uiir.json").exists())
            self.assertFalse((output_dir / "goldens" / "screen2" / "uiir.json").exists())

    def test_missing_screenshot_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "out"
            input_dir.mkdir()
            (input_dir / "screen1.json").write_text(
                json.dumps({"class": "android.widget.TextView", "bounds": [0, 0, 10, 10], "text": "No image"}),
                encoding="utf-8",
            )

            manifest = import_rico_dataset(input_dir, output_dir)

            self.assertEqual(manifest["count"], 0)
            self.assertEqual(manifest["skipped_count"], 1)
            self.assertEqual(manifest["skipped"][0]["reason"], "missing_screenshot")
            self.assertFalse((output_dir / "goldens" / "screen1" / "uiir.json").exists())

    def test_android_class_mapping_covers_common_widgets(self):
        self.assertEqual(map_android_class_to_uiir_type("android.widget.ImageButton"), "Button")
        self.assertEqual(map_android_class_to_uiir_type("androidx.recyclerview.widget.RecyclerView"), "List")
        self.assertEqual(map_android_class_to_uiir_type("android.widget.Switch"), "Toggle")
        self.assertEqual(map_android_class_to_uiir_type("android.widget.SeekBar"), "Slider")
        self.assertEqual(map_android_class_to_uiir_type("android.widget.HorizontalScrollView"), "ScrollView")


if __name__ == "__main__":
    unittest.main()
