import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from uiir.adapters import VISION_ADAPTERS, run_vision_adapter


class VisionAdapterTests(unittest.TestCase):
    def test_uied_adapter_writes_candidate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new("RGBA", (80, 60), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 12, 42, 34), fill=(20, 80, 160, 255))
            image.save(root / "composite.png")

            result = run_vision_adapter("uied", root, min_area=20, max_candidates=20)

            self.assertEqual(result.status, "ok")
            self.assertTrue(result.candidates_path.exists())
            self.assertTrue(result.manifest_path.exists())
            candidates = json.loads((root / "adapter_candidates.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "adapter_manifest.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["id"], "uied1")
            self.assertEqual(candidates[0]["source"], "adapter:uied")
            self.assertEqual(candidates[0]["metadata"]["adapter"], "uied")
            self.assertEqual(manifest["adapter"], "uied")
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["candidate_count"], len(candidates))
            self.assertEqual(manifest["outputs"]["candidates"], "adapter_candidates.json")
            self.assertFalse(manifest["weights_downloaded"])

    def test_heavy_adapters_are_skipped_with_dependency_notes(self):
        self.assertEqual(set(VISION_ADAPTERS), {"uied", "omniparser", "sam", "paddleocr"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            for adapter in ("omniparser", "sam", "paddleocr"):
                result = run_vision_adapter(adapter, root)
                manifest = json.loads((root / "adapter_manifest.json").read_text(encoding="utf-8"))
                candidates = json.loads((root / "adapter_candidates.json").read_text(encoding="utf-8"))

                self.assertEqual(result.status, "skipped")
                self.assertEqual(manifest["adapter"], adapter)
                self.assertEqual(manifest["status"], "skipped")
                self.assertEqual(manifest["candidate_count"], 0)
                self.assertIn("license", manifest["license_note"].lower())
                self.assertIn("never downloads weights", manifest["dependency_note"].lower())
                self.assertFalse(manifest["weights_downloaded"])
                self.assertEqual(manifest["downloads"], [])
                self.assertEqual(candidates, [])

    def test_invalid_adapter_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown vision adapter"):
                run_vision_adapter("not-real", Path(tmp))


if __name__ == "__main__":
    unittest.main()
