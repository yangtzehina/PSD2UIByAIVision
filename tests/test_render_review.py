import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from uiir.provider import LLMProviderConfig
from uiir.render_review import review_render


class RenderReviewTests(unittest.TestCase):
    def test_render_review_quarantines_missing_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new("RGBA", (60, 40), (255, 255, 255, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 10, 40, 25), fill=(0, 0, 0, 255))
            image.save(root / "composite.png")
            (root / "uiir.json").write_text(
                json.dumps(
                    {
                        "version": "0.1",
                        "source": "sample.psd",
                        "width": 60,
                        "height": 40,
                        "assetsRoot": "assets/",
                        "metadata": {"documentKind": "screen"},
                        "root": {"id": "n1", "type": "Screen", "bbox": {"x": 0, "y": 0, "w": 60, "h": 40}, "confidence": 1, "sourceRefs": ["document"], "children": []},
                    }
                ),
                encoding="utf-8",
            )

            report = review_render(root)

            self.assertEqual(report["status"], "ok")
            self.assertGreaterEqual(report["issue_count"], 1)
            self.assertEqual(report["issues"][0]["type"], "missing")
            self.assertEqual(report["quarantine"][0]["status"], "quarantined")
            self.assertTrue((root / "render_diff.png").exists())
            self.assertTrue((root / "render_review.json").exists())

    def test_openai_render_review_skips_without_key(self):
        old_key = os.environ.pop("MISSING_PROVIDER_KEY", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                Image.new("RGBA", (20, 20), (255, 255, 255, 255)).save(root / "composite.png")
                (root / "uiir.json").write_text(
                    json.dumps(
                        {
                            "version": "0.1",
                            "source": "sample.psd",
                            "width": 20,
                            "height": 20,
                            "assetsRoot": "assets/",
                            "root": {"id": "n1", "type": "Screen", "bbox": {"x": 0, "y": 0, "w": 20, "h": 20}, "confidence": 1, "sourceRefs": ["document"], "children": []},
                        }
                    ),
                    encoding="utf-8",
                )

                report = review_render(
                    root,
                    use_openai=True,
                    provider=LLMProviderConfig(provider_name="third-party", api_key_env="MISSING_PROVIDER_KEY"),
                )

                self.assertEqual(report["openai"]["status"], "skipped")
                self.assertNotIn("token", json.dumps(report).lower())
        finally:
            if old_key is not None:
                os.environ["MISSING_PROVIDER_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
