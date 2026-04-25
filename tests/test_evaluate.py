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
            self.assertTrue((out / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
