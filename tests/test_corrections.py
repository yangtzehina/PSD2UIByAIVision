import unittest

from uiir.corrections import apply_candidate_corrections
from uiir.models import BBox, Candidate


class CorrectionTests(unittest.TestCase):
    def test_applies_candidate_field_overrides(self):
        candidates = [
            Candidate(id="c1", bbox=BBox(0, 0, 10, 10), source="visual"),
            Candidate(id="c2", bbox=BBox(20, 20, 10, 10), source="visual"),
        ]

        corrected, summary = apply_candidate_corrections(
            candidates,
            [
                {
                    "candidate_id": "c1",
                    "bbox": {"x": -5, "y": 5, "w": 20, "h": 20},
                    "type": "Button",
                    "text": "OK",
                    "parent_id": "c2",
                }
            ],
            width=100,
            height=100,
        )

        self.assertEqual(len(corrected), 2)
        self.assertEqual(summary.applied, 1)
        self.assertEqual(corrected[0].bbox, BBox(0, 5, 15, 20))
        self.assertEqual(corrected[0].type_hint, "Button")
        self.assertEqual(corrected[0].text, "OK")
        self.assertEqual(corrected[0].parent_hint, "c2")

    def test_ignored_candidate_is_removed(self):
        candidates = [Candidate(id="c1", bbox=BBox(0, 0, 10, 10), source="visual")]
        corrected, summary = apply_candidate_corrections(candidates, [{"candidate_id": "c1", "ignored": True}])
        self.assertEqual(corrected, [])
        self.assertEqual(summary.ignored, 1)


if __name__ == "__main__":
    unittest.main()
