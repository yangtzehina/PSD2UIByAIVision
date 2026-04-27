import unittest

from uiir.models import BBox, Candidate
from uiir.openai_semantics import _merge_semantic_type


class OpenAISemanticMergeTests(unittest.TestCase):
    def test_screen_type_is_reserved_for_synthetic_root(self):
        candidate = _candidate("Container", 0.9)
        patch = {"accepted": {}, "rejected": []}

        _merge_semantic_type(candidate, "Screen", patch)

        self.assertEqual(candidate.type_hint, "Container")
        self.assertEqual(candidate.metadata["openaiRejected"][0]["reason"], "screen_is_synthetic_root")
        self.assertEqual(patch["rejected"][0]["reason"], "screen_is_synthetic_root")

    def test_unknown_does_not_downgrade_high_confidence_local_type(self):
        candidate = _candidate("Text", 0.92)
        patch = {"accepted": {}, "rejected": []}

        _merge_semantic_type(candidate, "Unknown", patch)

        self.assertEqual(candidate.type_hint, "Text")
        self.assertEqual(candidate.metadata["openaiRejected"][0]["reason"], "unknown_downgrade_blocked")
        self.assertEqual(patch["rejected"][0]["reason"], "unknown_downgrade_blocked")

    def test_unknown_does_not_downgrade_low_confidence_local_type(self):
        candidate = _candidate("Container", 0.58)

        _merge_semantic_type(candidate, "Unknown")

        self.assertEqual(candidate.type_hint, "Container")
        self.assertEqual(candidate.metadata["openaiRejected"][0]["reason"], "unknown_downgrade_blocked")

    def test_specific_openai_type_can_refine_local_unknown(self):
        candidate = _candidate("Unknown", 0.45)
        patch = {"accepted": {}, "rejected": []}

        _merge_semantic_type(candidate, "Button", patch)

        self.assertEqual(candidate.type_hint, "Button")
        self.assertEqual(patch["accepted"]["type"], "Button")

    def test_text_candidate_is_not_reclassified_as_image(self):
        candidate = _candidate("Text", 0.9)
        patch = {"accepted": {}, "rejected": []}

        _merge_semantic_type(candidate, "Image", patch)

        self.assertEqual(candidate.type_hint, "Text")
        self.assertEqual(candidate.metadata["openaiRejected"][0]["reason"], "text_type_preserved")
        self.assertEqual(patch["rejected"][0]["reason"], "text_type_preserved")

    def test_interactive_candidate_is_not_reclassified_as_another_interactive_type(self):
        candidate = _candidate("Button", 0.88)
        patch = {"accepted": {}, "rejected": []}

        _merge_semantic_type(candidate, "Toggle", patch)

        self.assertEqual(candidate.type_hint, "Button")
        self.assertEqual(patch["rejected"][0]["reason"], "interactive_type_change_blocked")

    def test_high_confidence_cross_family_change_is_rejected(self):
        candidate = _candidate("ScrollView", 0.9)
        patch = {"accepted": {}, "rejected": []}

        _merge_semantic_type(candidate, "Slider", patch)

        self.assertEqual(candidate.type_hint, "ScrollView")
        self.assertEqual(patch["rejected"][0]["reason"], "cross_family_type_change_blocked")

    def test_container_to_button_requires_component_evidence(self):
        candidate = _candidate("Container", 0.62)
        patch = {"accepted": {}, "rejected": []}

        _merge_semantic_type(candidate, "Button", patch)

        self.assertEqual(candidate.type_hint, "Container")
        self.assertEqual(patch["rejected"][0]["reason"], "container_to_interactive_requires_group_evidence")

    def test_container_to_button_allows_component_evidence(self):
        candidate = _candidate("Container", 0.62)
        candidate.metadata["openaiComponentGroupId"] = "g1"
        patch = {"accepted": {}, "rejected": []}

        _merge_semantic_type(candidate, "Button", patch)

        self.assertEqual(candidate.type_hint, "Button")
        self.assertEqual(patch["accepted"]["type"], "Button")


def _candidate(node_type: str, confidence: float) -> Candidate:
    return Candidate(
        id="c1",
        bbox=BBox(0, 0, 100, 40),
        source="psd-layer",
        type_hint=node_type,
        confidence=confidence,
        source_refs=["layer:1"],
    )


if __name__ == "__main__":
    unittest.main()
