import unittest

from uiir.models import BBox, Candidate
from uiir.openai_vision import VISION_CONFIDENCE_CAP, _tile_origins, apply_vision_proposals


class OpenAIVisionProposalTests(unittest.TestCase):
    def test_clamps_bbox_and_caps_confidence_for_new_candidate(self):
        candidates, accepted, quarantined, rejected, _ = apply_vision_proposals(
            {
                "items": [
                    _proposal("p1", {"x": -5, "y": 5, "w": 20, "h": 20}, "Button", confidence=0.99),
                    _proposal("p2", {"x": 1, "y": 1, "w": 2, "h": 2}, "Icon"),
                ],
                "merge_suggestions": [],
                "split_suggestions": [],
            },
            [],
            width=100,
            height=100,
            min_area=25,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].bbox, BBox(0, 5, 15, 20))
        self.assertEqual(candidates[0].confidence, VISION_CONFIDENCE_CAP)
        self.assertEqual(candidates[0].source, "openai-vision-proposal")
        self.assertEqual(accepted[0]["action"], "created")
        self.assertEqual(accepted[0]["status"], "accepted_candidate")
        self.assertEqual(quarantined, [])
        self.assertEqual(rejected[0]["rejectionReason"], "area_below_minimum")

    def test_overlapping_proposal_merges_into_existing_unknown_candidate(self):
        existing = Candidate(id="c1", bbox=BBox(10, 10, 50, 30), source="visual", type_hint="Unknown", confidence=0.4)

        candidates, accepted, quarantined, rejected, _ = apply_vision_proposals(
            {
                "items": [_proposal("p1", {"x": 12, "y": 12, "w": 46, "h": 26}, "Input", role="search")],
                "merge_suggestions": [],
                "split_suggestions": [],
            },
            [existing],
            width=100,
            height=100,
            min_area=25,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].type_hint, "Input")
        self.assertEqual(candidates[0].role, "search")
        self.assertEqual(accepted[0]["action"], "merged")
        self.assertEqual(quarantined, [])
        self.assertEqual(rejected, [])

    def test_tile_origin_is_added_to_proposal_bbox(self):
        candidates, _, _, _, _ = apply_vision_proposals(
            {
                "items": [_proposal("p1", {"x": 10, "y": 20, "w": 30, "h": 40}, "Icon")],
                "merge_suggestions": [],
                "split_suggestions": [],
            },
            [],
            width=500,
            height=500,
            min_area=25,
            tile_origin=(100, 200),
        )

        self.assertEqual(candidates[0].bbox, BBox(110, 220, 30, 40))

    def test_merge_and_split_suggestions_do_not_delete_original_refs(self):
        c1 = Candidate(id="c1", bbox=BBox(0, 0, 100, 40), source="psd-layer", type_hint="Image", source_refs=["layer:1"])
        c2 = Candidate(id="c2", bbox=BBox(20, 8, 40, 20), source="psd-layer", type_hint="Text", source_refs=["layer:2"])

        candidates, accepted, quarantined, rejected, relations = apply_vision_proposals(
            {
                "items": [],
                "merge_suggestions": [
                    {"component_group_id": "g1", "type": "Button", "candidate_ids": ["c1", "c2"], "reason": "button bg and label"}
                ],
                "split_suggestions": [
                    {
                        "candidate_id": "c1",
                        "reason": "contains smaller icon",
                        "items": [
                            {
                                "proposal_id": "s1",
                                "bbox": {"x": 70, "y": 8, "w": 20, "h": 20},
                                "type": "Icon",
                                "confidence": 0.8,
                                "text": "",
                                "role": "",
                                "reason": "visible icon",
                            }
                        ],
                    }
                ],
            },
            [c1, c2],
            width=200,
            height=100,
            min_area=25,
        )

        self.assertEqual(candidates[0].source_refs, ["layer:1"])
        self.assertEqual(candidates[1].source_refs, ["layer:2"])
        self.assertEqual(candidates[0].metadata["openaiComponentGroupId"], "g1")
        self.assertEqual(candidates[-1].source, "openai-vision-proposal")
        self.assertEqual(accepted[-1]["action"], "created")
        self.assertEqual(quarantined, [])
        self.assertEqual(rejected, [])
        self.assertTrue(relations["merge_suggestions"][0]["accepted"])
        self.assertEqual(relations["split_suggestions"][0]["candidate_id"], "c1")

    def test_audit_policy_quarantines_without_mutating_candidates(self):
        existing = Candidate(id="c1", bbox=BBox(10, 10, 50, 30), source="visual", type_hint="Unknown", confidence=0.4)

        candidates, accepted, quarantined, rejected, _ = apply_vision_proposals(
            {
                "items": [_proposal("p1", {"x": 12, "y": 12, "w": 46, "h": 26}, "Input", role="search")],
                "merge_suggestions": [],
                "split_suggestions": [],
            },
            [existing],
            width=100,
            height=100,
            min_area=25,
            vision_policy="audit",
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].type_hint, "Unknown")
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [])
        self.assertEqual(quarantined[0]["status"], "quarantined")
        self.assertEqual(quarantined[0]["quarantineReason"], "audit_policy")

    def test_strict_policy_quarantines_new_candidate_without_local_overlap(self):
        candidates, accepted, quarantined, rejected, _ = apply_vision_proposals(
            {
                "items": [_proposal("p1", {"x": 10, "y": 10, "w": 30, "h": 30}, "Button")],
                "merge_suggestions": [],
                "split_suggestions": [],
            },
            [],
            width=100,
            height=100,
            min_area=25,
            vision_policy="strict",
        )

        self.assertEqual(candidates, [])
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [])
        self.assertEqual(quarantined[0]["quarantineReason"], "strict_requires_local_overlap")

    def test_asset_sheet_quarantines_new_candidate_even_in_balanced_policy(self):
        candidates, accepted, quarantined, rejected, _ = apply_vision_proposals(
            {
                "items": [_proposal("p1", {"x": 10, "y": 10, "w": 30, "h": 30}, "Button")],
                "merge_suggestions": [],
                "split_suggestions": [],
            },
            [],
            width=100,
            height=100,
            min_area=25,
            vision_policy="balanced",
            document_kind="asset_sheet",
        )

        self.assertEqual(candidates, [])
        self.assertEqual(accepted, [])
        self.assertEqual(rejected, [])
        self.assertEqual(quarantined[0]["quarantineReason"], "asset_sheet_proposal_not_runtime_node")

    def test_tile_origins_cover_large_canvas_with_overlap(self):
        origins = _tile_origins(2800, 1500, tile_size=1400, overlap=0.1)

        self.assertIn((0, 0, 1400, 1400), origins)
        self.assertIn((1400, 100, 1400, 1400), origins)
        self.assertGreater(len(origins), 2)


def _proposal(proposal_id, bbox, node_type, confidence=0.8, role="", text=""):
    return {
        "proposal_id": proposal_id,
        "bbox": bbox,
        "type": node_type,
        "confidence": confidence,
        "text": text,
        "role": role,
        "reason": "unit test",
        "related_candidate_ids": [],
    }


if __name__ == "__main__":
    unittest.main()
