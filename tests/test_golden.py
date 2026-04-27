import json
import tempfile
import unittest
from pathlib import Path

from uiir.golden import apply_golden_decisions, load_golden_decisions
from uiir.models import BBox, Candidate


class GoldenDecisionTests(unittest.TestCase):
    def test_accept_quarantined_proposal_creates_human_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "vision_quarantined.json").write_text(
                json.dumps(
                    [
                        {
                            "proposal_id": "p1",
                            "bbox": {"x": -4, "y": 3, "w": 20, "h": 10},
                            "type": "Button",
                            "confidence": 0.55,
                            "text": "OK",
                            "role": "primary_action",
                            "reason": "visible button",
                            "related_candidate_ids": ["c1"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            candidates, summary = apply_golden_decisions(
                [Candidate(id="c1", bbox=BBox(40, 40, 10, 10), source="visual")],
                [{"decision": "accept", "target_kind": "proposal", "target_id": "openai-vision-quarantined:p1"}],
                root,
                width=100,
                height=100,
            )

            self.assertEqual(len(candidates), 2)
            accepted = candidates[-1]
            self.assertEqual(accepted.source, "human-accepted-vision-proposal")
            self.assertEqual(accepted.confidence, 0.9)
            self.assertEqual(accepted.source_refs, ["openai-vision:p1"])
            self.assertEqual(accepted.bbox, BBox(0, 3, 16, 10))
            self.assertEqual(accepted.type_hint, "Button")
            self.assertEqual(summary.proposal_accepted, 1)

    def test_reject_and_ignore_do_not_change_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidates, summary = apply_golden_decisions(
                [Candidate(id="c1", bbox=BBox(0, 0, 10, 10), source="visual")],
                [
                    {"decision": "reject", "target_kind": "proposal", "target_id": "p1"},
                    {"decision": "ignore", "target_kind": "candidate", "target_id": "c1"},
                ],
                tmp,
            )

            self.assertEqual(len(candidates), 1)
            self.assertEqual(summary.rejected, 1)
            self.assertEqual(summary.ignored, 1)

    def test_edit_candidate_overrides_fields_and_relation_groups_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "relations.json").write_text(
                json.dumps({"merge_suggestions": [{"component_group_id": "g1", "type": "Button", "candidate_ids": ["c1", "c2"]}]}),
                encoding="utf-8",
            )
            candidates, summary = apply_golden_decisions(
                [
                    Candidate(id="c1", bbox=BBox(0, 0, 10, 10), source="visual"),
                    Candidate(id="c2", bbox=BBox(20, 0, 10, 10), source="visual"),
                ],
                [
                    {"decision": "edit", "target_kind": "candidate", "target_id": "c1", "type": "Text", "text": "Name"},
                    {"decision": "accept", "target_kind": "relation", "target_id": "g1"},
                ],
                root,
            )

            self.assertEqual(candidates[0].type_hint, "Text")
            self.assertEqual(candidates[0].text, "Name")
            self.assertEqual(candidates[0].metadata["openaiComponentGroupId"], "g1")
            self.assertEqual(candidates[1].metadata["openaiComponentGroupId"], "g1")
            self.assertEqual(summary.edited, 1)
            self.assertEqual(summary.relation_accepted, 1)

    def test_accept_relation_quarantined_proposal_creates_human_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "relation_quarantined.json").write_text(
                json.dumps(
                    [
                        {
                            "proposal_id": "rp1",
                            "bbox": {"x": 4, "y": 5, "w": 20, "h": 10},
                            "type": "Icon",
                            "reason": "missing close icon in graph review",
                            "related_candidate_ids": ["c1"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            candidates, summary = apply_golden_decisions(
                [Candidate(id="c1", bbox=BBox(0, 0, 10, 10), source="visual")],
                [{"decision": "accept", "target_kind": "proposal", "target_id": "openai-relation-quarantined:rp1"}],
                root,
                width=100,
                height=100,
            )

            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[-1].type_hint, "Icon")
            self.assertEqual(candidates[-1].source_refs, ["openai-vision:rp1"])
            self.assertEqual(summary.proposal_accepted, 1)

    def test_accept_relation_patch_from_relation_patches_groups_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "relation_patches.json").write_text(
                json.dumps(
                    {
                        "accepted_relation_patches": [
                            {
                                "patch_id": "rel1",
                                "relation_type": "text_on_image",
                                "from_id": "c2",
                                "to_id": "c1",
                                "reason": "text label belongs to button background",
                            }
                        ],
                        "accepted_component_group_patches": [
                            {
                                "component_group_id": "group1",
                                "type": "Button",
                                "candidate_ids": ["c1", "c2"],
                                "reason": "button group",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            candidates, summary = apply_golden_decisions(
                [
                    Candidate(id="c1", bbox=BBox(0, 0, 20, 10), source="visual"),
                    Candidate(id="c2", bbox=BBox(2, 2, 8, 4), source="visual"),
                ],
                [
                    {"decision": "accept", "target_kind": "relation", "target_id": "rel1"},
                    {"decision": "accept", "target_kind": "relation", "target_id": "group1"},
                ],
                root,
            )

            self.assertEqual(summary.relation_accepted, 2)
            self.assertEqual(candidates[0].metadata["openaiComponentGroupId"], "group1")
            self.assertEqual(candidates[1].metadata["openaiComponentGroupId"], "group1")
            self.assertEqual(len(candidates[0].metadata["goldenRelations"]), 2)

    def test_load_decisions_strips_sensitive_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden_decisions.json"
            path.write_text(
                json.dumps({"decisions": [{"decision": "accept", "target_kind": "proposal", "target_id": "p1", "token": "secret"}]}),
                encoding="utf-8",
            )

            decisions = load_golden_decisions(path)

            self.assertNotIn("token", decisions[0])


if __name__ == "__main__":
    unittest.main()
