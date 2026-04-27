import unittest

from uiir.models import BBox, Candidate
from uiir.openai_relations import apply_relation_patches, normalize_relation_patch_payload


class OpenAIRelationPatchTests(unittest.TestCase):
    def test_valid_relation_patch_is_accepted(self):
        c1 = Candidate(id="c1", bbox=BBox(0, 0, 100, 40), source="psd-layer", type_hint="Image")
        c2 = Candidate(id="c2", bbox=BBox(20, 8, 40, 20), source="psd-layer", type_hint="Text")

        result = apply_relation_patches(
            _graph(),
            [c1, c2],
            {
                "relation_patches": [
                    {
                        "patch_id": "rp1",
                        "action": "accept",
                        "edge_id": "e1",
                        "relation_type": "text_on_image",
                        "from_id": "c1",
                        "to_id": "c2",
                        "confidence": 0.93,
                        "reason": "label is centered on the visual button base",
                    }
                ],
                "component_group_patches": [],
                "missing_region_proposals": [],
                "render_diff_notes": [],
            },
        )

        self.assertEqual(result["summary"]["accepted_relation_patches"], 1)
        self.assertEqual(result["accepted_relation_patches"][0]["status"], "accepted_relation")
        self.assertEqual(result["accepted_relation_patches"][0]["edge_id"], "e1")
        self.assertEqual(c1.metadata["openaiRelationPatches"][0]["patch_id"], "rp1")
        self.assertEqual(c2.metadata["openaiRelationPatches"][0]["patch_id"], "rp1")

    def test_invalid_ids_are_rejected(self):
        c1 = Candidate(id="c1", bbox=BBox(0, 0, 100, 40), source="psd-layer", type_hint="Image")

        result = apply_relation_patches(
            _graph(),
            [c1],
            {
                "relation_patches": [
                    {
                        "patch_id": "rp_bad",
                        "action": "accept",
                        "relation_type": "contains",
                        "from_id": "c1",
                        "to_id": "missing",
                        "confidence": 0.7,
                        "reason": "bad model ref",
                    }
                ],
                "component_group_patches": [],
                "missing_region_proposals": [],
                "render_diff_notes": [],
            },
        )

        self.assertEqual(result["accepted_relation_patches"], [])
        self.assertEqual(result["rejected_relation_patches"][0]["rejectionReason"], "invalid_node_refs")
        self.assertEqual(result["rejected_relation_patches"][0]["invalid_ids"], ["missing"])
        self.assertNotIn("openaiRelationPatches", c1.metadata)

    def test_component_group_patch_records_without_bbox_mutation(self):
        c1 = Candidate(id="c1", bbox=BBox(0, 0, 100, 40), source="psd-layer", type_hint="Image")
        c2 = Candidate(id="c2", bbox=BBox(20, 8, 40, 20), source="psd-layer", type_hint="Text")
        original_bboxes = [c1.bbox, c2.bbox]

        result = apply_relation_patches(
            _graph(),
            [c1, c2],
            {
                "relation_patches": [],
                "component_group_patches": [
                    {
                        "component_group_id": "g_button",
                        "type": "Button",
                        "candidate_ids": ["c1", "c2"],
                        "confidence": 0.91,
                        "reason": "button base plus centered label",
                        "bbox": {"x": 999, "y": 999, "w": 1, "h": 1},
                    }
                ],
                "missing_region_proposals": [],
                "render_diff_notes": [],
            },
        )

        self.assertEqual(result["summary"]["accepted_component_group_patches"], 1)
        self.assertEqual([c1.bbox, c2.bbox], original_bboxes)
        self.assertEqual(c1.metadata["openaiComponentGroupId"], "g_button")
        self.assertEqual(c2.metadata["openaiComponentGroupPatches"][0]["type"], "Button")

    def test_missing_region_becomes_quarantined_proposal(self):
        c1 = Candidate(id="c1", bbox=BBox(0, 0, 100, 40), source="psd-layer", type_hint="Image")

        result = apply_relation_patches(
            _graph(),
            [c1],
            {
                "relation_patches": [],
                "component_group_patches": [],
                "missing_region_proposals": [
                    {
                        "proposal_id": "mr1",
                        "bbox": {"x": 108, "y": 10, "w": 20, "h": 20},
                        "type": "Icon",
                        "confidence": 0.8,
                        "reason": "visible icon not represented by a graph node",
                        "related_candidate_ids": ["c1"],
                    }
                ],
                "render_diff_notes": [],
            },
        )

        self.assertEqual(result["summary"]["quarantined_proposals"], 1)
        self.assertEqual(result["quarantined_proposals"][0]["status"], "quarantined")
        self.assertEqual(result["quarantined_proposals"][0]["quarantineReason"], "missing_region_requires_human_review")
        self.assertNotIn("openaiComponentGroupId", c1.metadata)

    def test_render_diff_notes_are_persisted(self):
        uiir = {
            "version": "0.1",
            "source": "sample.psd",
            "width": 120,
            "height": 80,
            "assetsRoot": "assets/",
            "root": {
                "id": "n1",
                "type": "Screen",
                "bbox": {"x": 0, "y": 0, "w": 120, "h": 80},
                "confidence": 1,
                "sourceRefs": ["document"],
                "children": [
                    {
                        "id": "c1",
                        "type": "Button",
                        "bbox": {"x": 0, "y": 0, "w": 100, "h": 40},
                        "confidence": 0.9,
                        "sourceRefs": ["layer:button"],
                        "metadata": {},
                        "children": [],
                    }
                ],
            },
            "metadata": {},
        }

        result = apply_relation_patches(
            _graph(),
            uiir,
            {
                "relation_patches": [],
                "component_group_patches": [],
                "missing_region_proposals": [],
                "render_diff_notes": [
                    {
                        "note_id": "rd1",
                        "severity": "major",
                        "message": "Rendered label is shifted down relative to the reference.",
                        "related_candidate_ids": ["c1"],
                    }
                ],
            },
        )

        self.assertEqual(result["summary"]["render_diff_notes"], 1)
        persisted = uiir["metadata"]["openaiRelationPatchResult"]["render_diff_notes"][0]
        self.assertEqual(persisted["note_id"], "rd1")
        self.assertEqual(uiir["root"]["children"][0]["metadata"]["openaiRenderDiffNotes"][0]["note_id"], "rd1")

    def test_normalizer_accepts_json_string_and_aliases(self):
        normalized = normalize_relation_patch_payload(
            '{"relation_patches":[{"id":"rp1","type":"contains","from":"c1","to":"c2"}]}'
        )

        self.assertEqual(normalized["relation_patches"][0]["patch_id"], "rp1")
        self.assertEqual(normalized["relation_patches"][0]["relation_type"], "contains")
        self.assertEqual(normalized["relation_patches"][0]["action"], "accept")


def _graph():
    return {
        "nodes": [
            {"id": "c1", "type": "Image", "bbox": {"x": 0, "y": 0, "w": 100, "h": 40}},
            {"id": "c2", "type": "Text", "bbox": {"x": 20, "y": 8, "w": 40, "h": 20}},
        ],
        "edges": [
            {
                "id": "e1",
                "type": "text_on_image",
                "from": "c1",
                "to": "c2",
                "confidence": 0.86,
                "reason": "text centered over visual element",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
