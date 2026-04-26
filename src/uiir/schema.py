from __future__ import annotations

from .models import NODE_TYPES


OPENAI_SEMANTICS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "type", "confidence", "role", "text", "style", "layout", "parent_candidate_id", "component_group_id"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "type": {"type": "string", "enum": list(NODE_TYPES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "role": {"type": "string"},
                    "text": {"type": "string"},
                    "style": {"type": "string"},
                    "layout": {"type": "string"},
                    "parent_candidate_id": {"type": "string"},
                    "component_group_id": {"type": "string"},
                },
            },
        }
    },
}


OPENAI_VISION_PROPOSALS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items", "merge_suggestions", "split_suggestions"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["proposal_id", "bbox", "type", "confidence", "text", "role", "reason", "related_candidate_ids"],
                "properties": {
                    "proposal_id": {"type": "string"},
                    "bbox": {"$ref": "#/$defs/proposal_bbox"},
                    "type": {"type": "string", "enum": list(NODE_TYPES)},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "text": {"type": "string"},
                    "role": {"type": "string"},
                    "reason": {"type": "string"},
                    "related_candidate_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "merge_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["component_group_id", "type", "candidate_ids", "reason"],
                "properties": {
                    "component_group_id": {"type": "string"},
                    "type": {"type": "string", "enum": list(NODE_TYPES)},
                    "candidate_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
            },
        },
        "split_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["candidate_id", "reason", "items"],
                "properties": {
                    "candidate_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["proposal_id", "bbox", "type", "confidence", "text", "role", "reason"],
                            "properties": {
                                "proposal_id": {"type": "string"},
                                "bbox": {"$ref": "#/$defs/proposal_bbox"},
                                "type": {"type": "string", "enum": list(NODE_TYPES)},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                                "text": {"type": "string"},
                                "role": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    },
    "$defs": {
        "proposal_bbox": {
            "type": "object",
            "additionalProperties": False,
            "required": ["x", "y", "w", "h"],
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "w": {"type": "integer", "minimum": 0},
                "h": {"type": "integer", "minimum": 0},
            },
        }
    },
}


UIIR_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "UIIRDocument",
    "type": "object",
    "required": ["version", "source", "width", "height", "assetsRoot", "root"],
    "properties": {
        "version": {"type": "string"},
        "source": {"type": "string"},
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "assetsRoot": {"type": "string"},
        "root": {"$ref": "#/$defs/node"},
    },
    "$defs": {
        "bbox": {
            "type": "object",
            "required": ["x", "y", "w", "h"],
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
                "w": {"type": "integer", "minimum": 0},
                "h": {"type": "integer", "minimum": 0},
            },
        },
        "node": {
            "type": "object",
            "required": ["id", "type", "bbox", "confidence", "sourceRefs", "children"],
            "properties": {
                "id": {"type": "string"},
                "type": {"type": "string", "enum": list(NODE_TYPES)},
                "bbox": {"$ref": "#/$defs/bbox"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "sourceRefs": {"type": "array", "items": {"type": "string"}},
                "role": {"type": ["string", "null"]},
                "text": {"type": ["string", "null"]},
                "style": {"type": ["string", "null"]},
                "layout": {"type": ["string", "null"]},
                "asset": {"type": ["string", "null"]},
                "interaction": {"type": ["string", "null"]},
                "metadata": {"type": "object"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            },
        },
    },
}
