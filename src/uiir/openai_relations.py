from __future__ import annotations

import base64
import json
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

from .heuristics import coerce_node_type
from .models import BBox, Candidate, UIIRDocument, UINode
from .provider import LLMProviderConfig, create_openai_compatible_client, provider_summary


RELATION_PATCH_VERSION = "relation_v1"
RELATION_PATCH_SOURCE = "openai-relation-patch"
RELATION_ACTIONS = {"accept", "reject"}
RENDER_DIFF_SEVERITIES = {"info", "minor", "major", "critical"}


OPENAI_RELATION_PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relation_patches", "component_group_patches", "missing_region_proposals", "render_diff_notes"],
    "properties": {
        "relation_patches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["patch_id", "action", "relation_type", "from_id", "to_id", "confidence", "reason"],
                "properties": {
                    "patch_id": {"type": "string"},
                    "action": {"type": "string", "enum": sorted(RELATION_ACTIONS)},
                    "edge_id": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "from_id": {"type": "string"},
                    "to_id": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
            },
        },
        "component_group_patches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["component_group_id", "type", "candidate_ids", "confidence", "reason"],
                "properties": {
                    "component_group_id": {"type": "string"},
                    "type": {"type": "string"},
                    "candidate_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                },
            },
        },
        "missing_region_proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["proposal_id", "bbox", "type", "confidence", "reason", "related_candidate_ids"],
                "properties": {
                    "proposal_id": {"type": "string"},
                    "bbox": {"$ref": "#/$defs/bbox"},
                    "type": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                    "related_candidate_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "render_diff_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["note_id", "severity", "message", "related_candidate_ids"],
                "properties": {
                    "note_id": {"type": "string"},
                    "severity": {"type": "string", "enum": sorted(RENDER_DIFF_SEVERITIES)},
                    "message": {"type": "string"},
                    "related_candidate_ids": {"type": "array", "items": {"type": "string"}},
                    "bbox": {"$ref": "#/$defs/bbox"},
                },
            },
        },
    },
    "$defs": {
        "bbox": {
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


def review_relations_with_openai(
    extract_output: str | Path,
    *,
    model: str = "gpt-5.5",
    detail: str = "original",
    prompt_version: str = RELATION_PATCH_VERSION,
    provider: LLMProviderConfig | None = None,
) -> dict[str, Any]:
    """Ask an OpenAI-compatible vision model to review graph relations.

    This writes audit artifacts only. It may attach relation metadata to
    candidates.json, but it never edits bbox values, UIIR XML, or UIIR node
    geometry.
    """

    output_dir = Path(extract_output).expanduser().resolve()
    graph_path = output_dir / "ui_graph.json"
    graph_overlay_path = output_dir / "graph_overlay.png"
    candidates_path = output_dir / "candidates.json"
    render_review_path = output_dir / "render_review.json"
    if not graph_path.exists():
        raise FileNotFoundError(graph_path)
    if not graph_overlay_path.exists():
        raise FileNotFoundError(graph_overlay_path)
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)

    provider = (provider or LLMProviderConfig()).normalized()
    graph = _load_json(graph_path, default={})
    candidates = _load_json(candidates_path, default=[])
    render_review = _load_json(render_review_path, default={}) if render_review_path.exists() else {}
    payload = _relation_payload(graph, candidates, render_review, prompt_version)
    prompt = (
        "You are reviewing a PSD-to-UIIR Graph-of-Mark output. "
        "Confirm useful graph relations, suggest component groups, and note render differences. "
        "Do not write XML, do not edit coordinates, and do not delete supplied candidates.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    client = create_openai_compatible_client(provider)
    started = time.time()
    response = _create_relation_response(client, provider, model, prompt, _image_data_url(graph_overlay_path), detail)
    raw = _response_text(response)
    parsed = json.loads(raw)
    result = apply_relation_patches(graph, candidates, parsed)
    candidates_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_relation_audit(
        output_dir=output_dir,
        model=model,
        detail=detail,
        prompt_version=prompt_version,
        payload=payload,
        raw=raw,
        parsed=parsed,
        result=result,
        seconds=round(time.time() - started, 3),
        provider=provider,
    )
    return result


def normalize_relation_patch_payload(patch_payload: Any) -> dict[str, list[dict[str, Any]]]:
    """Normalize model output into the strict relation patch contract."""
    payload = _coerce_payload_object(patch_payload)
    return {
        "relation_patches": [
            _normalize_relation_patch(item, index)
            for index, item in enumerate(_as_dict_list(payload.get("relation_patches")), start=1)
        ],
        "component_group_patches": [
            _normalize_component_group_patch(item, index)
            for index, item in enumerate(_as_dict_list(payload.get("component_group_patches")), start=1)
        ],
        "missing_region_proposals": [
            _normalize_missing_region_proposal(item, index)
            for index, item in enumerate(_as_dict_list(payload.get("missing_region_proposals")), start=1)
        ],
        "render_diff_notes": [
            _normalize_render_diff_note(item, index)
            for index, item in enumerate(_as_dict_list(payload.get("render_diff_notes")), start=1)
        ],
    }


def apply_relation_patches(graph: dict[str, Any] | None, candidates_or_uiir: Any, patch_payload: Any) -> dict[str, Any]:
    """Apply relation evidence without changing geometry or writing XML artifacts."""
    normalized = normalize_relation_patch_payload(patch_payload)
    target = _TargetIndex.from_any(candidates_or_uiir)
    graph_index = _GraphIndex.from_graph(graph or {})
    known_ids = graph_index.ids | target.ids
    result = _empty_result()

    for patch in normalized["relation_patches"]:
        record, rejection_reason = _validated_relation_patch(patch, graph_index, known_ids)
        if rejection_reason:
            result["rejected_relation_patches"].append(_rejected(record, rejection_reason))
            continue
        record["status"] = "accepted_relation" if record["action"] == "accept" else "rejected_relation"
        result["accepted_relation_patches"].append(record)
        _record_for_ids(target, (record["from_id"], record["to_id"]), "openaiRelationPatches", record)

    for patch in normalized["component_group_patches"]:
        record, rejection_reason = _validated_component_group_patch(patch, known_ids)
        if rejection_reason:
            result["rejected_component_group_patches"].append(_rejected(record, rejection_reason))
            continue
        record["status"] = "accepted_component_group_patch"
        result["accepted_component_group_patches"].append(record)
        _record_component_group(target, record)

    for proposal in normalized["missing_region_proposals"]:
        record, rejection_reason = _validated_missing_region_proposal(proposal, known_ids)
        if rejection_reason:
            result["rejected_missing_region_proposals"].append(_rejected(record, rejection_reason))
            continue
        record["status"] = "quarantined"
        record["quarantineReason"] = "missing_region_requires_human_review"
        result["quarantined_proposals"].append(record)

    for note in normalized["render_diff_notes"]:
        record, rejection_reason = _validated_render_diff_note(note, known_ids)
        if rejection_reason:
            result["rejected_render_diff_notes"].append(_rejected(record, rejection_reason))
            continue
        record["status"] = "accepted_render_diff_note"
        result["render_diff_notes"].append(record)
        _record_for_ids(target, record["related_candidate_ids"], "openaiRenderDiffNotes", record)

    result["summary"] = _summary(result)
    _persist_result_metadata(target, result)
    return result


def _empty_result() -> dict[str, Any]:
    return {
        "version": RELATION_PATCH_VERSION,
        "accepted_relation_patches": [],
        "rejected_relation_patches": [],
        "accepted_component_group_patches": [],
        "rejected_component_group_patches": [],
        "quarantined_proposals": [],
        "rejected_missing_region_proposals": [],
        "render_diff_notes": [],
        "rejected_render_diff_notes": [],
        "summary": {},
    }


def _coerce_payload_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("relation patch payload must be a JSON object")
    return value


def _normalize_relation_patch(item: dict[str, Any], index: int) -> dict[str, Any]:
    action = str(item.get("action") or "accept").strip().lower()
    if action in {"accepted", "confirm", "confirmed"}:
        action = "accept"
    if action in {"rejected", "deny", "denied"}:
        action = "reject"
    return {
        "patch_id": _string(item.get("patch_id") or item.get("relation_patch_id") or item.get("id") or f"relation_patch_{index}"),
        "action": action,
        "edge_id": _string(item.get("edge_id") or item.get("edgeId")),
        "relation_type": _string(item.get("relation_type") or item.get("type")),
        "from_id": _string(item.get("from_id") or item.get("from") or item.get("source_id") or item.get("source")),
        "to_id": _string(item.get("to_id") or item.get("to") or item.get("target_id") or item.get("target")),
        "confidence": _confidence(item.get("confidence")),
        "reason": _string(item.get("reason")),
        "source": RELATION_PATCH_SOURCE,
    }


def _normalize_component_group_patch(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "component_group_id": _string(
            item.get("component_group_id") or item.get("componentGroupId") or item.get("group_id") or item.get("id") or f"relation_group_{index}"
        ),
        "type": coerce_node_type(item.get("type")),
        "candidate_ids": _unique_strings(item.get("candidate_ids") or item.get("candidateIds") or item.get("members")),
        "confidence": _confidence(item.get("confidence")),
        "reason": _string(item.get("reason")),
        "source": RELATION_PATCH_SOURCE,
    }


def _normalize_missing_region_proposal(item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "proposal_id": _string(item.get("proposal_id") or item.get("proposalId") or item.get("id") or f"missing_region_{index}"),
        "bbox": item.get("bbox"),
        "type": coerce_node_type(item.get("type")),
        "confidence": _confidence(item.get("confidence")),
        "reason": _string(item.get("reason")),
        "related_candidate_ids": _unique_strings(item.get("related_candidate_ids") or item.get("relatedCandidateIds")),
        "source": RELATION_PATCH_SOURCE,
    }


def _normalize_render_diff_note(item: dict[str, Any], index: int) -> dict[str, Any]:
    severity = str(item.get("severity") or "info").strip().lower()
    if severity not in RENDER_DIFF_SEVERITIES:
        severity = "info"
    normalized = {
        "note_id": _string(item.get("note_id") or item.get("noteId") or item.get("id") or f"render_diff_note_{index}"),
        "severity": severity,
        "message": _string(item.get("message") or item.get("note") or item.get("summary")),
        "related_candidate_ids": _unique_strings(item.get("related_candidate_ids") or item.get("relatedCandidateIds")),
        "source": RELATION_PATCH_SOURCE,
    }
    if item.get("bbox") is not None:
        normalized["bbox"] = item.get("bbox")
    return normalized


def _validated_relation_patch(
    patch: dict[str, Any], graph_index: "_GraphIndex", known_ids: set[str]
) -> tuple[dict[str, Any], str | None]:
    record = dict(patch)
    if record["action"] not in RELATION_ACTIONS:
        return record, "invalid_relation_action"

    edge = graph_index.edges.get(record["edge_id"]) if record["edge_id"] else None
    if record["edge_id"] and edge is None:
        return record, "unknown_edge_id"
    if edge:
        if record["from_id"] and record["from_id"] != edge.get("from"):
            return record, "edge_from_id_mismatch"
        if record["to_id"] and record["to_id"] != edge.get("to"):
            return record, "edge_to_id_mismatch"
        if record["relation_type"] and record["relation_type"] != edge.get("type"):
            return record, "edge_relation_type_mismatch"
        record["from_id"] = _string(record["from_id"] or edge.get("from"))
        record["to_id"] = _string(record["to_id"] or edge.get("to"))
        record["relation_type"] = _string(record["relation_type"] or edge.get("type"))

    if not record["from_id"] or not record["to_id"]:
        return record, "missing_relation_endpoint"
    if not record["relation_type"]:
        return record, "missing_relation_type"
    invalid = _invalid_ids((record["from_id"], record["to_id"]), known_ids)
    if invalid:
        record["invalid_ids"] = invalid
        return record, "invalid_node_refs"
    return record, None


def _validated_component_group_patch(patch: dict[str, Any], known_ids: set[str]) -> tuple[dict[str, Any], str | None]:
    record = dict(patch)
    if not record["component_group_id"]:
        return record, "missing_component_group_id"
    if record["type"] == "Screen":
        return record, "screen_component_group_blocked"
    if len(record["candidate_ids"]) < 2:
        return record, "component_group_requires_at_least_two_candidates"
    invalid = _invalid_ids(record["candidate_ids"], known_ids)
    if invalid:
        record["invalid_ids"] = invalid
        return record, "invalid_candidate_refs"
    return record, None


def _validated_missing_region_proposal(patch: dict[str, Any], known_ids: set[str]) -> tuple[dict[str, Any], str | None]:
    record = dict(patch)
    try:
        bbox = BBox.from_any(record["bbox"])
    except Exception:
        return record, "invalid_bbox"
    record["bbox"] = bbox.to_dict()
    if bbox.is_empty:
        return record, "empty_bbox"
    if record["type"] == "Screen":
        return record, "screen_missing_region_blocked"
    invalid = _invalid_ids(record["related_candidate_ids"], known_ids)
    if invalid:
        record["invalid_ids"] = invalid
        return record, "invalid_related_candidate_refs"
    return record, None


def _validated_render_diff_note(patch: dict[str, Any], known_ids: set[str]) -> tuple[dict[str, Any], str | None]:
    record = dict(patch)
    if not record["message"]:
        return record, "missing_render_diff_message"
    invalid = _invalid_ids(record["related_candidate_ids"], known_ids)
    if invalid:
        record["invalid_ids"] = invalid
        return record, "invalid_related_candidate_refs"
    if "bbox" in record:
        try:
            record["bbox"] = BBox.from_any(record["bbox"]).to_dict()
        except Exception:
            return record, "invalid_bbox"
    return record, None


def _record_component_group(target: "_TargetIndex", record: dict[str, Any]) -> None:
    for candidate_id in record["candidate_ids"]:
        for metadata in target.metadata_for_id(candidate_id):
            metadata["openaiComponentGroupId"] = record["component_group_id"]
            metadata.setdefault("openaiComponentGroupPatches", []).append(deepcopy(record))


def _record_for_ids(target: "_TargetIndex", ids: tuple[str, ...] | list[str], metadata_key: str, record: dict[str, Any]) -> None:
    for item_id in ids:
        for metadata in target.metadata_for_id(item_id):
            metadata.setdefault(metadata_key, []).append(deepcopy(record))


def _persist_result_metadata(target: "_TargetIndex", result: dict[str, Any]) -> None:
    if target.root_metadata is not None:
        target.root_metadata["openaiRelationPatchResult"] = deepcopy(result)


def _summary(result: dict[str, Any]) -> dict[str, int]:
    return {
        "accepted_relation_patches": len(result["accepted_relation_patches"]),
        "rejected_relation_patches": len(result["rejected_relation_patches"]),
        "accepted_component_group_patches": len(result["accepted_component_group_patches"]),
        "rejected_component_group_patches": len(result["rejected_component_group_patches"]),
        "quarantined_proposals": len(result["quarantined_proposals"]),
        "rejected_missing_region_proposals": len(result["rejected_missing_region_proposals"]),
        "render_diff_notes": len(result["render_diff_notes"]),
        "rejected_render_diff_notes": len(result["rejected_render_diff_notes"]),
    }


def _rejected(record: dict[str, Any], reason: str) -> dict[str, Any]:
    rejected = dict(record)
    rejected["status"] = "rejected"
    rejected["rejectionReason"] = reason
    return rejected


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _unique_strings(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, set):
        values = sorted(values, key=lambda item: str(item))
    elif not isinstance(values, (list, tuple)):
        values = [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _string(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return round(max(0.0, min(1.0, number)), 4)


def _invalid_ids(values: tuple[str, ...] | list[str], known_ids: set[str]) -> list[str]:
    return [value for value in values if value and value not in known_ids]


class _GraphIndex:
    def __init__(self, ids: set[str], edges: dict[str, dict[str, Any]]) -> None:
        self.ids = ids
        self.edges = edges

    @classmethod
    def from_graph(cls, graph: dict[str, Any]) -> "_GraphIndex":
        ids: set[str] = set()
        for node in graph.get("nodes", []) or []:
            if isinstance(node, dict) and node.get("id"):
                ids.add(str(node["id"]))
        edges: dict[str, dict[str, Any]] = {}
        for edge in graph.get("edges", []) or []:
            if not isinstance(edge, dict):
                continue
            source = _string(edge.get("from"))
            target = _string(edge.get("to"))
            if source:
                ids.add(source)
            if target:
                ids.add(target)
            edge_id = _string(edge.get("id"))
            if edge_id:
                edges[edge_id] = {
                    "id": edge_id,
                    "type": _string(edge.get("type")),
                    "from": source,
                    "to": target,
                }
        return cls(ids=ids, edges=edges)


class _TargetIndex:
    def __init__(self, ids: set[str], metadata_by_id: dict[str, list[dict[str, Any]]], root_metadata: dict[str, Any] | None) -> None:
        self.ids = ids
        self._metadata_by_id = metadata_by_id
        self.root_metadata = root_metadata

    @classmethod
    def from_any(cls, value: Any) -> "_TargetIndex":
        ids: set[str] = set()
        metadata_by_id: dict[str, list[dict[str, Any]]] = {}
        root_metadata = _root_metadata(value)

        def add(item_id: Any, metadata: dict[str, Any] | None) -> None:
            text = _string(item_id)
            if not text:
                return
            ids.add(text)
            if metadata is not None:
                metadata_by_id.setdefault(text, []).append(metadata)

        for candidate in _iter_candidates(value):
            if isinstance(candidate, Candidate):
                add(candidate.id, candidate.metadata)
            elif isinstance(candidate, dict):
                metadata = candidate.setdefault("metadata", {})
                add(candidate.get("id"), metadata if isinstance(metadata, dict) else None)

        for node in _iter_nodes(value):
            if isinstance(node, UINode):
                add(node.id, node.metadata)
                add(node.metadata.get("candidateId"), node.metadata)
            elif isinstance(node, dict):
                metadata = node.setdefault("metadata", {})
                metadata_dict = metadata if isinstance(metadata, dict) else None
                add(node.get("id"), metadata_dict)
                if metadata_dict is not None:
                    add(metadata_dict.get("candidateId"), metadata_dict)

        return cls(ids=ids, metadata_by_id=metadata_by_id, root_metadata=root_metadata)

    def metadata_for_id(self, item_id: str) -> list[dict[str, Any]]:
        return self._metadata_by_id.get(item_id, [])


def _root_metadata(value: Any) -> dict[str, Any] | None:
    if isinstance(value, UIIRDocument):
        return value.metadata
    if isinstance(value, dict) and "root" in value:
        metadata = value.setdefault("metadata", {})
        return metadata if isinstance(metadata, dict) else None
    return None


def _iter_candidates(value: Any) -> list[Any]:
    if isinstance(value, UIIRDocument):
        return list(value.candidates)
    if isinstance(value, dict):
        candidates = value.get("candidates")
        return list(candidates) if isinstance(candidates, list) else []
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _iter_nodes(value: Any) -> list[Any]:
    if isinstance(value, UIIRDocument):
        return _flatten_node_objects(value.root)
    if isinstance(value, dict):
        return _flatten_node_dicts(value.get("root"))
    return []


def _flatten_node_objects(node: UINode | None) -> list[UINode]:
    if node is None:
        return []
    nodes = [node]
    for child in node.children:
        nodes.extend(_flatten_node_objects(child))
    return nodes


def _flatten_node_dicts(node: Any) -> list[dict[str, Any]]:
    if not isinstance(node, dict):
        return []
    nodes = [node]
    for child in node.get("children", []) or []:
        nodes.extend(_flatten_node_dicts(child))
    return nodes


def _relation_payload(graph: dict[str, Any], candidates: list[dict[str, Any]], render_review: dict[str, Any], prompt_version: str) -> dict[str, Any]:
    return {
        "task": "Review PSD-aware UI graph relations and component group candidates.",
        "prompt_version": prompt_version,
        "rules": [
            "Use only supplied ids.",
            "Do not invent final coordinates or XML.",
            "Confirm or reject relation edges through relation_patches.",
            "Use component_group_patches only for existing candidate ids.",
            "Put missing visible regions into missing_region_proposals; the program will quarantine them.",
            "Use render_diff_notes for replay/composite discrepancies.",
        ],
        "graph": {
            "source": graph.get("source"),
            "width": graph.get("width"),
            "height": graph.get("height"),
            "stats": graph.get("stats", {}),
            "nodes": (graph.get("nodes") or [])[:260],
            "edges": (graph.get("edges") or [])[:520],
        },
        "candidates": [
            {
                "id": item.get("id"),
                "type_hint": item.get("type_hint") or item.get("type"),
                "bbox": item.get("bbox"),
                "text": item.get("text") or "",
                "name": item.get("name") or "",
                "role": item.get("role") or "",
                "source": item.get("source") or "",
            }
            for item in candidates[:260]
            if isinstance(item, dict)
        ],
        "render_review": {
            "status": render_review.get("status"),
            "issue_count": render_review.get("issue_count", 0),
            "issues": (render_review.get("issues") or [])[:80],
        },
    }


def _create_relation_response(client: Any, provider: LLMProviderConfig, model: str, prompt: str, data_url: str, detail: str) -> Any:
    if provider.api_mode == "chat-completions":
        return client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": _chat_image_detail(detail)}},
                    ],
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "uiir_relation_patches",
                    "strict": True,
                    "schema": OPENAI_RELATION_PATCH_SCHEMA,
                },
            },
        )
    return client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url, "detail": detail},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "uiir_relation_patches",
                "strict": True,
                "schema": OPENAI_RELATION_PATCH_SCHEMA,
            }
        },
    )


def _write_relation_audit(
    *,
    output_dir: Path,
    model: str,
    detail: str,
    prompt_version: str,
    payload: dict[str, Any],
    raw: str,
    parsed: dict[str, Any],
    result: dict[str, Any],
    seconds: float,
    provider: LLMProviderConfig,
) -> None:
    summary = {
        "model": model,
        "detail": detail,
        "prompt_version": prompt_version,
        "seconds": seconds,
        "graph_nodes": len(payload.get("graph", {}).get("nodes", [])),
        "graph_edges": len(payload.get("graph", {}).get("edges", [])),
        "candidate_count": len(payload.get("candidates", [])),
        "summary": result.get("summary", {}),
        "provider": provider_summary(provider),
    }
    (output_dir / "relation_request_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "relation_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "relation_raw.json").write_text(raw, encoding="utf-8")
    (output_dir / "relation_patches.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "relation_quarantined.json").write_text(
        json.dumps(result.get("quarantined_proposals", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _image_data_url(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _chat_image_detail(detail: str) -> str:
    return "high" if detail == "original" else detail


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    chunks: list[str] = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                chunks.append(str(value))
    if chunks:
        return "".join(chunks)
    choices = getattr(response, "choices", []) or []
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            text_chunks = []
            for item in content:
                if isinstance(item, dict):
                    value = item.get("text")
                else:
                    value = getattr(item, "text", None)
                if value:
                    text_chunks.append(str(value))
            if text_chunks:
                return "".join(text_chunks)
    raise RuntimeError("OpenAI response did not contain text output.")
