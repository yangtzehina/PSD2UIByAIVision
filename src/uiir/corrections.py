from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .heuristics import coerce_node_type
from .models import BBox, Candidate


@dataclass
class CorrectionSummary:
    loaded: int = 0
    applied: int = 0
    ignored: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "applied": self.applied,
            "ignored": self.ignored,
            "warnings": self.warnings,
        }


def load_corrections(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        items = data.get("corrections", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    raise ValueError("Corrections must be a JSON array or an object with a corrections array.")


def apply_candidate_corrections(
    candidates: list[Candidate],
    corrections: list[dict[str, Any]],
    width: int | None = None,
    height: int | None = None,
) -> tuple[list[Candidate], CorrectionSummary]:
    summary = CorrectionSummary(loaded=len(corrections))
    if not corrections:
        return candidates, summary

    by_candidate = {candidate.id: candidate for candidate in candidates}
    kept = list(candidates)
    for correction in corrections:
        candidate = _resolve_candidate(correction, kept, by_candidate)
        if candidate is None:
            target = correction.get("candidate_id") or correction.get("candidateId") or correction.get("node_id") or correction.get("nodeId")
            summary.warnings.append(f"Correction target not found: {target}")
            continue
        if correction.get("ignored") is True:
            kept = [item for item in kept if item is not candidate]
            by_candidate.pop(candidate.id, None)
            summary.applied += 1
            summary.ignored += 1
            continue
        _apply_candidate_fields(candidate, correction, width, height)
        candidate.metadata["correction"] = {key: value for key, value in correction.items() if key not in {"node", "candidate"}}
        summary.applied += 1
    return kept, summary


def _resolve_candidate(correction: dict[str, Any], candidates: list[Candidate], by_candidate: dict[str, Candidate]) -> Candidate | None:
    candidate_id = correction.get("candidate_id") or correction.get("candidateId")
    if candidate_id in by_candidate:
        return by_candidate[str(candidate_id)]
    node_id = correction.get("node_id") or correction.get("nodeId")
    index = _node_index(node_id)
    if index is not None and 0 <= index < len(candidates):
        return candidates[index]
    return None


def _node_index(node_id: Any) -> int | None:
    if not isinstance(node_id, str):
        return None
    match = re.fullmatch(r"n(\d+)", node_id.strip())
    if not match:
        return None
    return int(match.group(1)) - 2


def _apply_candidate_fields(candidate: Candidate, correction: dict[str, Any], width: int | None, height: int | None) -> None:
    bbox = correction.get("bbox")
    if bbox is not None:
        candidate.bbox = BBox.from_any(bbox)
        if width is not None and height is not None:
            candidate.bbox = candidate.bbox.clamp(width, height)
    node_type = correction.get("type")
    if node_type:
        candidate.type_hint = coerce_node_type(str(node_type))
        candidate.confidence = max(candidate.confidence, 0.95)
    for source_key, attr in (
        ("role", "role"),
        ("text", "text"),
        ("style", "style"),
        ("layout", "layout"),
        ("asset", "asset"),
    ):
        if source_key in correction:
            value = correction[source_key]
            setattr(candidate, attr, str(value) if value is not None else None)
    parent = correction.get("parent_id") or correction.get("parentId")
    if parent is not None:
        candidate.parent_hint = str(parent) if parent else None
