from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corrections import apply_candidate_corrections
from .detect import infer_uiir_document
from .heuristics import coerce_node_type
from .models import BBox, Candidate
from .psd import extract_psd
from .xml_writer import write_json, write_xml


DECISIONS = {"accept", "reject", "edit", "ignore"}
TARGET_KINDS = {"proposal", "candidate", "node", "relation"}
SENSITIVE_KEYS = {"token", "api_key", "apikey", "authorization", "base_url", "url", "request", "response", "raw"}


@dataclass
class GoldenDecisionSummary:
    loaded: int = 0
    accepted: int = 0
    edited: int = 0
    rejected: int = 0
    ignored: int = 0
    proposal_accepted: int = 0
    proposal_rejected: int = 0
    relation_accepted: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "accepted": self.accepted,
            "edited": self.edited,
            "rejected": self.rejected,
            "ignored": self.ignored,
            "proposal_accepted": self.proposal_accepted,
            "proposal_rejected": self.proposal_rejected,
            "relation_accepted": self.relation_accepted,
            "warnings": self.warnings,
        }


def load_golden_decisions(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("decisions"), list):
        items = data["decisions"]
    else:
        raise ValueError("golden_decisions.json must be an array or an object with a decisions array.")
    return [_normalize_decision(item) for item in items if isinstance(item, dict)]


def build_golden_from_decisions(psd_path: str | Path, run_dir: str | Path, decisions_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    run_root = Path(run_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extract = extract_psd(psd_path, out_dir)
    candidates = _load_candidates(run_root / "candidates.json")
    decisions = load_golden_decisions(decisions_path)
    candidates, summary = apply_golden_decisions(candidates, decisions, run_root, width=extract.width, height=extract.height)
    document = infer_uiir_document(extract, candidates)
    source_metadata = _load_optional_json(run_root / "uiir.json").get("metadata", {})
    document.metadata.update(
        {
            "golden": {
                "version": "0.1",
                "sourceRun": run_root.as_posix(),
                "decisions": summary.to_dict(),
            },
            "documentKind": source_metadata.get("documentKind"),
            "visionPolicy": source_metadata.get("visionPolicy"),
        }
    )

    candidates_json = out_dir / "candidates.json"
    candidates_json.write_text(json.dumps([candidate.to_dict() for candidate in candidates], ensure_ascii=False, indent=2), encoding="utf-8")
    uiir_json = write_json(document, out_dir / "uiir.json")
    uiir_xml = write_xml(document, out_dir / "uiir.xml")
    manifest = {
        "version": "0.1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": Path(psd_path).expanduser().resolve().name,
        "run_dir": run_root.as_posix(),
        "decisions_path": Path(decisions_path).expanduser().resolve().as_posix(),
        "uiir_json": uiir_json.as_posix(),
        "uiir_xml": uiir_xml.as_posix(),
        "candidates_json": candidates_json.as_posix(),
        "decision_summary": summary.to_dict(),
        "sensitive": {
            "api_key_recorded": False,
            "base_url_recorded": False,
            "raw_request_recorded": False,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def apply_golden_decisions(
    candidates: list[Candidate],
    decisions: list[dict[str, Any]],
    run_dir: str | Path,
    width: int | None = None,
    height: int | None = None,
) -> tuple[list[Candidate], GoldenDecisionSummary]:
    run_root = Path(run_dir).expanduser().resolve()
    summary = GoldenDecisionSummary(loaded=len(decisions))
    proposal_index = _proposal_index(run_root)
    relation_index = _relation_index(run_root)
    next_candidate = _next_candidate_index(candidates)
    corrections: list[dict[str, Any]] = []

    for decision in decisions:
        if _contains_sensitive_key(decision):
            summary.warnings.append(f"Decision {decision.get('target_id') or '<unknown>'} contains sensitive-looking fields; ignored those fields.")
        action = decision.get("decision")
        target_kind = decision.get("target_kind")
        if action not in DECISIONS or target_kind not in TARGET_KINDS:
            summary.warnings.append(f"Invalid decision skipped: {decision}")
            continue
        if action == "reject":
            summary.rejected += 1
            if target_kind == "proposal":
                summary.proposal_rejected += 1
            continue
        if action == "ignore":
            summary.ignored += 1
            if target_kind == "proposal":
                summary.proposal_rejected += 1
            continue
        if target_kind == "proposal":
            proposal_id = _target_id(decision)
            proposal = proposal_index.get(proposal_id)
            if proposal is None:
                summary.warnings.append(f"Proposal target not found: {proposal_id}")
                continue
            candidate = _candidate_from_proposal(proposal, decision, f"c{next_candidate}", width, height)
            next_candidate += 1
            candidates.append(candidate)
            summary.proposal_accepted += 1
            if action == "accept":
                summary.accepted += 1
            else:
                summary.edited += 1
            continue
        if target_kind in {"candidate", "node"}:
            correction = _decision_to_correction(decision, target_kind)
            if correction:
                corrections.append(correction)
                if action == "accept":
                    summary.accepted += 1
                else:
                    summary.edited += 1
            continue
        if target_kind == "relation":
            relation_id = _target_id(decision)
            relation = relation_index.get(relation_id)
            if relation is None and not decision.get("candidate_ids"):
                summary.warnings.append(f"Relation target not found: {relation_id}")
                continue
            _apply_relation_decision(candidates, relation or decision, decision)
            summary.relation_accepted += 1
            if action == "accept":
                summary.accepted += 1
            else:
                summary.edited += 1

    if corrections:
        candidates, correction_summary = apply_candidate_corrections(candidates, corrections, width, height)
        summary.warnings.extend(correction_summary.warnings)
    return candidates, summary


def _normalize_decision(item: dict[str, Any]) -> dict[str, Any]:
    normalized = _sanitize_decision(item)
    normalized["decision"] = str(normalized.get("decision") or "").strip().lower()
    normalized["target_kind"] = str(normalized.get("target_kind") or normalized.get("targetKind") or "").strip().lower()
    target_id = normalized.get("target_id") or normalized.get("targetId")
    if target_id is not None:
        normalized["target_id"] = _strip_target_prefix(str(target_id))
    return normalized


def _sanitize_decision(item: dict[str, Any]) -> dict[str, Any]:
    return {key: _sanitize_value(value) for key, value in item.items() if not _is_sensitive_key(key)}


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _sanitize_decision(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _candidate_from_proposal(proposal: dict[str, Any], decision: dict[str, Any], candidate_id: str, width: int | None, height: int | None) -> Candidate:
    proposal_id = str(proposal.get("proposal_id") or _target_id(decision))
    bbox = BBox.from_any(decision.get("bbox") or proposal.get("bbox") or {"x": 0, "y": 0, "w": 0, "h": 0})
    if width is not None and height is not None:
        bbox = bbox.clamp(width, height)
    component_group_id = decision.get("component_group_id") or decision.get("componentGroupId")
    metadata = {
        "goldenDecision": _decision_metadata(decision),
        "proposalReason": decision.get("reason") or proposal.get("reason") or "",
        "proposalStatus": "human_accepted",
        "relatedCandidateIds": proposal.get("related_candidate_ids", []) or [],
        "openaiVisionProposal": proposal,
    }
    if component_group_id:
        metadata["openaiComponentGroupId"] = str(component_group_id)
    return Candidate(
        id=candidate_id,
        bbox=bbox,
        source="human-accepted-vision-proposal",
        type_hint=coerce_node_type(decision.get("type") or proposal.get("type")),
        confidence=0.9,
        source_refs=[f"openai-vision:{proposal_id}"],
        name=f"Human accepted {proposal_id}",
        text=_optional_string(decision, "text", proposal.get("text")),
        role=_optional_string(decision, "role", proposal.get("role")),
        layout=_optional_string(decision, "layout", proposal.get("layout")),
        parent_hint=_optional_string(decision, "parent_id", decision.get("parentId")),
        metadata=metadata,
    )


def _decision_to_correction(decision: dict[str, Any], target_kind: str) -> dict[str, Any] | None:
    target_id = _target_id(decision)
    if not target_id:
        return None
    correction: dict[str, Any] = {"candidate_id" if target_kind == "candidate" else "node_id": target_id}
    for key in ("bbox", "type", "role", "text", "style", "layout", "asset", "parent_id", "parentId"):
        if key in decision:
            correction[key] = decision[key]
    return correction


def _apply_relation_decision(candidates: list[Candidate], relation: dict[str, Any], decision: dict[str, Any]) -> None:
    group_id = str(decision.get("component_group_id") or decision.get("componentGroupId") or relation.get("component_group_id") or _target_id(decision))
    candidate_ids = [str(value) for value in decision.get("candidate_ids") or relation.get("candidate_ids") or []]
    by_id = {candidate.id: candidate for candidate in candidates}
    for candidate_id in candidate_ids:
        candidate = by_id.get(candidate_id)
        if not candidate:
            continue
        candidate.metadata["openaiComponentGroupId"] = group_id
        candidate.metadata.setdefault("goldenRelations", []).append(
            {
                "component_group_id": group_id,
                "type": coerce_node_type(decision.get("type") or relation.get("type")),
                "reason": decision.get("reason") or relation.get("reason") or "",
                "decision": decision.get("decision"),
            }
        )


def _proposal_index(run_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for name in ("vision_quarantined.json", "vision_rejected.json", "vision_accepted.json"):
        path = run_root / name
        if not path.exists():
            continue
        data = _load_optional_json(path)
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and item.get("proposal_id"):
                index[_strip_target_prefix(str(item["proposal_id"]))] = item
    return index


def _relation_index(run_root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    relations = _load_optional_json(run_root / "relations.json")
    for item in relations.get("merge_suggestions", []) if isinstance(relations, dict) else []:
        if isinstance(item, dict) and item.get("component_group_id"):
            index[_strip_target_prefix(str(item["component_group_id"]))] = item
    quarantined = _load_optional_json(run_root / "vision_quarantined.json")
    for item in quarantined if isinstance(quarantined, list) else []:
        if isinstance(item, dict) and item.get("component_group_id"):
            index[_strip_target_prefix(str(item["component_group_id"]))] = item
    return index


def _load_candidates(path: Path) -> list[Candidate]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Candidate.from_dict(item) for item in data if isinstance(item, dict)]


def _load_optional_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _next_candidate_index(candidates: list[Candidate]) -> int:
    highest = 0
    for candidate in candidates:
        if candidate.id.startswith("c") and candidate.id[1:].isdigit():
            highest = max(highest, int(candidate.id[1:]))
    return highest + 1


def _target_id(decision: dict[str, Any]) -> str:
    return _strip_target_prefix(str(decision.get("target_id") or decision.get("targetId") or ""))


def _strip_target_prefix(value: str) -> str:
    value = value.strip()
    if ":" in value:
        return value.rsplit(":", 1)[-1]
    return value


def _optional_string(data: dict[str, Any], key: str, fallback: Any = None) -> str | None:
    value = data[key] if key in data else fallback
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _decision_metadata(decision: dict[str, Any]) -> dict[str, Any]:
    allowed = ("decision", "target_kind", "target_id", "reason", "component_group_id", "parent_id")
    return {key: decision[key] for key in allowed if key in decision}


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_is_sensitive_key(key) or _contains_sensitive_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return normalized in SENSITIVE_KEYS or normalized.endswith("_token") or normalized.endswith("_api_key")
