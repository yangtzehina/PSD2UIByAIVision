from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .heuristics import coerce_node_type
from .models import BBox, Candidate, LayerRecord
from .openai_semantics import _chat_image_detail, _response_text
from .provider import LLMProviderConfig, create_openai_compatible_client, provider_summary
from .schema import OPENAI_VISION_PROPOSALS_SCHEMA


VISION_CONFIDENCE_CAP = 0.55
VISION_TILE_SIZE = 1400
VISION_TILE_OVERLAP = 0.1
VISION_POLICIES = ("audit", "strict", "balanced")


@dataclass(frozen=True)
class TileSpec:
    index: int
    origin_x: int
    origin_y: int
    width: int
    height: int
    composite_data_url: str
    overlay_data_url: str | None = None


def add_openai_vision_proposals(
    candidates: list[Candidate],
    layers: list[LayerRecord],
    composite_path: str | Path,
    overlay_path: str | Path,
    width: int,
    height: int,
    min_area: int,
    model: str = "gpt-5.5",
    detail: str = "original",
    audit_dir: str | Path | None = None,
    prompt_version: str = "vision_v1",
    provider: LLMProviderConfig | None = None,
    vision_adapter: str = "openai",
    vision_policy: str = "strict",
    document_kind: str = "screen",
) -> list[Candidate]:
    if vision_adapter != "openai":
        raise RuntimeError(f"vision adapter {vision_adapter!r} is not bundled; use --vision-adapter openai")
    if vision_policy not in VISION_POLICIES:
        raise ValueError(f"Unsupported vision policy {vision_policy!r}; expected audit, strict, or balanced")

    provider = (provider or LLMProviderConfig()).normalized()
    client = create_openai_compatible_client(provider)
    tiles = _build_tiles(composite_path, overlay_path, width, height, audit_dir)
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    relations = {"merge_suggestions": [], "split_suggestions": []}
    raw_responses: list[dict[str, Any]] = []
    started = time.time()

    for tile in tiles:
        payload = _vision_payload(candidates, layers, width, height, tile, prompt_version)
        prompt = _vision_prompt(payload)
        response = _create_vision_response(client, provider, model, prompt, tile, detail)
        raw = _response_text(response)
        parsed = json.loads(raw)
        raw_responses.append({"tile": _tile_summary(tile), "raw": raw, "parsed": parsed})
        candidates, tile_accepted, tile_quarantined, tile_rejected, tile_relations = apply_vision_proposals(
            parsed,
            candidates,
            width=width,
            height=height,
            min_area=min_area,
            tile_origin=(tile.origin_x, tile.origin_y),
            vision_policy=vision_policy,
            document_kind=document_kind,
        )
        accepted.extend(tile_accepted)
        quarantined.extend(tile_quarantined)
        rejected.extend(tile_rejected)
        relations["merge_suggestions"].extend(tile_relations["merge_suggestions"])
        relations["split_suggestions"].extend(tile_relations["split_suggestions"])

    _write_vision_audit(
        audit_dir=audit_dir,
        model=model,
        detail=detail,
        prompt_version=prompt_version,
        provider=provider,
        seconds=round(time.time() - started, 3),
        tiles=tiles,
        raw_responses=raw_responses,
        accepted=accepted,
        quarantined=quarantined,
        rejected=rejected,
        relations=relations,
        vision_policy=vision_policy,
        document_kind=document_kind,
    )
    return candidates


def apply_vision_proposals(
    parsed: dict[str, Any],
    candidates: list[Candidate],
    width: int,
    height: int,
    min_area: int,
    tile_origin: tuple[int, int] = (0, 0),
    vision_policy: str = "balanced",
    document_kind: str = "screen",
) -> tuple[list[Candidate], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if vision_policy not in VISION_POLICIES:
        raise ValueError(f"Unsupported vision policy {vision_policy!r}; expected audit, strict, or balanced")
    next_index = len(candidates) + 1
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    relations = {"merge_suggestions": [], "split_suggestions": []}

    for index, item in enumerate(parsed.get("items", []) or [], start=1):
        candidates, next_index = _apply_single_proposal(
            item,
            candidates,
            next_index,
            accepted,
            quarantined,
            rejected,
            width,
            height,
            min_area,
            tile_origin,
            fallback_id=f"p{index}",
            merge_duplicates=True,
            vision_policy=vision_policy,
            document_kind=document_kind,
        )

    _apply_merge_suggestions(parsed.get("merge_suggestions", []) or [], candidates, relations, quarantined, vision_policy)
    for split in parsed.get("split_suggestions", []) or []:
        relation = {
            "candidate_id": split.get("candidate_id"),
            "reason": split.get("reason") or "",
            "accepted_items": [],
            "quarantined_items": [],
            "rejected_items": [],
        }
        for index, item in enumerate(split.get("items", []) or [], start=1):
            split_item = dict(item)
            split_item["related_candidate_ids"] = [str(split.get("candidate_id") or "")]
            split_item["reason"] = " / ".join(part for part in (split.get("reason"), item.get("reason")) if part)
            before_accept = len(accepted)
            before_quarantine = len(quarantined)
            before_reject = len(rejected)
            candidates, next_index = _apply_single_proposal(
                split_item,
                candidates,
                next_index,
                accepted,
                quarantined,
                rejected,
                width,
                height,
                min_area,
                tile_origin,
                fallback_id=f"split{index}",
                merge_duplicates=False,
                vision_policy=vision_policy,
                document_kind=document_kind,
            )
            if len(accepted) > before_accept:
                relation["accepted_items"].append(accepted[-1])
            if len(quarantined) > before_quarantine:
                relation["quarantined_items"].append(quarantined[-1])
            if len(rejected) > before_reject:
                relation["rejected_items"].append(rejected[-1])
        relations["split_suggestions"].append(relation)

    return candidates, accepted, quarantined, rejected, relations


def _apply_single_proposal(
    item: dict[str, Any],
    candidates: list[Candidate],
    next_index: int,
    accepted: list[dict[str, Any]],
    quarantined: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    width: int,
    height: int,
    min_area: int,
    tile_origin: tuple[int, int],
    fallback_id: str,
    merge_duplicates: bool,
    vision_policy: str,
    document_kind: str,
) -> tuple[list[Candidate], int]:
    proposal_id = str(item.get("proposal_id") or fallback_id)
    proposal_type = coerce_node_type(item.get("type"))
    bbox = _proposal_bbox(item.get("bbox"), tile_origin).clamp(width, height)
    proposal = _proposal_record(item, proposal_id, proposal_type, bbox)
    rejection_reason = _proposal_rejection_reason(bbox, proposal_type, width, height, min_area)
    if rejection_reason:
        rejected.append({**proposal, "status": "rejected", "rejectionReason": rejection_reason})
        return candidates, next_index
    if vision_policy == "audit":
        quarantined.append({**proposal, "status": "quarantined", "quarantineReason": "audit_policy"})
        return candidates, next_index

    duplicate = _find_matching_candidate(bbox, candidates) if merge_duplicates else None
    if duplicate:
        accepted_fields = _merge_duplicate(duplicate, proposal_type, item)
        record = {**proposal, "status": "accepted_candidate", "action": "merged", "candidate_id": duplicate.id, "acceptedFields": accepted_fields}
        duplicate.metadata.setdefault("openaiVisionProposals", []).append(record)
        duplicate.metadata["openaiVision"] = {"accepted": True, "action": "merged", "proposal_id": proposal_id}
        accepted.append(record)
        return candidates, next_index
    if not _policy_allows_new_candidate(vision_policy, document_kind):
        quarantined.append({**proposal, "status": "quarantined", "quarantineReason": _quarantine_reason(vision_policy, document_kind)})
        return candidates, next_index

    candidate = Candidate(
        id=f"c{next_index}",
        bbox=bbox,
        source="openai-vision-proposal",
        type_hint=proposal_type,
        confidence=min(VISION_CONFIDENCE_CAP, max(0.0, float(item.get("confidence") or 0.0))),
        source_refs=[f"openai-vision:{proposal_id}"],
        name=f"OpenAI vision {proposal_id}",
        text=str(item.get("text") or "") or None,
        role=str(item.get("role") or "") or None,
        metadata={
            "proposalReason": item.get("reason") or "",
            "proposalStatus": "accepted_candidate",
            "relatedCandidateIds": [str(value) for value in item.get("related_candidate_ids", []) or [] if value],
            "openaiVision": {"accepted": True, "action": "created", "proposal_id": proposal_id},
            "openaiVisionProposal": proposal,
        },
    )
    candidates.append(candidate)
    accepted.append({**proposal, "status": "accepted_candidate", "action": "created", "candidate_id": candidate.id})
    return candidates, next_index + 1


def _policy_allows_new_candidate(vision_policy: str, document_kind: str) -> bool:
    return vision_policy == "balanced" and document_kind == "screen"


def _quarantine_reason(vision_policy: str, document_kind: str) -> str:
    if document_kind == "asset_sheet":
        return "asset_sheet_proposal_not_runtime_node"
    if vision_policy == "strict":
        return "strict_requires_local_overlap"
    return f"{vision_policy}_policy"


def _proposal_bbox(value: Any, tile_origin: tuple[int, int]) -> BBox:
    box = BBox.from_any(value or {"x": 0, "y": 0, "w": 0, "h": 0})
    return BBox(box.x + tile_origin[0], box.y + tile_origin[1], box.w, box.h)


def _proposal_rejection_reason(bbox: BBox, proposal_type: str, width: int, height: int, min_area: int) -> str | None:
    if bbox.is_empty:
        return "empty_bbox"
    if bbox.area < min_area:
        return "area_below_minimum"
    if proposal_type != "Background" and bbox.area > width * height * 0.9:
        return "area_too_large_for_non_background"
    if proposal_type == "Screen":
        return "screen_is_synthetic_root"
    return None


def _find_matching_candidate(bbox: BBox, candidates: list[Candidate]) -> Candidate | None:
    best: tuple[float, Candidate] | None = None
    for candidate in candidates:
        score = max(bbox.iou(candidate.bbox), bbox.overlap_ratio(candidate.bbox))
        if score >= 0.72 and (best is None or score > best[0]):
            best = (score, candidate)
    return best[1] if best else None


def _merge_duplicate(candidate: Candidate, proposal_type: str, item: dict[str, Any]) -> dict[str, Any]:
    accepted: dict[str, Any] = {}
    if candidate.type_hint == "Unknown" and proposal_type not in {"Unknown", "Screen"}:
        candidate.type_hint = proposal_type
        accepted["type"] = proposal_type
    if not candidate.role and item.get("role"):
        candidate.role = str(item["role"])
        accepted["role"] = candidate.role
    if not candidate.text and item.get("text"):
        candidate.text = str(item["text"])
        accepted["text"] = candidate.text
    return accepted


def _apply_merge_suggestions(
    suggestions: list[dict[str, Any]],
    candidates: list[Candidate],
    relations: dict[str, list[dict[str, Any]]],
    quarantined: list[dict[str, Any]],
    vision_policy: str,
) -> None:
    by_id = {candidate.id: candidate for candidate in candidates}
    for index, suggestion in enumerate(suggestions, start=1):
        candidate_ids = [str(value) for value in suggestion.get("candidate_ids", []) or []]
        existing = [by_id[candidate_id] for candidate_id in candidate_ids if candidate_id in by_id]
        group_id = str(suggestion.get("component_group_id") or f"vision_group_{index}")
        record = {
            "component_group_id": group_id,
            "type": coerce_node_type(suggestion.get("type")),
            "candidate_ids": candidate_ids,
            "reason": suggestion.get("reason") or "",
            "accepted": len(existing) >= 2 and len(existing) == len(candidate_ids),
        }
        if vision_policy == "audit":
            record["accepted"] = False
            record["status"] = "quarantined"
            record["quarantineReason"] = "audit_policy"
            quarantined.append(record)
        elif record["accepted"]:
            record["status"] = "accepted_relation"
            for candidate in existing:
                candidate.metadata["openaiComponentGroupId"] = group_id
                candidate.metadata.setdefault("openaiVisionRelations", []).append(record)
        else:
            record["status"] = "rejected"
            record["rejectionReason"] = "missing_or_insufficient_candidate_refs"
        relations["merge_suggestions"].append(record)


def _proposal_record(item: dict[str, Any], proposal_id: str, proposal_type: str, bbox: BBox) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "bbox": bbox.to_dict(),
        "type": proposal_type,
        "confidence": min(VISION_CONFIDENCE_CAP, max(0.0, float(item.get("confidence") or 0.0))),
        "text": item.get("text") or "",
        "role": item.get("role") or "",
        "reason": item.get("reason") or "",
        "related_candidate_ids": [str(value) for value in item.get("related_candidate_ids", []) or [] if value],
    }


def _vision_payload(candidates: list[Candidate], layers: list[LayerRecord], width: int, height: int, tile: TileSpec, prompt_version: str) -> dict[str, Any]:
    return {
        "task": "Find missing UI elements and component grouping opportunities in a PSD UI image.",
        "prompt_version": prompt_version,
        "canvas": {"width": width, "height": height},
        "tile": _tile_summary(tile),
        "rules": [
            "Return bounding boxes relative to the supplied image crop, not global canvas coordinates.",
            "Suggest only visible UI elements missing from the candidate list or useful split/merge relations.",
            "Do not create Screen nodes. Screen is a synthetic root.",
            "Prefer merge_suggestions for button background + text and repeated controls.",
            "Use split_suggestions when one existing candidate visually contains multiple separate UI controls.",
        ],
        "candidates": [_candidate_summary(candidate) for candidate in candidates[:220]],
        "layers": [_layer_summary(layer) for layer in layers[:220]],
    }


def _vision_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are a vision parser for game UI PSD screenshots. "
        "Analyze the image crop and return JSON matching the supplied schema. "
        "The second image, when present, is an overlay with existing candidate ids.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _create_vision_response(client: Any, provider: LLMProviderConfig, model: str, prompt: str, tile: TileSpec, detail: str) -> Any:
    image_urls = [tile.composite_data_url]
    if tile.overlay_data_url:
        image_urls.append(tile.overlay_data_url)
    if provider.api_mode == "chat-completions":
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": image_url, "detail": _chat_image_detail(detail)}} for image_url in image_urls)
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "uiir_vision_proposals",
                    "strict": True,
                    "schema": OPENAI_VISION_PROPOSALS_SCHEMA,
                },
            },
        )
    content = [{"type": "input_text", "text": prompt}]
    content.extend({"type": "input_image", "image_url": image_url, "detail": detail} for image_url in image_urls)
    return client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "uiir_vision_proposals",
                "strict": True,
                "schema": OPENAI_VISION_PROPOSALS_SCHEMA,
            }
        },
    )


def _build_tiles(composite_path: str | Path, overlay_path: str | Path, width: int, height: int, audit_dir: str | Path | None) -> list[TileSpec]:
    composite = Image.open(composite_path).convert("RGBA")
    overlay = Image.open(overlay_path).convert("RGBA") if overlay_path and Path(overlay_path).exists() else None
    origins = _tile_origins(width, height)
    output_root = Path(audit_dir) / "vision_tiles" if audit_dir else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)
    tiles: list[TileSpec] = []
    for index, (x, y, w, h) in enumerate(origins, start=1):
        crop_box = (x, y, x + w, y + h)
        composite_crop = composite.crop(crop_box)
        overlay_crop = overlay.crop(crop_box) if overlay else None
        if output_root:
            composite_crop.save(output_root / f"tile_{index:02d}_composite.png")
            if overlay_crop:
                overlay_crop.save(output_root / f"tile_{index:02d}_overlay.png")
        tiles.append(
            TileSpec(
                index=index,
                origin_x=x,
                origin_y=y,
                width=w,
                height=h,
                composite_data_url=_image_to_data_url(composite_crop),
                overlay_data_url=_image_to_data_url(overlay_crop) if overlay_crop else None,
            )
        )
    return tiles


def _tile_origins(width: int, height: int, tile_size: int = VISION_TILE_SIZE, overlap: float = VISION_TILE_OVERLAP) -> list[tuple[int, int, int, int]]:
    if width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    xs = _axis_origins(width, tile_size, stride)
    ys = _axis_origins(height, tile_size, stride)
    return [(x, y, min(tile_size, width - x), min(tile_size, height - y)) for y in ys for x in xs]


def _axis_origins(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    origins = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_vision_audit(
    audit_dir: str | Path | None,
    model: str,
    detail: str,
    prompt_version: str,
    provider: LLMProviderConfig,
    seconds: float,
    tiles: list[TileSpec],
    raw_responses: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    quarantined: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    relations: dict[str, list[dict[str, Any]]],
    vision_policy: str,
    document_kind: str,
) -> None:
    if not audit_dir:
        return
    output = Path(audit_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model,
        "detail": detail,
        "prompt_version": prompt_version,
        "seconds": seconds,
        "tile_count": len(tiles),
        "accepted_count": len(accepted),
        "quarantined_count": len(quarantined),
        "rejected_count": len(rejected),
        "vision_policy": vision_policy,
        "document_kind": document_kind,
        "provider": provider_summary(provider),
    }
    (output / "vision_request_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "vision_proposals.json").write_text(json.dumps(raw_responses, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "vision_accepted.json").write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "vision_quarantined.json").write_text(json.dumps(quarantined, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "vision_rejected.json").write_text(json.dumps(rejected, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "relations.json").write_text(json.dumps(relations, ensure_ascii=False, indent=2), encoding="utf-8")


def _tile_summary(tile: TileSpec) -> dict[str, int]:
    return {"index": tile.index, "x": tile.origin_x, "y": tile.origin_y, "w": tile.width, "h": tile.height}


def _candidate_summary(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "bbox": candidate.bbox.to_dict(),
        "source": candidate.source,
        "type_hint": candidate.type_hint,
        "confidence": round(candidate.confidence, 3),
        "name": candidate.name or "",
        "text": candidate.text or "",
        "role": candidate.role or "",
        "source_refs": candidate.source_refs,
    }


def _layer_summary(layer: LayerRecord) -> dict[str, Any]:
    return {
        "id": layer.id,
        "name": layer.name,
        "path": layer.path,
        "kind": layer.kind,
        "bbox": layer.bbox.to_dict(),
        "visible": layer.visible,
        "is_group": layer.is_group,
        "text": layer.text or "",
    }
