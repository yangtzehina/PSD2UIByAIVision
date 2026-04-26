from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from .heuristics import coerce_node_type
from .models import Candidate, LayerRecord
from .provider import LLMProviderConfig, create_openai_compatible_client, provider_summary
from .schema import OPENAI_SEMANTICS_SCHEMA


def refine_candidates_with_openai(
    candidates: list[Candidate],
    layers: list[LayerRecord],
    overlay_path: str | Path,
    model: str = "gpt-5.5",
    detail: str = "original",
    audit_dir: str | Path | None = None,
    prompt_version: str = "semantic_v2",
    provider: LLMProviderConfig | None = None,
) -> list[Candidate]:
    provider = (provider or LLMProviderConfig()).normalized()
    client = create_openai_compatible_client(provider)
    payload = {
        "task": "Classify PSD UI candidates and refine UI semantic hints.",
        "prompt_version": prompt_version,
        "rules": [
            "Do not invent pixel coordinates.",
            "Return one item per useful candidate id when possible.",
            "Prefer PSD layer text over OCR guesses.",
            "Do not return Screen for candidates; Screen is a synthetic root created by the program.",
            "Do not downgrade a concrete local type to Unknown unless the local type is already Unknown.",
            "Do not reclassify local Text candidates as Image, Container, or decorative controls.",
            "Do not switch high-confidence concrete candidates across type families.",
            "Only change Container into Button/Input/Toggle/Slider when component_group_id evidence is present.",
            "Use Unknown for ambiguous decorative elements.",
            "Use List/Grid/ScrollView only for repeated or scrollable regions.",
            "style may be empty unless a supplied text style is clearly useful.",
            "parent_candidate_id may be empty when uncertain.",
            "component_group_id may be empty; use it only when multiple candidates should be wrapped into one UI component.",
        ],
        "candidates": [_candidate_summary(candidate) for candidate in candidates[:220]],
        "layers": [_layer_summary(layer) for layer in layers[:260]],
    }
    data_url = _image_data_url(overlay_path)
    prompt = (
        "You are refining a PSD-to-UI intermediate representation. "
        "The image has candidate boxes overlaid with ids. "
        "Return semantic classifications that match the supplied JSON schema.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    started = time.time()
    response = _create_semantic_response(client, provider, model, prompt, data_url, detail)
    raw = _response_text(response)
    parsed = json.loads(raw)
    _write_audit(
        audit_dir=audit_dir,
        model=model,
        detail=detail,
        prompt_version=prompt_version,
        payload=payload,
        raw=raw,
        parsed=parsed,
        seconds=round(time.time() - started, 3),
        provider=provider,
    )
    by_id = {candidate.id: candidate for candidate in candidates}
    valid_parent_refs = set(by_id)
    valid_parent_refs.update(ref for candidate in candidates for ref in candidate.source_refs if ref.startswith("layer:"))
    semantic_patches = []
    for item in parsed.get("items", []):
        candidate = by_id.get(item.get("candidate_id"))
        if not candidate:
            continue
        patch = _new_semantic_patch(candidate, item)
        if item.get("component_group_id"):
            candidate.metadata["openaiProposedComponentGroupId"] = item["component_group_id"]
        _merge_semantic_type(candidate, item.get("type"), patch)
        candidate.confidence = max(candidate.confidence, float(item.get("confidence") or 0))
        _merge_optional_text_field(candidate, patch, "role", item.get("role"))
        _merge_text(candidate, patch, item.get("text"))
        _merge_optional_text_field(candidate, patch, "style", item.get("style"))
        _merge_optional_text_field(candidate, patch, "layout", item.get("layout"))
        _merge_parent_hint(candidate, patch, item.get("parent_candidate_id"), valid_parent_refs)
        if item.get("component_group_id"):
            candidate.metadata["openaiComponentGroupId"] = item["component_group_id"]
            patch["accepted"]["component_group_id"] = item["component_group_id"]
        candidate.metadata["openai"] = item
        candidate.metadata.setdefault("openaiSemanticPatches", []).append(patch)
        semantic_patches.append(patch)
    _write_semantic_patch_audit(audit_dir, semantic_patches)
    return candidates


def _new_semantic_patch(candidate: Candidate, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "proposed": {
            "type": item.get("type"),
            "role": item.get("role") or "",
            "text": item.get("text") or "",
            "style": item.get("style") or "",
            "layout": item.get("layout") or "",
            "parent_candidate_id": item.get("parent_candidate_id") or "",
            "component_group_id": item.get("component_group_id") or "",
        },
        "accepted": {},
        "rejected": [],
    }


def _merge_semantic_type(candidate: Candidate, value: Any, patch: dict[str, Any] | None = None) -> None:
    semantic_type = coerce_node_type(value)
    if semantic_type == "Screen":
        _reject_openai_field(candidate, "type", value, "screen_is_synthetic_root", patch)
        return
    if semantic_type == "Unknown" and candidate.type_hint != "Unknown":
        _reject_openai_field(candidate, "type", value, "unknown_downgrade_blocked", patch)
        return
    if candidate.type_hint == "Text" and semantic_type != "Text":
        _reject_openai_field(candidate, "type", value, "text_type_preserved", patch)
        return
    if _is_interactive_type(candidate.type_hint) and _is_interactive_type(semantic_type) and candidate.type_hint != semantic_type:
        _reject_openai_field(candidate, "type", value, "interactive_type_change_blocked", patch)
        return
    if candidate.type_hint == "Container" and _is_interactive_type(semantic_type) and not _has_component_evidence(candidate):
        _reject_openai_field(candidate, "type", value, "container_to_interactive_requires_group_evidence", patch)
        return
    if (
        candidate.confidence >= 0.68
        and candidate.type_hint not in {"Unknown", semantic_type}
        and semantic_type != "Unknown"
        and _type_family(candidate.type_hint) != _type_family(semantic_type)
    ):
        _reject_openai_field(candidate, "type", value, "cross_family_type_change_blocked", patch)
        return
    candidate.type_hint = semantic_type
    if patch is not None:
        patch["accepted"]["type"] = semantic_type


def _is_interactive_type(node_type: str) -> bool:
    return node_type in {"Button", "Input", "Toggle", "Slider"}


def _has_component_evidence(candidate: Candidate) -> bool:
    metadata = candidate.metadata or {}
    return bool(
        metadata.get("openaiComponentGroupId")
        or metadata.get("openaiProposedComponentGroupId")
        or metadata.get("openaiVisionRelations")
        or metadata.get("componentGroupId")
    )


def _type_family(node_type: str) -> str:
    if node_type == "Text":
        return "text"
    if node_type in {"Image", "Icon", "Background"}:
        return "visual"
    if node_type in {"Container", "List", "Grid", "ScrollView"}:
        return "structure"
    if _is_interactive_type(node_type):
        return "interactive"
    return node_type.lower()


def _merge_optional_text_field(candidate: Candidate, patch: dict[str, Any], field: str, value: Any) -> None:
    if not value:
        return
    setattr(candidate, field, str(value))
    patch["accepted"][field] = str(value)


def _merge_text(candidate: Candidate, patch: dict[str, Any], value: Any) -> None:
    if not value:
        return
    if candidate.text and candidate.text != value:
        _reject_openai_field(candidate, "text", value, "existing_text_preserved", patch)
        return
    candidate.text = str(value)
    patch["accepted"]["text"] = str(value)


def _merge_parent_hint(candidate: Candidate, patch: dict[str, Any], value: Any, valid_parent_refs: set[str]) -> None:
    if not value:
        return
    hint = str(value)
    if hint not in valid_parent_refs:
        _reject_openai_field(candidate, "parent_candidate_id", hint, "invalid_parent_hint", patch)
        return
    candidate.parent_hint = hint
    patch["accepted"]["parent_candidate_id"] = hint


def _reject_openai_field(candidate: Candidate, field: str, value: Any, reason: str, patch: dict[str, Any] | None = None) -> None:
    rejection = {
        "field": field,
        "value": value,
        "reason": reason,
    }
    candidate.metadata.setdefault("openaiRejected", []).append(
        rejection
    )
    if patch is not None:
        patch["rejected"].append(rejection)


def _write_semantic_patch_audit(audit_dir: str | Path | None, patches: list[dict[str, Any]]) -> None:
    if not audit_dir:
        return
    output = Path(audit_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "semantic_patches.json").write_text(json.dumps(patches, ensure_ascii=False, indent=2), encoding="utf-8")


def _create_semantic_response(client: Any, provider: LLMProviderConfig, model: str, prompt: str, data_url: str, detail: str) -> Any:
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
                    "name": "uiir_semantics",
                    "strict": True,
                    "schema": OPENAI_SEMANTICS_SCHEMA,
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
                "name": "uiir_semantics",
                "strict": True,
                "schema": OPENAI_SEMANTICS_SCHEMA,
            }
        },
    )


def _chat_image_detail(detail: str) -> str:
    return "high" if detail == "original" else detail


def _write_audit(
    audit_dir: str | Path | None,
    model: str,
    detail: str,
    prompt_version: str,
    payload: dict[str, Any],
    raw: str,
    parsed: dict[str, Any],
    seconds: float,
    provider: LLMProviderConfig,
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
        "candidate_count": len(payload.get("candidates", [])),
        "layer_count": len(payload.get("layers", [])),
        "returned_items": len(parsed.get("items", [])),
        "rules": payload.get("rules", []),
        "provider": provider_summary(provider),
    }
    (output / "openai_request_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "openai_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "openai_raw.json").write_text(raw, encoding="utf-8")


def _candidate_summary(candidate: Candidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "bbox": candidate.bbox.to_dict(),
        "source": candidate.source,
        "type_hint": candidate.type_hint,
        "confidence": round(candidate.confidence, 3),
        "name": candidate.name or "",
        "text": candidate.text or "",
        "style": candidate.style or "",
        "role": candidate.role or "",
        "asset": candidate.asset or "",
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
        "style": layer.style,
    }


def _image_data_url(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


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
