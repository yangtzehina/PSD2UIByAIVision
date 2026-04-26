from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .corrections import apply_candidate_corrections, load_corrections
from .detect import detect_candidates, infer_uiir_document
from .document_kind import resolve_document_kind
from .openai_semantics import refine_candidates_with_openai
from .openai_vision import add_openai_vision_proposals
from .overlay import draw_overlay
from .provider import LLMProviderConfig
from .psd import extract_psd
from .xml_writer import write_json, write_xml


@dataclass
class ExtractOptions:
    include_visual: bool = True
    include_ocr: bool = False
    min_area: int = 96
    use_openai: bool = False
    model: str = "gpt-5.5"
    detail: str = "original"
    corrections: str | Path | None = None
    openai_audit: bool = False
    prompt_version: str = "semantic_v2"
    provider_name: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    api_mode: str = "responses"
    openai_vision_proposals: bool = False
    vision_adapter: str = "openai"
    vision_policy: str = "strict"
    document_kind: str = "auto"


@dataclass
class ExtractArtifacts:
    output_dir: Path
    composite: Path
    overlay: Path
    layers_json: Path
    candidates_json: Path
    uiir_json: Path
    uiir_xml: Path


def run_extract(psd_path: str | Path, output_dir: str | Path, options: ExtractOptions) -> ExtractArtifacts:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    extract = extract_psd(psd_path, out_dir)
    document_kind = resolve_document_kind(options.document_kind, extract.source, extract.layers)
    candidates = detect_candidates(
        extract,
        include_visual=options.include_visual,
        include_ocr=options.include_ocr,
        min_area=options.min_area,
    )
    overlay = draw_overlay(extract.composite_path, candidates, out_dir / "overlay.png")
    provider = LLMProviderConfig(
        provider_name=options.provider_name,
        api_key_env=options.api_key_env,
        base_url=options.base_url,
        api_mode=options.api_mode,
    )

    if options.openai_vision_proposals:
        candidates = add_openai_vision_proposals(
            candidates=candidates,
            layers=extract.layers,
            composite_path=extract.composite_path,
            overlay_path=overlay,
            width=extract.width,
            height=extract.height,
            min_area=options.min_area,
            model=options.model,
            detail=options.detail,
            audit_dir=out_dir,
            prompt_version="vision_v1",
            provider=provider,
            vision_adapter=options.vision_adapter,
            vision_policy=options.vision_policy,
            document_kind=document_kind,
        )
        overlay = draw_overlay(extract.composite_path, candidates, out_dir / "overlay.png")

    if options.use_openai:
        candidates = refine_candidates_with_openai(
            candidates=candidates,
            layers=extract.layers,
            overlay_path=overlay,
            model=options.model,
            detail=options.detail,
            audit_dir=out_dir if options.openai_audit else None,
            prompt_version=options.prompt_version,
            provider=provider,
        )
        overlay = draw_overlay(extract.composite_path, candidates, out_dir / "overlay.png")

    corrections = load_corrections(options.corrections)
    candidates, correction_summary = apply_candidate_corrections(candidates, corrections, extract.width, extract.height)
    if corrections:
        overlay = draw_overlay(extract.composite_path, candidates, out_dir / "overlay.png")

    document = infer_uiir_document(extract, candidates)
    document.metadata["corrections"] = correction_summary.to_dict()
    document.metadata["documentKind"] = document_kind
    document.metadata["requestedDocumentKind"] = options.document_kind
    document.metadata["visionPolicy"] = options.vision_policy
    candidates_json = out_dir / "candidates.json"
    candidates_json.write_text(
        json.dumps([candidate.to_dict() for candidate in candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    uiir_json = write_json(document, out_dir / "uiir.json")
    uiir_xml = write_xml(document, out_dir / "uiir.xml")
    return ExtractArtifacts(
        output_dir=out_dir,
        composite=extract.composite_path,
        overlay=overlay,
        layers_json=out_dir / "layer_metadata.json",
        candidates_json=candidates_json,
        uiir_json=uiir_json,
        uiir_xml=uiir_xml,
    )
