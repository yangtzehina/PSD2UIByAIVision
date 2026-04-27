from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .batch import _find_psd_files
from .curate import curate_run
from .evaluate import evaluate_outputs
from .fidelity import build_parser_fidelity_report
from .focus import build_focus_tiles
from .graph import build_ui_graph
from .openai_relations import review_relations_with_openai
from .pipeline import ExtractOptions, run_extract
from .provider import LLMProviderConfig, missing_api_key_reason, provider_summary, resolve_api_key
from .render_review import review_render


PREFERRED_OPENAI_SMOKE_FILES = ("interface.psd", "ui.psd")
PIXEL_BASELINE = 0.88785
PIXEL_MIN_RATIO = 0.95


@dataclass
class CompareOptions:
    model: str = "gpt-5.5"
    detail: str = "original"
    limit: int = 2
    prompt_version: str = "semantic_v2"
    include_visual: bool = True
    include_ocr: bool = False
    min_area: int = 96
    provider_name: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    api_mode: str = "responses"
    openai_vision_proposals: bool = False
    vision_adapter: str = "openai"
    vision_policy: str = "strict"
    document_kind: str = "auto"
    golden_root: str | Path | None = None
    graph_overlay: bool = False
    render_review: bool = False
    curation_report: bool = False
    focus_tiles: bool = False
    parser_fidelity: bool = False


@dataclass
class IterateOptions:
    model: str = "gpt-5.5"
    detail: str = "original"
    limit: int = 2
    prompt_version: str = "semantic_v2"
    include_visual: bool = True
    include_ocr: bool = False
    min_area: int = 96
    provider_name: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    api_mode: str = "responses"
    document_kind: str = "auto"
    golden_root: str | Path | None = None
    prompt_versions: tuple[str, ...] = ("semantic_v2",)
    policies: tuple[str, ...] = ("audit", "strict", "balanced")
    graph_overlay: bool = False
    render_review: bool = False
    curation_report: bool = False
    focus_tiles: bool = False
    parser_fidelity: bool = False


def run_compare_openai(input_dir: str | Path, output_dir: str | Path, options: CompareOptions) -> dict[str, Any]:
    if options.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    provider = LLMProviderConfig(
        provider_name=options.provider_name,
        api_key_env=options.api_key_env,
        base_url=options.base_url,
        api_mode=options.api_mode,
    ).normalized()
    provider_info = provider_summary(provider)

    if not resolve_api_key(provider):
        report = {
            "status": "skipped",
            "reason": missing_api_key_reason(provider),
            "model": options.model,
            "detail": options.detail,
            "prompt_version": options.prompt_version,
            "limit": options.limit,
            "openai_vision_proposals": options.openai_vision_proposals,
            "vision_adapter": options.vision_adapter,
            "vision_policy": options.vision_policy,
            "document_kind": options.document_kind,
            "golden": Path(options.golden_root).expanduser().resolve().as_posix() if options.golden_root else None,
            "graph_overlay": options.graph_overlay,
            "render_review": options.render_review,
            "curation_report": options.curation_report,
            "focus_tiles": options.focus_tiles,
            "parser_fidelity": options.parser_fidelity,
            "provider": provider_info,
            "created_at": _now(),
        }
        _write_report(out_dir, report)
        return report

    psd_paths = _select_psd_files(Path(input_dir).expanduser().resolve(), options.limit)
    baseline_dir = out_dir / "baseline"
    openai_dir = out_dir / "openai"
    baseline_items = []
    openai_items = []

    for index, psd_path in enumerate(psd_paths, start=1):
        slug = _slug(psd_path, index)
        baseline_items.append(_extract_one(psd_path, baseline_dir / slug, options, use_openai=False))
        openai_items.append(_extract_one(psd_path, openai_dir / slug, options, use_openai=True))

    baseline_metrics = evaluate_outputs(baseline_dir, golden_root=options.golden_root, report_path=baseline_dir / "metrics.json")
    openai_metrics = evaluate_outputs(openai_dir, golden_root=options.golden_root, report_path=openai_dir / "metrics.json")
    comparisons = [_compare_pair(base, refined) for base, refined in zip(baseline_items, openai_items)]
    report = {
        "status": "ok",
        "created_at": _now(),
        "model": options.model,
        "detail": options.detail,
        "prompt_version": options.prompt_version,
        "limit": options.limit,
        "openai_vision_proposals": options.openai_vision_proposals,
        "vision_adapter": options.vision_adapter,
        "vision_policy": options.vision_policy,
        "document_kind": options.document_kind,
        "golden": Path(options.golden_root).expanduser().resolve().as_posix() if options.golden_root else None,
        "graph_overlay": options.graph_overlay,
        "render_review": options.render_review,
        "curation_report": options.curation_report,
        "focus_tiles": options.focus_tiles,
        "parser_fidelity": options.parser_fidelity,
        "provider": provider_info,
        "input": Path(input_dir).expanduser().resolve().as_posix(),
        "output": out_dir.as_posix(),
        "baseline": _metrics_summary(baseline_metrics),
        "openai": _metrics_summary(openai_metrics),
        "gates": _gates(baseline_metrics, openai_metrics, comparisons),
        "items": comparisons,
    }
    _write_report(out_dir, report)
    _write_summary(out_dir, report)
    if options.curation_report:
        report["curation"] = curate_run(out_dir, golden_root=options.golden_root, output_dir=out_dir / "curation")
        _write_report(out_dir, report)
    return report


def run_iterate_openai(input_dir: str | Path, output_dir: str | Path, options: IterateOptions) -> dict[str, Any]:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    provider = LLMProviderConfig(
        provider_name=options.provider_name,
        api_key_env=options.api_key_env,
        base_url=options.base_url,
        api_mode=options.api_mode,
    ).normalized()
    provider_info = provider_summary(provider)
    if not resolve_api_key(provider):
        report = {
            "status": "skipped",
            "reason": missing_api_key_reason(provider),
            "model": options.model,
            "detail": options.detail,
            "prompt_version": options.prompt_version,
            "prompt_versions": list(options.prompt_versions),
            "policies": list(options.policies),
            "document_kind": options.document_kind,
            "golden": Path(options.golden_root).expanduser().resolve().as_posix() if options.golden_root else None,
            "graph_overlay": options.graph_overlay,
            "render_review": options.render_review,
            "curation_report": options.curation_report,
            "focus_tiles": options.focus_tiles,
            "parser_fidelity": options.parser_fidelity,
            "provider": provider_info,
            "created_at": _now(),
            "runs": [],
        }
        _write_leaderboard(out_dir, report)
        return report

    runs = []
    for policy in options.policies:
        if policy not in {"audit", "strict", "balanced"}:
            raise ValueError(f"Unsupported vision policy {policy!r}; expected audit, strict, or balanced")
    for prompt_version in options.prompt_versions:
        for policy in options.policies:
            run_dir = out_dir / f"{prompt_version}-{policy}"
            started = time.perf_counter()
            report = run_compare_openai(
                input_dir,
                run_dir,
                CompareOptions(
                    model=options.model,
                    detail=options.detail,
                    limit=options.limit,
                    prompt_version=prompt_version,
                    include_visual=options.include_visual,
                    include_ocr=options.include_ocr,
                    min_area=options.min_area,
                    provider_name=options.provider_name,
                    api_key_env=options.api_key_env,
                    base_url=options.base_url,
                    api_mode=options.api_mode,
                    openai_vision_proposals=True,
                    vision_adapter="openai",
                    vision_policy=policy,
                    document_kind=options.document_kind,
                    golden_root=options.golden_root,
                    graph_overlay=options.graph_overlay,
                    render_review=options.render_review,
                    curation_report=False,
                    focus_tiles=options.focus_tiles,
                    parser_fidelity=options.parser_fidelity,
                ),
            )
            seconds = round(time.perf_counter() - started, 3)
            _write_experiment_manifest(run_dir, input_dir, report, options, prompt_version, policy, seconds, provider)
            runs.append(_leaderboard_entry(policy, run_dir, report, prompt_version=prompt_version))

    runs = sorted(runs, key=_leaderboard_sort_key, reverse=True)
    report = {
        "status": "ok",
        "created_at": _now(),
        "model": options.model,
        "detail": options.detail,
        "prompt_version": options.prompt_version,
        "prompt_versions": list(options.prompt_versions),
        "policies": list(options.policies),
        "document_kind": options.document_kind,
        "golden": Path(options.golden_root).expanduser().resolve().as_posix() if options.golden_root else None,
        "graph_overlay": options.graph_overlay,
        "render_review": options.render_review,
        "curation_report": options.curation_report,
        "focus_tiles": options.focus_tiles,
        "parser_fidelity": options.parser_fidelity,
        "provider": provider_info,
        "runs": runs,
    }
    _write_leaderboard(out_dir, report)
    if options.curation_report:
        report["curation"] = curate_run(out_dir, golden_root=options.golden_root, output_dir=out_dir / "curation")
        _write_leaderboard(out_dir, report)
    return report


def review_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    comparison_path = root / "comparison.json"
    if not comparison_path.exists():
        raise FileNotFoundError(comparison_path)
    report = json.loads(comparison_path.read_text(encoding="utf-8"))
    findings = []
    if report.get("status") != "ok":
        findings.append({"severity": "info", "message": f"Run status is {report.get('status')}: {report.get('reason', '')}".strip()})
    for item in report.get("items", []):
        name = item.get("name")
        if item.get("unknown_delta", 0) > 0:
            findings.append({"severity": "warning", "sample": name, "message": f"Unknown nodes increased by {item['unknown_delta']}"})
        if item.get("invalid_parent_hints", 0) > 0:
            findings.append({"severity": "warning", "sample": name, "message": f"{item['invalid_parent_hints']} OpenAI parent hints are invalid"})
        if item.get("pixel_similarity_delta") is not None and item["pixel_similarity_delta"] < -0.04:
            findings.append({"severity": "warning", "sample": name, "message": f"Pixel similarity dropped by {item['pixel_similarity_delta']:.5f}"})
        if item.get("vision", {}).get("rejected_proposals", 0) > 0:
            findings.append({"severity": "info", "sample": name, "message": f"{item['vision']['rejected_proposals']} vision proposals rejected"})
        if item.get("semantic_patches", {}).get("rejected", 0) > 0:
            findings.append({"severity": "info", "sample": name, "message": f"{item['semantic_patches']['rejected']} semantic fields rejected"})
        if item.get("type_changes"):
            findings.append({"severity": "info", "sample": name, "message": f"{len(item['type_changes'])} node/candidate type changes"})
    review = {
        "run": root.as_posix(),
        "status": "ok",
        "finding_count": len(findings),
        "findings": findings,
    }
    (root / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "review.md").write_text(_review_markdown(review), encoding="utf-8")
    return review


def _extract_one(psd_path: Path, output_dir: Path, options: CompareOptions, use_openai: bool) -> dict[str, Any]:
    semantic_prompt_version = "semantic_v3" if options.prompt_version.startswith("relation") else options.prompt_version
    artifacts = run_extract(
        psd_path,
        output_dir,
        ExtractOptions(
            include_visual=options.include_visual,
            include_ocr=options.include_ocr,
            min_area=options.min_area,
            use_openai=use_openai,
            model=options.model,
            detail=options.detail,
            openai_audit=use_openai,
            prompt_version=semantic_prompt_version,
            provider_name=options.provider_name,
            api_key_env=options.api_key_env,
            base_url=options.base_url,
            api_mode=options.api_mode,
            openai_vision_proposals=use_openai and options.openai_vision_proposals,
            vision_adapter=options.vision_adapter,
            vision_policy=options.vision_policy,
            document_kind=options.document_kind,
        ),
    )
    graph_json = None
    graph_overlay = None
    render_review_json = None
    focus_tiles_json = None
    parser_fidelity_json = None
    relation_patches_json = None
    should_build_graph = options.graph_overlay or (use_openai and options.prompt_version.startswith("relation"))
    if should_build_graph:
        graph_result = build_ui_graph(output_dir)
        graph_json = graph_result.graph_json.as_posix()
        graph_overlay = graph_result.graph_overlay.as_posix()
    if options.render_review or options.focus_tiles:
        render_report = review_render(output_dir)
        render_review_json = (output_dir / "render_review.json").as_posix() if render_report else None
    if options.focus_tiles:
        focus_report = build_focus_tiles(output_dir)
        focus_tiles_json = (output_dir / "focus_tiles.json").as_posix() if focus_report else None
    if options.parser_fidelity:
        fidelity_report = build_parser_fidelity_report(output_dir)
        parser_fidelity_json = (output_dir / "parser_fidelity.json").as_posix() if fidelity_report else None
    if use_openai and options.prompt_version.startswith("relation"):
        provider = LLMProviderConfig(
            provider_name=options.provider_name,
            api_key_env=options.api_key_env,
            base_url=options.base_url,
            api_mode=options.api_mode,
        )
        review_relations_with_openai(
            output_dir,
            model=options.model,
            detail=options.detail,
            prompt_version=options.prompt_version,
            provider=provider,
        )
        relation_patches_json = (output_dir / "relation_patches.json").as_posix()
    return {
        "source": psd_path.as_posix(),
        "name": output_dir.name,
        "output_dir": output_dir.as_posix(),
        "uiir_json": artifacts.uiir_json.as_posix(),
        "candidates_json": artifacts.candidates_json.as_posix(),
        "ui_graph_json": graph_json,
        "graph_overlay": graph_overlay,
        "render_review_json": render_review_json,
        "focus_tiles_json": focus_tiles_json,
        "parser_fidelity_json": parser_fidelity_json,
        "relation_patches_json": relation_patches_json,
    }


def _select_psd_files(input_dir: Path, limit: int) -> list[Path]:
    psd_paths = _find_psd_files(input_dir)
    by_name = {path.name: path for path in psd_paths}
    selected = [by_name[name] for name in PREFERRED_OPENAI_SMOKE_FILES if name in by_name]
    for path in psd_paths:
        if path not in selected:
            selected.append(path)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _compare_pair(base: dict[str, Any], refined: dict[str, Any]) -> dict[str, Any]:
    base_uiir = _load_json(base["uiir_json"])
    refined_uiir = _load_json(refined["uiir_json"])
    base_candidates = _load_json(base["candidates_json"])
    refined_candidates = _load_json(refined["candidates_json"])
    base_nodes = _flatten_nodes(base_uiir.get("root"))
    refined_nodes = _flatten_nodes(refined_uiir.get("root"))
    base_types = Counter(node.get("type") for node in base_nodes)
    refined_types = Counter(node.get("type") for node in refined_nodes)
    type_changes = _candidate_type_changes(base_candidates, refined_candidates)
    semantic_patch_counts = _semantic_patch_counts(refined_candidates)
    vision_counts = _vision_counts(refined["output_dir"], refined_candidates)
    graph_counts = _graph_counts(refined["output_dir"])
    render_counts = _render_review_counts(refined["output_dir"])
    focus_counts = _focus_tile_counts(refined["output_dir"])
    fidelity_counts = _parser_fidelity_counts(refined["output_dir"])
    relation_patch_counts = _relation_patch_counts(refined["output_dir"])
    baseline_visual = _metric_item(base["output_dir"])
    openai_visual = _metric_item(refined["output_dir"])
    base_pixel = baseline_visual.get("visual", {}).get("pixel_similarity")
    openai_pixel = openai_visual.get("visual", {}).get("pixel_similarity")
    openai_render_pixel = openai_visual.get("visual", {}).get("render_pixel_similarity", openai_pixel)
    base_render_pixel = baseline_visual.get("visual", {}).get("render_pixel_similarity", base_pixel)
    return {
        "name": refined["name"],
        "source": refined["source"],
        "document_kind": refined_uiir.get("metadata", {}).get("documentKind", "screen"),
        "vision_policy": refined_uiir.get("metadata", {}).get("visionPolicy"),
        "baseline_output": base["output_dir"],
        "openai_output": refined["output_dir"],
        "baseline_node_count": len(base_nodes),
        "openai_node_count": len(refined_nodes),
        "diagnostic_node_delta": len(refined_nodes) - len(base_nodes),
        "baseline_type_counts": dict(base_types),
        "openai_type_counts": dict(refined_types),
        "unknown_delta": refined_types.get("Unknown", 0) - base_types.get("Unknown", 0),
        "role_fill_delta": _fill_rate(refined_nodes, "role") - _fill_rate(base_nodes, "role"),
        "layout_fill_delta": _fill_rate(refined_nodes, "layout") - _fill_rate(base_nodes, "layout"),
        "parent_hint_fill_delta": _candidate_fill_rate(refined_candidates, "parent_hint") - _candidate_fill_rate(base_candidates, "parent_hint"),
        "max_depth_delta": _max_depth(refined_uiir.get("root")) - _max_depth(base_uiir.get("root")),
        "baseline_pixel_similarity": base_pixel,
        "openai_pixel_similarity": openai_pixel,
        "pixel_similarity_delta": None if base_pixel is None or openai_pixel is None else round(openai_pixel - base_pixel, 5),
        "baseline_render_pixel_similarity": base_render_pixel,
        "openai_render_pixel_similarity": openai_render_pixel,
        "render_pixel_similarity": openai_render_pixel,
        "render_pixel_similarity_delta": None if base_render_pixel is None or openai_render_pixel is None else round(openai_render_pixel - base_render_pixel, 5),
        "invalid_parent_hints": _invalid_parent_hints(refined_candidates),
        "type_changes": type_changes,
        "semantic_patches": semantic_patch_counts,
        "type_guard_rejections": semantic_patch_counts.get("type_guard_rejections", 0),
        "vision": vision_counts,
        "quarantined_proposals": vision_counts.get("quarantined_proposals", 0),
        "graph": graph_counts,
        "relation_count": graph_counts.get("edge_count", 0),
        "relation_patches": relation_patch_counts,
        "accepted_relation_patches": relation_patch_counts.get("accepted_relation_patches", 0),
        "accepted_component_group_patches": relation_patch_counts.get("accepted_component_group_patches", 0),
        "render_review": render_counts,
        "render_review_issue_count": render_counts.get("issue_count", 0),
        "focus_tiles": focus_counts,
        "focus_tile_count": focus_counts.get("tile_count", 0),
        "parser_fidelity": fidelity_counts,
        "baseline_golden": baseline_visual.get("golden"),
        "openai_golden": openai_visual.get("golden"),
    }


def _metric_item(output_dir: str | Path) -> dict[str, Any]:
    metrics_path = Path(output_dir).parent / "metrics.json"
    if not metrics_path.exists():
        return {}
    metrics = _load_json(metrics_path)
    for item in metrics.get("items", []):
        if item.get("output_dir") == str(output_dir):
            return item
    return {}


def _candidate_type_changes(base_candidates: list[dict[str, Any]], refined_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refined_by_id = {candidate.get("id"): candidate for candidate in refined_candidates}
    changes = []
    for candidate in base_candidates:
        refined = refined_by_id.get(candidate.get("id"))
        if not refined:
            continue
        before = candidate.get("type_hint")
        after = refined.get("type_hint")
        if before != after:
            changes.append({"candidate_id": candidate.get("id"), "before": before, "after": after})
    return changes


def _semantic_patch_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    accepted = 0
    rejected = 0
    patch_count = 0
    type_guard_rejections = 0
    for candidate in candidates:
        for patch in candidate.get("metadata", {}).get("openaiSemanticPatches", []) or []:
            patch_count += 1
            accepted += len(patch.get("accepted", {}) or {})
            patch_rejections = patch.get("rejected", []) or []
            rejected += len(patch_rejections)
            type_guard_rejections += sum(1 for item in patch_rejections if item.get("field") == "type")
    return {"patches": patch_count, "accepted": accepted, "rejected": rejected, "type_guard_rejections": type_guard_rejections}


def _vision_counts(output_dir: str | Path, candidates: list[dict[str, Any]]) -> dict[str, int]:
    created = sum(1 for candidate in candidates if candidate.get("source") == "openai-vision-proposal")
    merged = sum(1 for candidate in candidates if candidate.get("metadata", {}).get("openaiVision", {}).get("action") == "merged")
    accepted_path = Path(output_dir) / "vision_accepted.json"
    quarantined_path = Path(output_dir) / "vision_quarantined.json"
    rejected_path = Path(output_dir) / "vision_rejected.json"
    relations_path = Path(output_dir) / "relations.json"
    accepted = len(_load_json(accepted_path)) if accepted_path.exists() else created + merged
    quarantined = len(_load_json(quarantined_path)) if quarantined_path.exists() else 0
    rejected = len(_load_json(rejected_path)) if rejected_path.exists() else 0
    relations = _load_json(relations_path) if relations_path.exists() else {}
    merge_suggestions = sum(1 for item in relations.get("merge_suggestions", []) or [] if item.get("accepted"))
    split_suggestions = len(relations.get("split_suggestions", []) or [])
    return {
        "created_candidates": created,
        "merged_proposals": merged,
        "accepted_proposals": accepted,
        "quarantined_proposals": quarantined,
        "rejected_proposals": rejected,
        "accepted_merge_suggestions": merge_suggestions,
        "split_suggestions": split_suggestions,
    }


def _graph_counts(output_dir: str | Path) -> dict[str, Any]:
    graph_path = Path(output_dir) / "ui_graph.json"
    if not graph_path.exists():
        return {"edge_count": 0, "edge_type_counts": {}}
    graph = _load_json(graph_path)
    stats = graph.get("stats", {}) or {}
    return {
        "path": graph_path.as_posix(),
        "node_count": stats.get("node_count", 0),
        "edge_count": stats.get("edge_count", 0),
        "edge_type_counts": stats.get("edge_type_counts", {}) or {},
    }


def _render_review_counts(output_dir: str | Path) -> dict[str, Any]:
    review_path = Path(output_dir) / "render_review.json"
    if not review_path.exists():
        return {"issue_count": 0, "issue_counts": {}}
    review = _load_json(review_path)
    return {
        "path": review_path.as_posix(),
        "status": review.get("status"),
        "issue_count": review.get("issue_count", len(review.get("issues", []) or [])),
        "issue_counts": review.get("issue_counts", {}) or {},
        "pixel_similarity": review.get("pixel_similarity"),
    }


def _focus_tile_counts(output_dir: str | Path) -> dict[str, Any]:
    focus_path = Path(output_dir) / "focus_tiles.json"
    if not focus_path.exists():
        return {"tile_count": 0}
    report = _load_json(focus_path)
    return {
        "path": focus_path.as_posix(),
        "status": report.get("status"),
        "tile_count": report.get("tile_count", len(report.get("tiles", []) or [])),
        "eligible_issue_count": report.get("eligible_issue_count", 0),
        "truncated": bool(report.get("truncated")),
    }


def _parser_fidelity_counts(output_dir: str | Path) -> dict[str, Any]:
    fidelity_path = Path(output_dir) / "parser_fidelity.json"
    if not fidelity_path.exists():
        return {}
    report = _load_json(fidelity_path)
    counts = report.get("counts", {}) or {}
    coverage = report.get("psd_tools_coverage", {}) or {}
    return {
        "path": fidelity_path.as_posix(),
        "layers": counts.get("layers", 0),
        "text_layers": counts.get("text_layers", 0),
        "smart_object_ish_layers": counts.get("smart_object_ish_layers", 0),
        "layer_effects": counts.get("layer_effects", 0),
        "warnings": counts.get("warnings", 0),
        "candidate_layer_coverage": coverage.get("candidate_layer_coverage"),
        "asset_coverage": coverage.get("asset_coverage"),
    }


def _relation_patch_counts(output_dir: str | Path) -> dict[str, Any]:
    relation_path = Path(output_dir) / "relation_patches.json"
    if not relation_path.exists():
        return {
            "accepted_relation_patches": 0,
            "accepted_component_group_patches": 0,
            "quarantined_proposals": 0,
            "render_diff_notes": 0,
        }
    report = _load_json(relation_path)
    summary = report.get("summary", {}) or {}
    return {
        "path": relation_path.as_posix(),
        "accepted_relation_patches": summary.get("accepted_relation_patches", 0),
        "rejected_relation_patches": summary.get("rejected_relation_patches", 0),
        "accepted_component_group_patches": summary.get("accepted_component_group_patches", 0),
        "rejected_component_group_patches": summary.get("rejected_component_group_patches", 0),
        "quarantined_proposals": summary.get("quarantined_proposals", 0),
        "render_diff_notes": summary.get("render_diff_notes", 0),
    }


def _invalid_parent_hints(candidates: list[dict[str, Any]]) -> int:
    ids = {candidate.get("id") for candidate in candidates}
    layer_refs = {ref for candidate in candidates for ref in candidate.get("source_refs", []) if isinstance(ref, str) and ref.startswith("layer:")}
    invalid = 0
    for candidate in candidates:
        hint = candidate.get("parent_hint")
        if hint and hint not in ids and hint not in layer_refs:
            invalid += 1
    return invalid


def _gates(baseline: dict[str, Any], openai: dict[str, Any], comparisons: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    baseline_pixel = baseline.get("avg_pixel_similarity")
    openai_pixel = openai.get("avg_pixel_similarity")
    pixel_floor = round((baseline_pixel or PIXEL_BASELINE) * PIXEL_MIN_RATIO, 5)
    items = comparisons or []
    return {
        "schema_ok_not_lower": openai.get("schema_ok", 0) >= baseline.get("schema_ok", 0),
        "failed_zero": True,
        "pixel_floor": pixel_floor,
        "pixel_not_significantly_lower": openai_pixel is None or openai_pixel >= pixel_floor,
        "unknown_not_increased": all((item.get("unknown_delta") or 0) <= 0 for item in items),
        "invalid_parent_hints_zero": all((item.get("invalid_parent_hints") or 0) == 0 for item in items),
        "semantic_fill_positive": any(
            (item.get("role_fill_delta") or 0) > 0
            or (item.get("layout_fill_delta") or 0) > 0
            or (item.get("parent_hint_fill_delta") or 0) > 0
            for item in items
        ),
        "golden_not_degraded": _golden_not_degraded(baseline, openai),
    }


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": metrics.get("count"),
        "schema_ok": metrics.get("schema_ok"),
        "avg_pixel_similarity": metrics.get("avg_pixel_similarity"),
        "avg_bbox_iou": metrics.get("avg_bbox_iou"),
        "avg_type_f1": metrics.get("avg_type_f1"),
        "avg_proposal_precision": metrics.get("avg_proposal_precision"),
        "avg_proposal_recall": metrics.get("avg_proposal_recall"),
        "avg_relation_f1": metrics.get("avg_relation_f1"),
        "avg_relation_precision": metrics.get("avg_relation_precision"),
        "avg_relation_recall": metrics.get("avg_relation_recall"),
        "avg_component_group_f1": metrics.get("avg_component_group_f1"),
        "avg_human_accept_rate": metrics.get("avg_human_accept_rate"),
        "avg_quarantine_usefulness": metrics.get("avg_quarantine_usefulness"),
        "report": metrics.get("report"),
    }


def _write_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _leaderboard_entry(policy: str, run_dir: Path, report: dict[str, Any], prompt_version: str | None = None) -> dict[str, Any]:
    items = report.get("items", []) or []
    gates = report.get("gates", {}) or {}
    type_changes = sum(len(item.get("type_changes", []) or []) for item in items)
    type_guard_rejections = sum(item.get("type_guard_rejections", 0) or 0 for item in items)
    quarantined = sum(item.get("quarantined_proposals", 0) or 0 for item in items)
    render_issues = sum(item.get("render_review_issue_count", 0) or 0 for item in items)
    relation_count = sum(item.get("relation_count", 0) or 0 for item in items)
    accepted_relation_patches = sum(item.get("accepted_relation_patches", 0) or 0 for item in items)
    accepted_component_group_patches = sum(item.get("accepted_component_group_patches", 0) or 0 for item in items)
    focus_tile_count = sum(item.get("focus_tile_count", 0) or 0 for item in items)
    semantic_gain = sum(
        max(0.0, item.get("role_fill_delta") or 0.0)
        + max(0.0, item.get("layout_fill_delta") or 0.0)
        + max(0.0, item.get("parent_hint_fill_delta") or 0.0)
        for item in items
    )
    openai_pixel = report.get("openai", {}).get("avg_pixel_similarity")
    baseline_pixel = report.get("baseline", {}).get("avg_pixel_similarity")
    score = 0
    score += 30 if gates.get("schema_ok_not_lower") else 0
    score += 20 if gates.get("pixel_not_significantly_lower") else 0
    score += 15 if gates.get("invalid_parent_hints_zero") else 0
    score += 15 if gates.get("unknown_not_increased") else 0
    score += 10 if gates.get("semantic_fill_positive") else 0
    score += 15 if gates.get("golden_not_degraded") else 0
    score += 10 * float(report.get("openai", {}).get("avg_type_f1") or 0.0)
    score += 10 * float(report.get("openai", {}).get("avg_proposal_recall") or 0.0)
    score += 5 * float(report.get("openai", {}).get("avg_relation_f1") or 0.0)
    score += 5 * float(report.get("openai", {}).get("avg_component_group_f1") or 0.0)
    score += min(10, quarantined * 0.5 + relation_count * 0.03 + accepted_relation_patches + accepted_component_group_patches)
    score -= type_changes
    score -= render_issues * 0.5
    curation_value_score = round(quarantined * 4 + render_issues * 5 + type_changes * 2 + relation_count * 0.02 + focus_tile_count, 5)
    return {
        "prompt_version": prompt_version or report.get("prompt_version"),
        "policy": policy,
        "status": report.get("status"),
        "run_dir": run_dir.as_posix(),
        "score": round(score + semantic_gain, 5),
        "schema_ok": report.get("openai", {}).get("schema_ok"),
        "baseline_pixel_similarity": baseline_pixel,
        "openai_pixel_similarity": openai_pixel,
        "pixel_not_significantly_lower": gates.get("pixel_not_significantly_lower"),
        "invalid_parent_hints_zero": gates.get("invalid_parent_hints_zero"),
        "unknown_not_increased": gates.get("unknown_not_increased"),
        "semantic_fill_positive": gates.get("semantic_fill_positive"),
        "golden_not_degraded": gates.get("golden_not_degraded"),
        "avg_type_f1": report.get("openai", {}).get("avg_type_f1"),
        "avg_bbox_iou": report.get("openai", {}).get("avg_bbox_iou"),
        "avg_proposal_precision": report.get("openai", {}).get("avg_proposal_precision"),
        "avg_proposal_recall": report.get("openai", {}).get("avg_proposal_recall"),
        "avg_relation_f1": report.get("openai", {}).get("avg_relation_f1"),
        "avg_relation_precision": report.get("openai", {}).get("avg_relation_precision"),
        "avg_relation_recall": report.get("openai", {}).get("avg_relation_recall"),
        "avg_component_group_f1": report.get("openai", {}).get("avg_component_group_f1"),
        "avg_quarantine_usefulness": report.get("openai", {}).get("avg_quarantine_usefulness"),
        "type_changes": type_changes,
        "type_guard_rejections": type_guard_rejections,
        "quarantined_proposals": quarantined,
        "render_review_issue_count": render_issues,
        "relation_count": relation_count,
        "accepted_relation_patches": accepted_relation_patches,
        "accepted_component_group_patches": accepted_component_group_patches,
        "focus_tile_count": focus_tile_count,
        "curation_value_score": curation_value_score,
    }


def _leaderboard_sort_key(entry: dict[str, Any]) -> tuple:
    return (
        bool(entry.get("pixel_not_significantly_lower")),
        bool(entry.get("invalid_parent_hints_zero")),
        bool(entry.get("unknown_not_increased")),
        bool(entry.get("golden_not_degraded")),
        float(entry.get("avg_type_f1") or 0.0),
        float(entry.get("avg_proposal_recall") or 0.0),
        float(entry.get("avg_relation_f1") or 0.0),
        float(entry.get("avg_component_group_f1") or 0.0),
        -int(entry.get("type_changes") or 0),
        -int(entry.get("render_review_issue_count") or 0),
        int(entry.get("accepted_relation_patches") or 0),
        int(entry.get("accepted_component_group_patches") or 0),
        float(entry.get("score") or 0),
    )


def _write_leaderboard(out_dir: Path, report: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "leaderboard.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# OpenAI Iteration Leaderboard",
        "",
        f"- status: {report.get('status')}",
        f"- provider: {report.get('provider', {}).get('provider_name')}",
        f"- model: {report.get('model')}",
        f"- prompt_version: {report.get('prompt_version')}",
        "",
        "| prompt | policy | score | pixel | type F1 | relation F1 | proposal recall | gate | type changes | quarantined | relation patches | focus tiles | render issues |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report.get("runs", []) or []:
        gate = (
            "pass"
            if run.get("pixel_not_significantly_lower")
            and run.get("invalid_parent_hints_zero")
            and run.get("unknown_not_increased")
            and run.get("golden_not_degraded")
            else "review"
        )
        lines.append(
            f"| {run.get('prompt_version')} | {run.get('policy')} | {run.get('score')} | {run.get('openai_pixel_similarity')} | "
            f"{run.get('avg_type_f1')} | {run.get('avg_relation_f1')} | {run.get('avg_proposal_recall')} | {gate} | "
            f"{run.get('type_changes')} | {run.get('quarantined_proposals')} | {run.get('accepted_relation_patches')} | "
            f"{run.get('focus_tile_count')} | {run.get('render_review_issue_count')} |"
        )
    (out_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _golden_not_degraded(baseline: dict[str, Any], openai: dict[str, Any]) -> bool:
    for field in ("avg_type_f1", "avg_bbox_iou", "avg_proposal_recall", "avg_relation_f1", "avg_component_group_f1"):
        base = baseline.get(field)
        refined = openai.get(field)
        if isinstance(base, (int, float)) and isinstance(refined, (int, float)) and refined + 1e-9 < base:
            return False
    return True


def _write_experiment_manifest(
    run_dir: Path,
    input_dir: str | Path,
    report: dict[str, Any],
    options: IterateOptions,
    prompt_version: str,
    policy: str,
    seconds: float,
    provider: LLMProviderConfig,
) -> None:
    manifest = {
        "version": "0.1",
        "created_at": _now(),
        "git_sha": _git_sha(),
        "input": Path(input_dir).expanduser().resolve().as_posix(),
        "samples": [item.get("source") for item in report.get("items", []) or []],
        "model": options.model,
        "detail": options.detail,
        "prompt_version": prompt_version,
        "vision_policy": policy,
        "document_kind": options.document_kind,
        "golden": Path(options.golden_root).expanduser().resolve().as_posix() if options.golden_root else None,
        "graph_overlay": options.graph_overlay,
        "render_review": options.render_review,
        "curation_report": options.curation_report,
        "focus_tiles": options.focus_tiles,
        "parser_fidelity": options.parser_fidelity,
        "api_key_env": provider.api_key_env,
        "api_key_present": bool(resolve_api_key(provider)),
        "base_url_present": bool(provider.base_url),
        "api_mode": provider.api_mode,
        "seconds": seconds,
        "estimated_cost": None,
        "comparison": (run_dir / "comparison.json").as_posix(),
    }
    (run_dir / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def _write_summary(out_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# OpenAI Smoke Regression Summary",
        "",
        f"- status: {report.get('status')}",
        f"- provider: {report.get('provider', {}).get('provider_name')}",
        f"- model: {report.get('model')}",
        f"- prompt_version: {report.get('prompt_version')}",
        f"- baseline avg pixel: {report.get('baseline', {}).get('avg_pixel_similarity')}",
        f"- openai avg pixel: {report.get('openai', {}).get('avg_pixel_similarity')}",
        f"- schema baseline/openai: {report.get('baseline', {}).get('schema_ok')}/{report.get('openai', {}).get('schema_ok')}",
        "",
        "## Items",
    ]
    for item in report.get("items", []):
        vision = item.get("vision", {})
        semantic = item.get("semantic_patches", {})
        lines.append(
            f"- {item['name']}: type_changes={len(item['type_changes'])}, "
            f"unknown_delta={item['unknown_delta']}, pixel_delta={item['pixel_similarity_delta']}, "
            f"document_kind={item.get('document_kind')}, "
            f"vision_created={vision.get('created_candidates', 0)}, "
            f"vision_quarantined={vision.get('quarantined_proposals', 0)}, "
            f"semantic_rejected={semantic.get('rejected', 0)}, "
            f"graph_edges={item.get('relation_count', 0)}, "
            f"relation_patches={item.get('accepted_relation_patches', 0)}, "
            f"focus_tiles={item.get('focus_tile_count', 0)}, "
            f"render_issues={item.get('render_review_issue_count', 0)}"
        )
    (out_dir / "regression_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _review_markdown(review: dict[str, Any]) -> str:
    lines = ["# UIIR Run Review", "", f"Findings: {review['finding_count']}", ""]
    for finding in review["findings"]:
        sample = f" [{finding['sample']}]" if finding.get("sample") else ""
        lines.append(f"- {finding['severity']}{sample}: {finding['message']}")
    return "\n".join(lines) + "\n"


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _flatten_nodes(root: Any) -> list[dict[str, Any]]:
    if not isinstance(root, dict):
        return []
    result = []

    def visit(node: dict[str, Any]) -> None:
        result.append(node)
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                visit(child)

    visit(root)
    return result


def _fill_rate(nodes: list[dict[str, Any]], field: str) -> float:
    relevant = [node for node in nodes if node.get("type") != "Screen"]
    if not relevant:
        return 0.0
    return round(sum(1 for node in relevant if node.get(field)) / len(relevant), 5)


def _candidate_fill_rate(candidates: list[dict[str, Any]], field: str) -> float:
    if not candidates:
        return 0.0
    return round(sum(1 for candidate in candidates if candidate.get(field)) / len(candidates), 5)


def _max_depth(root: Any) -> int:
    if not isinstance(root, dict):
        return 0
    children = [child for child in root.get("children", []) or [] if isinstance(child, dict)]
    if not children:
        return 1
    return 1 + max(_max_depth(child) for child in children)


def _slug(path: Path, index: int) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem).strip("._")
    return slug or f"item_{index}"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
