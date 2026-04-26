from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageStat

from .models import BBox
from .provider import LLMProviderConfig, missing_api_key_reason, provider_summary, resolve_api_key
from .render import render_uiir_preview


def review_render(
    extract_output: str | Path,
    output_dir: str | Path | None = None,
    *,
    use_openai: bool = False,
    provider: LLMProviderConfig | None = None,
    model: str = "gpt-5.5",
) -> dict[str, Any]:
    source_dir = Path(extract_output).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else source_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    uiir_path = source_dir / "uiir.json"
    composite_path = source_dir / "composite.png"
    replay_path = out_dir / "replay_preview.png"
    diagnostic_path = out_dir / "diagnostic_overlay.png"
    diff_path = out_dir / "render_diff.png"
    review_path = out_dir / "render_review.json"

    report: dict[str, Any] = {
        "version": "0.1",
        "source_output": source_dir.as_posix(),
        "output": out_dir.as_posix(),
        "model": model if use_openai else None,
        "openai": {"requested": use_openai, "status": "not_requested"},
        "issues": [],
        "quarantine": [],
    }

    if not uiir_path.exists():
        report["status"] = "error"
        report["error"] = "uiir.json missing"
        _write(review_path, report)
        return report

    uiir = _read_json(uiir_path, default={})
    report["document_kind"] = uiir.get("metadata", {}).get("documentKind") or "screen"

    try:
        render_uiir_preview(uiir_path, replay_path, mode="replay")
        render_uiir_preview(uiir_path, diagnostic_path, mode="diagnostic")
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"preview render failed: {exc}"
        _write(review_path, report)
        return report

    if not composite_path.exists():
        report["status"] = "ok"
        report["warnings"] = ["composite.png missing"]
        _write(review_path, report)
        return report

    composite = _open_rgb(composite_path)
    replay = _open_rgb(replay_path)
    if replay.size != composite.size:
        replay = replay.resize(composite.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(composite, replay)
    diff_overlay = _draw_diff_overlay(composite, diff)
    diff_overlay.save(diff_path)

    report["artifacts"] = {
        "replay_preview": replay_path.as_posix(),
        "diagnostic_overlay": diagnostic_path.as_posix(),
        "render_diff": diff_path.as_posix(),
    }
    report["pixel_similarity"] = round(_pixel_similarity_from_diff(diff), 5)
    report["issues"] = _diff_issues(diff, composite, replay, uiir)
    report["quarantine"] = [_issue_to_quarantine(issue, index + 1) for index, issue in enumerate(report["issues"]) if issue["type"] in {"missing", "extra", "misclassified"}]
    report["issue_counts"] = dict(Counter(issue["type"] for issue in report["issues"]))
    report["issue_count"] = len(report["issues"])
    _add_graph_issue_context(source_dir, out_dir, report)
    _add_openai_status(report, use_openai, provider)
    report["status"] = "ok"
    _write(review_path, report)
    return report


def _diff_issues(diff: Image.Image, composite: Image.Image, replay: Image.Image, uiir: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    bounds = _threshold_bbox(diff)
    if bounds:
        box = BBox.from_xyxy(*bounds)
        comp_ink = _ink(composite.crop(bounds))
        replay_ink = _ink(replay.crop(bounds))
        if comp_ink - replay_ink > 18:
            issue_type = "missing"
            reason = "composite has substantially more visible content than replay"
        elif replay_ink - comp_ink > 18:
            issue_type = "extra"
            reason = "replay has substantially more visible content than composite"
        else:
            issue_type = "misclassified"
            reason = "composite and replay differ in a region with content on both sides"
        issues.append(
            {
                "id": "rr1",
                "type": issue_type,
                "bbox": box.to_dict(),
                "severity": _severity(box.area, composite.width * composite.height),
                "reason": reason,
                "composite_ink": round(comp_ink, 3),
                "replay_ink": round(replay_ink, 3),
                "overlapping_nodes": _overlapping_nodes(uiir, box),
            }
        )

    invalid = _invalid_parent_hints(uiir)
    if invalid:
        issues.append(
            {
                "id": f"rr{len(issues) + 1}",
                "type": "bad_parent",
                "bbox": {"x": 0, "y": 0, "w": int(uiir.get("width") or 0), "h": int(uiir.get("height") or 0)},
                "severity": "medium",
                "reason": f"{invalid} invalid parent hints found in UIIR metadata",
                "count": invalid,
            }
        )
    return issues


def _add_graph_issue_context(source_dir: Path, out_dir: Path, report: dict[str, Any]) -> None:
    graph_path = source_dir / "ui_graph.json"
    if not graph_path.exists() and (out_dir / "ui_graph.json").exists():
        graph_path = out_dir / "ui_graph.json"
    if not graph_path.exists():
        return
    graph = _read_json(graph_path, default={})
    edge_counts = graph.get("stats", {}).get("edge_type_counts", {}) or {}
    report["graph"] = {
        "path": graph_path.as_posix(),
        "edge_type_counts": edge_counts,
        "relation_count": graph.get("stats", {}).get("edge_count", 0),
    }
    if int(edge_counts.get("repeated_pattern") or 0) > 0 and report.get("issue_count", 0) > 0:
        report["issues"].append(
            {
                "id": f"rr{len(report['issues']) + 1}",
                "type": "bad_repeat_group",
                "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
                "severity": "low",
                "reason": "render differs in a sample with repeated-pattern graph edges",
                "repeated_pattern_edges": edge_counts.get("repeated_pattern"),
            }
        )
        report["issue_counts"] = dict(Counter(issue["type"] for issue in report["issues"]))
        report["issue_count"] = len(report["issues"])


def _add_openai_status(report: dict[str, Any], use_openai: bool, provider: LLMProviderConfig | None) -> None:
    if not use_openai:
        return
    config = (provider or LLMProviderConfig()).normalized()
    info = provider_summary(config)
    if not resolve_api_key(config):
        report["openai"] = {"requested": True, "status": "skipped", "reason": missing_api_key_reason(config), "provider": info}
        return
    report["openai"] = {
        "requested": True,
        "status": "skipped",
        "reason": "render_review vision prompt is not enabled in this offline implementation",
        "provider": info,
    }


def _issue_to_quarantine(issue: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"render-review:{index}",
        "status": "quarantined",
        "source": "render-review",
        "issue_type": issue.get("type"),
        "bbox": issue.get("bbox"),
        "reason": issue.get("reason"),
        "related_node_ids": issue.get("overlapping_nodes", []),
    }


def _threshold_bbox(diff: Image.Image, threshold: int = 42) -> tuple[int, int, int, int] | None:
    gray = diff.convert("L")
    mask = gray.point(lambda value: 255 if value >= threshold else 0)
    return mask.getbbox()


def _draw_diff_overlay(composite: Image.Image, diff: Image.Image) -> Image.Image:
    base = composite.convert("RGBA")
    heat = diff.convert("L").point(lambda value: min(210, value * 2))
    red = Image.new("RGBA", base.size, (239, 68, 68, 0))
    red.putalpha(heat)
    base.alpha_composite(red)
    bounds = _threshold_bbox(diff)
    if bounds:
        draw = ImageDraw.Draw(base)
        draw.rectangle(bounds, outline=(239, 68, 68, 240), width=3)
    return base


def _pixel_similarity_from_diff(diff: Image.Image) -> float:
    histogram = diff.histogram()
    total = diff.size[0] * diff.size[1] * 3
    absolute = sum((index % 256) * count for index, count in enumerate(histogram))
    return max(0.0, min(1.0, 1.0 - absolute / (total * 255)))


def _ink(image: Image.Image) -> float:
    stat = ImageStat.Stat(image.convert("RGB"))
    mean = sum(stat.mean) / 3
    return 255 - mean


def _severity(area: int, canvas_area: int) -> str:
    ratio = area / max(1, canvas_area)
    if ratio >= 0.12:
        return "high"
    if ratio >= 0.03:
        return "medium"
    return "low"


def _overlapping_nodes(uiir: dict[str, Any], box: BBox) -> list[str]:
    hits = []
    for node in _flatten(uiir.get("root")):
        if node.get("type") == "Screen" or not node.get("bbox"):
            continue
        node_box = BBox.from_any(node["bbox"])
        if node_box.overlap_ratio(box) >= 0.2:
            hits.append(str(node.get("id")))
    return hits[:20]


def _invalid_parent_hints(uiir: dict[str, Any]) -> int:
    nodes = _flatten(uiir.get("root"))
    ids = {str(node.get("id")) for node in nodes if node.get("id")}
    invalid = 0
    for node in nodes:
        hint = (node.get("metadata") or {}).get("parentCandidateId") or (node.get("metadata") or {}).get("parent_hint")
        if hint and str(hint) not in ids:
            invalid += 1
    return invalid


def _flatten(root: Any) -> list[dict[str, Any]]:
    if not isinstance(root, dict):
        return []
    nodes = [root]
    for child in root.get("children", []) or []:
        nodes.extend(_flatten(child))
    return nodes


def _open_rgb(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    return background.convert("RGB")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
