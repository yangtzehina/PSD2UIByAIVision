from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def curate_run(run_root: str | Path, golden_root: str | Path | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else root / "curation"
    out_dir.mkdir(parents=True, exist_ok=True)
    golden = Path(golden_root).expanduser().resolve() if golden_root else None
    samples = _collect_samples(root)
    queue = sorted((_score_sample(sample, golden) for sample in samples), key=lambda item: item["curation_value_score"], reverse=True)
    report = {
        "version": "0.1",
        "run_root": root.as_posix(),
        "golden": golden.as_posix() if golden else None,
        "count": len(queue),
        "queue": queue,
    }
    (out_dir / "curation_queue.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "curation_queue.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _collect_samples(root: Path) -> list[dict[str, Any]]:
    comparison_paths = sorted(root.rglob("comparison.json"), key=lambda path: path.as_posix())
    samples: list[dict[str, Any]] = []
    for comparison_path in comparison_paths:
        comparison = _read_json(comparison_path, default={})
        for item in comparison.get("items", []) or []:
            sample_dir = Path(item.get("openai_output") or item.get("output_dir") or "")
            if sample_dir and not sample_dir.is_absolute():
                sample_dir = (comparison_path.parent / sample_dir).resolve()
            samples.append(
                {
                    "name": item.get("name") or sample_dir.name,
                    "run_dir": comparison_path.parent.as_posix(),
                    "sample_dir": sample_dir.as_posix() if sample_dir else None,
                    "comparison": comparison_path.as_posix(),
                    "prompt_version": comparison.get("prompt_version"),
                    "policy": comparison.get("vision_policy"),
                    "document_kind": item.get("document_kind") or comparison.get("document_kind"),
                    "item": item,
                    "metrics": _metrics_for_sample(sample_dir),
                    "render_review": _read_json(sample_dir / "render_review.json", default={}) if sample_dir else {},
                    "graph": _read_json(sample_dir / "ui_graph.json", default={}) if sample_dir else {},
                }
            )
    if samples:
        return samples

    for uiir_path in sorted(root.rglob("uiir.json"), key=lambda path: path.as_posix()):
        sample_dir = uiir_path.parent
        samples.append(
            {
                "name": sample_dir.name,
                "run_dir": root.as_posix(),
                "sample_dir": sample_dir.as_posix(),
                "comparison": None,
                "prompt_version": None,
                "policy": None,
                "document_kind": _read_json(uiir_path, default={}).get("metadata", {}).get("documentKind"),
                "item": {},
                "metrics": _metrics_for_sample(sample_dir),
                "render_review": _read_json(sample_dir / "render_review.json", default={}),
                "graph": _read_json(sample_dir / "ui_graph.json", default={}),
            }
        )
    return samples


def _score_sample(sample: dict[str, Any], golden_root: Path | None) -> dict[str, Any]:
    item = sample.get("item", {}) or {}
    metrics = sample.get("metrics", {}) or {}
    golden = item.get("openai_golden") or metrics.get("golden") or {}
    render_review = sample.get("render_review", {}) or {}
    graph = sample.get("graph", {}) or {}
    vision = item.get("vision", {}) or {}
    quarantine = int(item.get("quarantined_proposals") or vision.get("quarantined_proposals") or _json_len(Path(sample.get("sample_dir") or "") / "vision_quarantined.json"))
    rejected = int(vision.get("rejected_proposals") or _json_len(Path(sample.get("sample_dir") or "") / "vision_rejected.json"))
    type_changes = len(item.get("type_changes", []) or [])
    unknown_delta = max(0, int(item.get("unknown_delta") or 0))
    render_issues = int(render_review.get("issue_count") or 0)
    relation_f1 = golden.get("relation_f1")
    type_f1 = golden.get("type_f1")
    golden_missing = golden_root is not None and not _has_golden(golden_root, str(sample.get("name") or ""))
    asset_sheet_misread = sample.get("document_kind") == "asset_sheet" and item.get("render_pixel_similarity_delta") is not None
    graph_edges = int(graph.get("stats", {}).get("edge_count") or 0)

    score = 0.0
    reasons: list[str] = []
    if quarantine:
        score += quarantine * 4
        reasons.append(f"quarantined_proposals={quarantine}")
    if rejected:
        score += rejected * 1.5
        reasons.append(f"rejected_proposals={rejected}")
    if type_changes:
        score += type_changes * 2
        reasons.append(f"type_changes={type_changes}")
    if unknown_delta:
        score += unknown_delta * 3
        reasons.append(f"unknown_delta={unknown_delta}")
    if render_issues:
        score += render_issues * 5
        reasons.append(f"render_review_issues={render_issues}")
    if isinstance(relation_f1, (int, float)) and relation_f1 < 0.75:
        score += (0.75 - relation_f1) * 10
        reasons.append(f"relation_f1={relation_f1}")
    if isinstance(type_f1, (int, float)) and type_f1 < 0.8:
        score += (0.8 - type_f1) * 10
        reasons.append(f"type_f1={type_f1}")
    if golden_missing:
        score += 3
        reasons.append("golden_missing")
    if asset_sheet_misread:
        score += 4
        reasons.append("asset_sheet_needs_non_replay_review")
    if graph_edges:
        score += min(5, graph_edges / 40)
        reasons.append(f"graph_edges={graph_edges}")

    return {
        "sample": sample.get("name"),
        "sample_dir": sample.get("sample_dir"),
        "run_dir": sample.get("run_dir"),
        "prompt_version": sample.get("prompt_version"),
        "policy": sample.get("policy"),
        "document_kind": sample.get("document_kind"),
        "curation_value_score": round(score, 5),
        "reasons": reasons or ["low_risk"],
        "recommended_files": _recommended_files(Path(sample.get("sample_dir") or "")),
        "metrics": {
            "quarantined_proposals": quarantine,
            "type_changes": type_changes,
            "unknown_delta": unknown_delta,
            "render_review_issue_count": render_issues,
            "relation_f1": relation_f1,
            "type_f1": type_f1,
        },
    }


def _metrics_for_sample(sample_dir: Path) -> dict[str, Any]:
    metrics_path = sample_dir.parent / "metrics.json"
    if not metrics_path.exists():
        return {}
    metrics = _read_json(metrics_path, default={})
    for item in metrics.get("items", []) or []:
        if item.get("output_dir") == sample_dir.as_posix():
            return item
    return {}


def _recommended_files(sample_dir: Path) -> list[str]:
    names = [
        "composite.png",
        "overlay.png",
        "graph_overlay.png",
        "render_diff.png",
        "ui_graph.json",
        "render_review.json",
        "vision_quarantined.json",
        "semantic_patches.json",
        "uiir.json",
        "uiir.xml",
    ]
    return [(sample_dir / name).as_posix() for name in names if (sample_dir / name).exists()]


def _has_golden(golden_root: Path, name: str) -> bool:
    return (golden_root / name / "uiir.json").exists() or (golden_root / f"{name}.json").exists()


def _json_len(path: Path) -> int:
    data = _read_json(path, default=[])
    return len(data) if isinstance(data, list) else 0


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# UIIR Curation Queue",
        "",
        f"- run_root: {report.get('run_root')}",
        f"- count: {report.get('count')}",
        "",
        "| sample | score | policy | document | reasons |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for item in report.get("queue", []) or []:
        lines.append(
            f"| {item.get('sample')} | {item.get('curation_value_score')} | {item.get('policy') or ''} | "
            f"{item.get('document_kind') or ''} | {', '.join(item.get('reasons', []))} |"
        )
    return "\n".join(lines) + "\n"
