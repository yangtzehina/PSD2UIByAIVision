from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops

from .models import NODE_TYPES, BBox
from .render import render_uiir_preview


def evaluate_outputs(output_root: str | Path, golden_root: str | Path | None = None, report_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(output_root).expanduser().resolve()
    golden = Path(golden_root).expanduser().resolve() if golden_root else None
    items = [_evaluate_one(path, golden) for path in _find_output_dirs(root)]
    report = {
        "root": root.as_posix(),
        "golden": golden.as_posix() if golden else None,
        "count": len(items),
        "schema_ok": sum(1 for item in items if item.get("schema", {}).get("ok")),
        "with_golden": sum(1 for item in items if item.get("golden")),
        "avg_pixel_similarity": _average([item.get("visual", {}).get("pixel_similarity") for item in items]),
        "avg_bbox_iou": _average([item.get("golden", {}).get("bbox_mean_iou") for item in items]),
        "avg_type_f1": _average([item.get("golden", {}).get("type_f1") for item in items]),
        "avg_proposal_precision": _average([item.get("golden", {}).get("proposal_precision") for item in items]),
        "avg_proposal_recall": _average([item.get("golden", {}).get("proposal_recall") for item in items]),
        "avg_relation_f1": _average([item.get("golden", {}).get("relation_f1") for item in items]),
        "avg_relation_precision": _average([item.get("golden", {}).get("relation_precision") for item in items]),
        "avg_relation_recall": _average([item.get("golden", {}).get("relation_recall") for item in items]),
        "avg_component_group_f1": _average([item.get("golden", {}).get("component_group_f1") for item in items]),
        "avg_human_accept_rate": _average([item.get("golden", {}).get("human_accept_rate") for item in items]),
        "avg_quarantine_usefulness": _average([item.get("golden", {}).get("quarantine_usefulness") for item in items]),
        "items": items,
    }
    output = Path(report_path).expanduser().resolve() if report_path else root / "metrics.json"
    report["report"] = output.as_posix()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _evaluate_one(output_dir: Path, golden_root: Path | None) -> dict[str, Any]:
    uiir_path = output_dir / "uiir.json"
    item: dict[str, Any] = {"name": output_dir.name, "output_dir": output_dir.as_posix(), "uiir": uiir_path.as_posix()}
    if not uiir_path.exists():
        item["schema"] = {"ok": False, "errors": ["uiir.json missing"]}
        return item
    try:
        data = json.loads(uiir_path.read_text(encoding="utf-8"))
    except Exception as exc:
        item["schema"] = {"ok": False, "errors": [f"uiir.json parse failed: {exc}"]}
        return item

    nodes = _flatten_nodes(data.get("root"))
    item["schema"] = _validate_uiir(data)
    item["document_kind"] = data.get("metadata", {}).get("documentKind") or "screen"
    item["node_count"] = len(nodes)
    item["type_counts"] = dict(Counter(str(node.get("type") or "Unknown") for node in nodes))
    item["visual"] = _visual_metrics(output_dir, uiir_path, item["document_kind"])
    if golden_root:
        golden_data = _load_golden(output_dir.name, golden_root)
        if golden_data:
            item["golden"] = _golden_metrics(nodes, _flatten_nodes(golden_data.get("root")), golden_data)
    return item


def _find_output_dirs(root: Path) -> list[Path]:
    if (root / "uiir.json").exists():
        return [root]
    report = root / "report.json"
    if report.exists():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
            dirs = [Path(item["output_dir"]) for item in data.get("items", []) if item.get("ok") and item.get("output_dir")]
            return [path for path in dirs if (path / "uiir.json").exists()]
        except Exception:
            pass
    return sorted({path.parent for path in root.rglob("uiir.json")}, key=lambda path: path.as_posix().lower())


def _validate_uiir(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    for key in ("version", "source", "width", "height", "assetsRoot", "root"):
        if key not in data:
            errors.append(f"Missing {key}")
    root = data.get("root")
    if isinstance(root, dict):
        _validate_node(root, errors, "root")
    else:
        errors.append("root must be an object")
    return {"ok": not errors, "errors": errors}


def _validate_node(node: dict[str, Any], errors: list[str], path: str) -> None:
    for key in ("id", "type", "bbox", "confidence", "sourceRefs", "children"):
        if key not in node:
            errors.append(f"{path}: missing {key}")
    if node.get("type") not in NODE_TYPES:
        errors.append(f"{path}: invalid type {node.get('type')!r}")
    bbox = node.get("bbox")
    if not isinstance(bbox, dict) or any(key not in bbox for key in ("x", "y", "w", "h")):
        errors.append(f"{path}: invalid bbox")
    children = node.get("children")
    if not isinstance(children, list):
        errors.append(f"{path}: children must be an array")
        return
    for index, child in enumerate(children):
        if isinstance(child, dict):
            _validate_node(child, errors, f"{path}.children[{index}]")
        else:
            errors.append(f"{path}.children[{index}]: child must be an object")


def _visual_metrics(output_dir: Path, uiir_path: Path, document_kind: str = "screen") -> dict[str, Any]:
    composite = output_dir / "composite.png"
    preview = output_dir / "preview.png"
    replay_preview = output_dir / "replay_preview.png"
    diagnostic_overlay = output_dir / "diagnostic_overlay.png"
    result: dict[str, Any] = {}
    try:
        render_uiir_preview(uiir_path, replay_preview, mode="replay")
        render_uiir_preview(uiir_path, preview, mode="replay")
        render_uiir_preview(uiir_path, diagnostic_overlay, mode="diagnostic")
        result["preview"] = preview.as_posix()
        result["replay_preview"] = replay_preview.as_posix()
        result["diagnostic_overlay"] = diagnostic_overlay.as_posix()
    except Exception as exc:
        result["preview_error"] = str(exc)
        return result
    if not composite.exists():
        return result
    try:
        similarity = round(_pixel_similarity(composite, replay_preview), 5)
        if document_kind == "asset_sheet":
            result["pixel_similarity_applicable"] = False
            result["asset_sheet_render_pixel_similarity"] = similarity
        else:
            result["pixel_similarity_applicable"] = True
            result["render_pixel_similarity"] = similarity
            result["pixel_similarity"] = similarity
    except Exception as exc:
        result["pixel_error"] = str(exc)
    return result


def _pixel_similarity(left_path: Path, right_path: Path) -> float:
    left = _open_rgb_over_white(left_path)
    right = _open_rgb_over_white(right_path)
    if right.size != left.size:
        right = right.resize(left.size, Image.Resampling.LANCZOS)
    diff = ImageChops.difference(left, right)
    histogram = diff.histogram()
    total = left.size[0] * left.size[1] * 3
    absolute = sum((index % 256) * count for index, count in enumerate(histogram))
    return max(0.0, min(1.0, 1.0 - absolute / (total * 255)))


def _open_rgb_over_white(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    return background.convert("RGB")


def _load_golden(name: str, golden_root: Path) -> dict[str, Any] | None:
    candidates = [
        golden_root / name / "uiir.json",
        golden_root / f"{name}.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return None


def _golden_metrics(nodes: list[dict[str, Any]], golden_nodes: list[dict[str, Any]], golden_data: dict[str, Any] | None = None) -> dict[str, Any]:
    predicted = [node for node in nodes if node.get("type") != "Screen"]
    expected = [node for node in golden_nodes if node.get("type") != "Screen"]
    predicted_types = Counter(str(node.get("type") or "Unknown") for node in predicted)
    expected_types = Counter(str(node.get("type") or "Unknown") for node in expected)
    overlap = sum((predicted_types & expected_types).values())
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(expected) if expected else 0.0
    type_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    proposal_precision, proposal_recall = _proposal_metrics(predicted, expected)
    relation_precision, relation_recall, relation_f1 = _relation_scores(predicted, expected)
    component_group_f1 = _component_group_f1(predicted, expected)
    decision_rates = _decision_rates(golden_data or {})
    return {
        "node_count_delta": len(predicted) - len(expected),
        "type_f1": round(type_f1, 5),
        "bbox_mean_iou": round(_mean_bbox_iou(predicted, expected), 5),
        "tree_distance_proxy": abs(len(predicted) - len(expected)) + sum(abs(predicted_types[key] - expected_types[key]) for key in set(predicted_types) | set(expected_types)),
        "proposal_precision": proposal_precision,
        "proposal_recall": proposal_recall,
        "relation_precision": relation_precision,
        "relation_recall": relation_recall,
        "relation_f1": relation_f1,
        "component_group_f1": component_group_f1,
        **decision_rates,
    }


def _proposal_metrics(predicted: list[dict[str, Any]], expected: list[dict[str, Any]]) -> tuple[float, float]:
    predicted_refs = _proposal_refs(predicted)
    expected_refs = _proposal_refs(expected)
    overlap = len(predicted_refs & expected_refs)
    precision = overlap / len(predicted_refs) if predicted_refs else (1.0 if not expected_refs else 0.0)
    recall = overlap / len(expected_refs) if expected_refs else 1.0
    return round(precision, 5), round(recall, 5)


def _proposal_refs(nodes: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for node in nodes:
        for ref in node.get("sourceRefs", []) or []:
            if isinstance(ref, str) and ref.startswith("openai-vision:"):
                refs.add(ref)
    return refs


def _relation_scores(predicted: list[dict[str, Any]], expected: list[dict[str, Any]]) -> tuple[float, float, float]:
    predicted_pairs = _relation_pairs(predicted)
    expected_pairs = _relation_pairs(expected)
    if not predicted_pairs and not expected_pairs:
        return 1.0, 1.0, 1.0
    overlap = len(predicted_pairs & expected_pairs)
    precision = overlap / len(predicted_pairs) if predicted_pairs else 0.0
    recall = overlap / len(expected_pairs) if expected_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 5), round(recall, 5), round(f1, 5)


def _component_group_f1(predicted: list[dict[str, Any]], expected: list[dict[str, Any]]) -> float:
    predicted_groups = _component_groups(predicted)
    expected_groups = _component_groups(expected)
    if not predicted_groups and not expected_groups:
        return 1.0
    overlap = len(predicted_groups & expected_groups)
    precision = overlap / len(predicted_groups) if predicted_groups else 0.0
    recall = overlap / len(expected_groups) if expected_groups else 0.0
    return round(2 * precision * recall / (precision + recall), 5) if precision + recall else 0.0


def _relation_pairs(nodes: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for node in nodes:
        metadata = node.get("metadata", {}) or {}
        group_id = metadata.get("openaiComponentGroupId") or metadata.get("componentGroupId")
        if not group_id:
            continue
        child_id = metadata.get("candidateId") or next(iter(node.get("sourceRefs", []) or []), node.get("id"))
        pairs.add((str(group_id), str(child_id)))
    return pairs


def _component_groups(nodes: list[dict[str, Any]]) -> set[tuple[str, tuple[str, ...]]]:
    groups: dict[str, list[str]] = {}
    for node in nodes:
        metadata = node.get("metadata", {}) or {}
        group_id = metadata.get("openaiComponentGroupId") or metadata.get("componentGroupId")
        if not group_id:
            continue
        child_id = metadata.get("candidateId") or next(iter(node.get("sourceRefs", []) or []), node.get("id"))
        groups.setdefault(str(group_id), []).append(str(child_id))
    return {(group_id, tuple(sorted(children))) for group_id, children in groups.items()}


def _decision_rates(golden_data: dict[str, Any]) -> dict[str, float | None]:
    summary = golden_data.get("metadata", {}).get("golden", {}).get("decisions", {})
    loaded = int(summary.get("loaded") or 0)
    accepted = int(summary.get("accepted") or 0)
    edited = int(summary.get("edited") or 0)
    proposal_accepted = int(summary.get("proposal_accepted") or 0)
    proposal_rejected = int(summary.get("proposal_rejected") or 0)
    proposal_total = proposal_accepted + proposal_rejected
    return {
        "human_accept_rate": round((accepted + edited) / loaded, 5) if loaded else None,
        "quarantine_usefulness": round(proposal_accepted / proposal_total, 5) if proposal_total else None,
    }


def _mean_bbox_iou(predicted: list[dict[str, Any]], expected: list[dict[str, Any]]) -> float:
    matches: list[float] = []
    used: set[int] = set()
    for node in predicted:
        bbox = _node_bbox(node)
        node_type = node.get("type")
        best_index = None
        best_iou = 0.0
        for index, golden in enumerate(expected):
            if index in used or golden.get("type") != node_type:
                continue
            iou = bbox.iou(_node_bbox(golden))
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index is not None:
            used.add(best_index)
            matches.append(best_iou)
    return sum(matches) / len(matches) if matches else 0.0


def _node_bbox(node: dict[str, Any]) -> BBox:
    return BBox.from_any(node.get("bbox") or {"x": 0, "y": 0, "w": 0, "h": 0})


def _flatten_nodes(root: Any) -> list[dict[str, Any]]:
    if not isinstance(root, dict):
        return []
    result: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        result.append(node)
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                visit(child)

    visit(root)
    return result


def _average(values: list[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 5)
