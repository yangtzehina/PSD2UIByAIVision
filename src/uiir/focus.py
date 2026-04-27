from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .models import BBox


FOCUS_ISSUE_TYPES = {"missing", "extra", "misclassified"}


def build_focus_tiles(
    extract_output: str | Path,
    output_dir: str | Path | None = None,
    padding: int = 32,
    max_tiles: int = 12,
) -> dict[str, Any]:
    source_dir = Path(extract_output).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else source_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    composite_path = source_dir / "composite.png"
    review_path = _review_path(source_dir, out_dir)
    tiles_dir = out_dir / "focus_tiles"
    manifest_path = out_dir / "focus_tiles.json"
    graph_path = _graph_path(source_dir, out_dir)

    report: dict[str, Any] = {
        "version": "0.1",
        "source_output": source_dir.as_posix(),
        "output": out_dir.as_posix(),
        "padding": max(0, int(padding)),
        "max_tiles": max(0, int(max_tiles)),
        "issue_types": sorted(FOCUS_ISSUE_TYPES),
        "artifacts": {
            "focus_tiles_dir": tiles_dir.as_posix(),
            "focus_tiles_json": manifest_path.as_posix(),
        },
        "source_artifacts": {
            "composite": composite_path.as_posix(),
            "render_review": review_path.as_posix(),
        },
        "tiles": [],
        "tile_count": 0,
        "eligible_issue_count": 0,
        "truncated": False,
    }
    if graph_path:
        report["graph_metadata_path"] = graph_path.as_posix()

    if not composite_path.exists():
        report["status"] = "error"
        report["error"] = "composite.png missing"
        _write_json(manifest_path, report)
        return report
    if not review_path.exists():
        report["status"] = "error"
        report["error"] = "render_review.json missing"
        _write_json(manifest_path, report)
        return report

    review = _read_json(review_path, default={})
    issues = review.get("issues", []) if isinstance(review, dict) else []
    if not isinstance(issues, list):
        issues = []

    with Image.open(composite_path) as opened:
        composite = opened.convert("RGBA")

    tiles_dir.mkdir(parents=True, exist_ok=True)
    eligible = [issue for issue in issues if isinstance(issue, dict) and issue.get("type") in FOCUS_ISSUE_TYPES]
    report["eligible_issue_count"] = len(eligible)

    for issue in eligible[: report["max_tiles"]]:
        tile = _tile_for_issue(issue, len(report["tiles"]) + 1, composite, tiles_dir, report["padding"], graph_path)
        if tile:
            report["tiles"].append(tile)

    report["tile_count"] = len(report["tiles"])
    report["truncated"] = len(eligible) > report["max_tiles"]
    report["status"] = "ok"
    _write_json(manifest_path, report)
    return report


def _tile_for_issue(
    issue: dict[str, Any],
    index: int,
    composite: Image.Image,
    tiles_dir: Path,
    padding: int,
    graph_path: Path | None,
) -> dict[str, Any] | None:
    try:
        source_bbox = BBox.from_any(issue["bbox"])
    except Exception:
        return None
    if source_bbox.is_empty:
        return None

    tile_bbox = _padded_bbox(source_bbox, padding, composite.width, composite.height)
    if tile_bbox.is_empty:
        return None

    issue_id = str(issue.get("id") or f"issue-{index}")
    issue_type = str(issue.get("type") or "unknown")
    tile_id = f"focus_{index:03d}"
    tile_path = tiles_dir / f"{tile_id}_{_slug(issue_id)}_{_slug(issue_type)}.png"
    crop = composite.crop((tile_bbox.x, tile_bbox.y, tile_bbox.right, tile_bbox.bottom))
    crop.save(tile_path)

    related_nodes = _related_nodes(issue)
    tile: dict[str, Any] = {
        "id": tile_id,
        "path": tile_path.as_posix(),
        "bbox": tile_bbox.to_dict(),
        "source_issue": {
            "id": issue_id,
            "type": issue_type,
            "bbox": source_bbox.to_dict(),
        },
        "source_issue_id": issue_id,
        "source_issue_type": issue_type,
        "related_nodes": related_nodes,
    }
    if graph_path:
        tile["graph_metadata_path"] = graph_path.as_posix()
    return tile


def _padded_bbox(box: BBox, padding: int, width: int, height: int) -> BBox:
    return BBox.from_xyxy(box.x - padding, box.y - padding, box.right + padding, box.bottom + padding).clamp(width, height)


def _related_nodes(issue: dict[str, Any]) -> list[str]:
    related = issue.get("overlapping_nodes")
    if related is None:
        related = issue.get("related_nodes")
    if related is None:
        related = issue.get("related_node_ids")
    if not isinstance(related, list):
        return []
    return [str(node_id) for node_id in related if node_id is not None]


def _graph_path(source_dir: Path, out_dir: Path) -> Path | None:
    for path in (source_dir / "ui_graph.json", out_dir / "ui_graph.json"):
        if path.exists():
            return path
    return None


def _review_path(source_dir: Path, out_dir: Path) -> Path:
    source_path = source_dir / "render_review.json"
    if source_path.exists():
        return source_path
    return out_dir / "render_review.json"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return slug or "issue"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
