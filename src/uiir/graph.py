from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .models import BBox


EDGE_COLORS = {
    "psd_parent": "#64748b",
    "contains": "#22c55e",
    "overlaps": "#f97316",
    "same_row": "#2563eb",
    "same_col": "#7c3aed",
    "same_size": "#06b6d4",
    "repeated_pattern": "#db2777",
    "text_on_image": "#f59e0b",
    "component_group_candidate": "#14b8a6",
}

NODE_COLORS = {
    "Button": "#14b8a6",
    "Text": "#2563eb",
    "Image": "#f59e0b",
    "Icon": "#8b5cf6",
    "Container": "#ef4444",
    "Background": "#64748b",
    "List": "#22c55e",
    "ScrollView": "#22c55e",
    "Grid": "#22c55e",
    "Input": "#06b6d4",
    "Unknown": "#f97316",
}


@dataclass(frozen=True)
class GraphBuildResult:
    graph_json: Path
    graph_overlay: Path
    graph: dict[str, Any]


def build_ui_graph(extract_output: str | Path, output_dir: str | Path | None = None) -> GraphBuildResult:
    source_dir = Path(extract_output).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else source_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    uiir = _read_json(source_dir / "uiir.json", default={})
    candidates = _read_json(source_dir / "candidates.json", default=[])
    layers_data = _read_json(source_dir / "layer_metadata.json", default=[])
    layers = layers_data.get("layers", []) if isinstance(layers_data, dict) else layers_data
    nodes = _candidate_nodes(candidates) or _uiir_nodes(uiir)
    width = int(uiir.get("width") or _infer_width(nodes) or 0)
    height = int(uiir.get("height") or _infer_height(nodes) or 0)
    edges = _build_edges(nodes, layers)

    graph = {
        "version": "0.1",
        "source": uiir.get("source") or source_dir.name,
        "source_output": source_dir.as_posix(),
        "width": width,
        "height": height,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "edge_type_counts": dict(Counter(edge["type"] for edge in edges)),
        },
    }
    graph_json = out_dir / "ui_graph.json"
    graph_json.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    overlay_path = draw_graph_overlay(source_dir / "composite.png", graph, out_dir / "graph_overlay.png")
    return GraphBuildResult(graph_json=graph_json, graph_overlay=overlay_path, graph=graph)


def draw_graph_overlay(image_path: str | Path, graph: dict[str, Any], output_path: str | Path, max_edges: int = 260) -> Path:
    image_file = Path(image_path)
    if image_file.exists():
        image = Image.open(image_file).convert("RGBA")
    else:
        width = int(graph.get("width") or 1)
        height = int(graph.get("height") or 1)
        image = Image.new("RGBA", (max(1, width), max(1, height)), (255, 255, 255, 255))

    draw = ImageDraw.Draw(image, "RGBA")
    font = _font()
    nodes = {node["id"]: node for node in graph.get("nodes", []) if node.get("id")}

    for edge in sorted(graph.get("edges", []) or [], key=lambda item: float(item.get("confidence") or 0), reverse=True)[:max_edges]:
        left = nodes.get(edge.get("from"))
        right = nodes.get(edge.get("to"))
        if not left or not right:
            continue
        a = BBox.from_any(left["bbox"])
        b = BBox.from_any(right["bbox"])
        ax, ay = a.center
        bx, by = b.center
        color = _hex_rgba(EDGE_COLORS.get(edge.get("type"), "#94a3b8"), 145)
        draw.line((ax, ay, bx, by), fill=color, width=2)

    for node in graph.get("nodes", []) or []:
        try:
            box = BBox.from_any(node["bbox"]).clamp(image.width, image.height)
        except Exception:
            continue
        if box.is_empty:
            continue
        color = _hex_rgba(NODE_COLORS.get(str(node.get("type") or "Unknown"), NODE_COLORS["Unknown"]), 210)
        fill = (*color[:3], 24)
        draw.rectangle((box.x, box.y, box.right, box.bottom), outline=color, fill=fill, width=2)
        label = str(node.get("label") or node.get("id"))
        label_box = draw.textbbox((0, 0), label, font=font)
        label_w = label_box[2] - label_box[0] + 8
        label_h = label_box[3] - label_box[1] + 6
        x0 = max(0, min(box.x, image.width - 1))
        y0 = max(0, box.y - label_h) if box.y >= label_h else min(image.height - 1, box.bottom)
        x1 = min(image.width, x0 + label_w)
        y1 = min(image.height, y0 + label_h)
        if x1 > x0 and y1 > y0:
            draw.rectangle((x0, y0, x1, y1), fill=color)
            draw.text((x0 + 4, y0 + 2), label, fill=(255, 255, 255, 255), font=font)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _build_edges(nodes: list[dict[str, Any]], layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    by_id = {node["id"]: node for node in nodes}
    by_layer = _nodes_by_layer(nodes)
    layer_parent = {str(layer.get("id")): layer.get("parent_id") for layer in layers if layer.get("id")}

    def add(edge_type: str, source: str, target: str, confidence: float, reason: str, metadata: dict[str, Any] | None = None) -> None:
        if source == target or source not in by_id or target not in by_id:
            return
        key = (edge_type, source, target)
        if key in seen:
            return
        seen.add(key)
        edges.append(
            {
                "id": f"e{len(edges) + 1}",
                "type": edge_type,
                "from": source,
                "to": target,
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "reason": reason,
                "metadata": metadata or {},
            }
        )

    for node in nodes:
        parent = node.get("parent_hint")
        if parent in by_id:
            add("psd_parent", str(parent), node["id"], 0.96, "candidate parent_hint references another candidate")
        for ref in node.get("sourceRefs", []) or []:
            ref_text = str(ref)
            parent_layer = layer_parent.get(ref_text)
            if parent_layer:
                for parent_node in by_layer.get(str(parent_layer), []):
                    add("psd_parent", parent_node["id"], node["id"], 0.9, "PSD layer parent relationship")

    ordered = sorted(nodes, key=lambda node: BBox.from_any(node["bbox"]).area, reverse=True)
    for i, left in enumerate(ordered):
        left_box = BBox.from_any(left["bbox"])
        if left_box.is_empty:
            continue
        for right in ordered[i + 1 :]:
            right_box = BBox.from_any(right["bbox"])
            if right_box.is_empty:
                continue
            larger, smaller = (left, right) if left_box.area >= right_box.area else (right, left)
            large_box = BBox.from_any(larger["bbox"])
            small_box = BBox.from_any(smaller["bbox"])
            if large_box.contains_bbox(small_box, padding=3) and small_box.area / max(1, large_box.area) < 0.96:
                add("contains", larger["id"], smaller["id"], _contain_confidence(large_box, small_box), "bbox containment")

            overlap = left_box.overlap_ratio(right_box)
            if overlap >= 0.25 and not left_box.contains_bbox(right_box, padding=3) and not right_box.contains_bbox(left_box, padding=3):
                add("overlaps", left["id"], right["id"], min(0.92, overlap), "bbox overlap")

            if _same_row(left_box, right_box):
                add("same_row", left["id"], right["id"], 0.72, "aligned vertical centers")
            if _same_col(left_box, right_box):
                add("same_col", left["id"], right["id"], 0.72, "aligned horizontal centers")
            if _same_size(left_box, right_box):
                add("same_size", left["id"], right["id"], 0.68, "similar width and height")
            if _text_on_image(left, right, left_box, right_box):
                text_node, image_node = (left, right) if _is_text(left) else (right, left)
                add("text_on_image", image_node["id"], text_node["id"], 0.86, "text centered over visual element")
                add("component_group_candidate", image_node["id"], text_node["id"], 0.82, "text plus visual element can form a component")

            left_group = left.get("metadata", {}).get("openaiComponentGroupId") or left.get("metadata", {}).get("componentGroupId")
            right_group = right.get("metadata", {}).get("openaiComponentGroupId") or right.get("metadata", {}).get("componentGroupId")
            if left_group and left_group == right_group:
                add(
                    "component_group_candidate",
                    left["id"],
                    right["id"],
                    0.9,
                    "shared component group id",
                    {"component_group_id": left_group},
                )

    for group in _repeat_groups(nodes):
        for left, right in zip(group, group[1:]):
            add(
                "repeated_pattern",
                left["id"],
                right["id"],
                0.84,
                "same type, similar size, repeated alignment",
                {"pattern_size": len(group)},
            )

    return edges


def _candidate_nodes(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("id") or not candidate.get("bbox"):
            continue
        metadata = dict(candidate.get("metadata") or {})
        node_type = candidate.get("type_hint") or candidate.get("type") or "Unknown"
        nodes.append(
            {
                "id": str(candidate["id"]),
                "label": f"{candidate['id']}:{node_type}",
                "type": node_type,
                "bbox": BBox.from_any(candidate["bbox"]).to_dict(),
                "confidence": candidate.get("confidence"),
                "source": candidate.get("source"),
                "sourceRefs": list(candidate.get("source_refs") or candidate.get("sourceRefs") or []),
                "parent_hint": candidate.get("parent_hint"),
                "text": candidate.get("text"),
                "role": candidate.get("role"),
                "layout": candidate.get("layout"),
                "metadata": metadata,
            }
        )
    return nodes


def _uiir_nodes(uiir: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = []
    for node in _flatten_uiir(uiir.get("root")):
        if node.get("type") == "Screen" or not node.get("bbox"):
            continue
        nodes.append(
            {
                "id": str(node.get("id")),
                "label": f"{node.get('id')}:{node.get('type')}",
                "type": node.get("type") or "Unknown",
                "bbox": BBox.from_any(node["bbox"]).to_dict(),
                "confidence": node.get("confidence"),
                "source": "uiir-node",
                "sourceRefs": list(node.get("sourceRefs") or []),
                "parent_hint": node.get("metadata", {}).get("parentCandidateId"),
                "text": node.get("text"),
                "role": node.get("role"),
                "layout": node.get("layout"),
                "metadata": dict(node.get("metadata") or {}),
            }
        )
    return nodes


def _nodes_by_layer(nodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        for ref in node.get("sourceRefs", []) or []:
            mapping[str(ref)].append(node)
    return mapping


def _repeat_groups(nodes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    buckets: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        box = BBox.from_any(node["bbox"])
        if box.is_empty or box.area <= 0:
            continue
        buckets[(str(node.get("type") or "Unknown"), round(box.w / 12), round(box.h / 12))].append(node)
    groups: list[list[dict[str, Any]]] = []
    for bucket in buckets.values():
        if len(bucket) < 3:
            continue
        row_sorted = sorted(bucket, key=lambda node: (BBox.from_any(node["bbox"]).center[1], BBox.from_any(node["bbox"]).center[0]))
        if _aligned_sequence(row_sorted, axis="row"):
            groups.append(row_sorted)
            continue
        col_sorted = sorted(bucket, key=lambda node: (BBox.from_any(node["bbox"]).center[0], BBox.from_any(node["bbox"]).center[1]))
        if _aligned_sequence(col_sorted, axis="col"):
            groups.append(col_sorted)
    return groups


def _aligned_sequence(nodes: list[dict[str, Any]], axis: str) -> bool:
    if len(nodes) < 3:
        return False
    centers = [BBox.from_any(node["bbox"]).center for node in nodes]
    if axis == "row":
        spread = max(y for _, y in centers) - min(y for _, y in centers)
        avg_h = sum(BBox.from_any(node["bbox"]).h for node in nodes) / len(nodes)
        return spread <= max(8, avg_h * 0.35)
    spread = max(x for x, _ in centers) - min(x for x, _ in centers)
    avg_w = sum(BBox.from_any(node["bbox"]).w for node in nodes) / len(nodes)
    return spread <= max(8, avg_w * 0.35)


def _text_on_image(left: dict[str, Any], right: dict[str, Any], left_box: BBox, right_box: BBox) -> bool:
    if _is_text(left) == _is_text(right):
        return False
    text_box = left_box if _is_text(left) else right_box
    image_box = right_box if _is_text(left) else left_box
    image_type = str((right if _is_text(left) else left).get("type") or "Unknown")
    if image_type not in {"Button", "Image", "Icon", "Background", "Container", "Unknown"}:
        return False
    tx, ty = text_box.center
    return image_box.contains_point(tx, ty) or image_box.overlap_ratio(text_box) >= 0.55


def _is_text(node: dict[str, Any]) -> bool:
    return str(node.get("type") or "") == "Text" or bool(node.get("text"))


def _same_row(left: BBox, right: BBox) -> bool:
    _, ly = left.center
    _, ry = right.center
    return abs(ly - ry) <= max(8, min(left.h, right.h) * 0.45)


def _same_col(left: BBox, right: BBox) -> bool:
    lx, _ = left.center
    rx, _ = right.center
    return abs(lx - rx) <= max(8, min(left.w, right.w) * 0.45)


def _same_size(left: BBox, right: BBox) -> bool:
    if left.area <= 0 or right.area <= 0:
        return False
    w_ratio = min(left.w, right.w) / max(left.w, right.w)
    h_ratio = min(left.h, right.h) / max(left.h, right.h)
    return w_ratio >= 0.88 and h_ratio >= 0.88


def _contain_confidence(outer: BBox, inner: BBox) -> float:
    area_ratio = inner.area / max(1, outer.area)
    return max(0.55, min(0.95, 1.0 - area_ratio * 0.4))


def _flatten_uiir(root: Any) -> list[dict[str, Any]]:
    if not isinstance(root, dict):
        return []
    nodes = [root]
    for child in root.get("children", []) or []:
        nodes.extend(_flatten_uiir(child))
    return nodes


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _infer_width(nodes: list[dict[str, Any]]) -> int:
    return max((BBox.from_any(node["bbox"]).right for node in nodes if node.get("bbox")), default=1)


def _infer_height(nodes: list[dict[str, Any]]) -> int:
    return max((BBox.from_any(node["bbox"]).bottom for node in nodes if node.get("bbox")), default=1)


def _hex_rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    normalized = value.lstrip("#")
    return (int(normalized[0:2], 16), int(normalized[2:4], 16), int(normalized[4:6], 16), alpha)


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", 12)
    except Exception:
        return ImageFont.load_default()
