from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from PIL import Image

from .heuristics import coerce_node_type, infer_node_type, infer_role, is_comment_layer
from .models import BBox, Candidate, PARENT_NODE_TYPES, UIIRDocument, UINode
from .psd import PSDExtractResult


def build_layer_candidates(layers: list, min_area: int = 16) -> list[Candidate]:
    candidates: list[Candidate] = []
    for layer in layers:
        if not layer.visible or layer.bbox.area < min_area or is_comment_layer(layer.name):
            continue
        node_type, base_confidence = infer_node_type(layer.name, layer.kind, layer.is_group)
        confidence = max(0.05, min(0.98, base_confidence * (0.9 + layer.opacity * 0.1)))
        candidate = Candidate(
            id=f"c{len(candidates) + 1}",
            bbox=layer.bbox,
            source="psd-layer",
            type_hint=node_type,
            confidence=confidence,
            source_refs=[layer.id],
            name=layer.name,
            text=layer.text,
            style=_style_to_attr(layer.style),
            role=infer_role(layer.name, node_type),
            asset=layer.asset,
            parent_hint=layer.parent_id,
            metadata={
                "name": layer.name,
                "path": layer.path,
                "kind": layer.kind,
                "isGroup": layer.is_group,
                "opacity": layer.opacity,
                "depth": layer.depth,
                "textStyle": layer.style,
                "psdParentId": layer.parent_id,
                "psdPath": layer.path,
                "psdDepth": layer.depth,
            },
        )
        candidates.append(candidate)
    return candidates


def build_visual_candidates(composite_path: str | Path, min_area: int = 96, max_candidates: int = 180) -> list[Candidate]:
    path = Path(composite_path)
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return _alpha_component_candidates(path, min_area=min_area, max_candidates=max_candidates)

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return []

    height, width = image.shape[:2]
    boxes: list[BBox] = []
    if image.shape[2] == 4 and image[:, :, 3].min() < 250:
        mask = (image[:, :, 3] > 8).astype("uint8") * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes.extend(_boxes_from_contours(contours, width, height, min_area, max_area_ratio=0.95))

    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes.extend(_boxes_from_contours(contours, width, height, min_area, max_area_ratio=0.6))

    # UIED-style low-level proposal pass: threshold local contrast, close gaps,
    # then treat rectangular connected regions as possible UI components.
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes.extend(_boxes_from_contours(contours, width, height, min_area, max_area_ratio=0.45))

    boxes = _dedupe_boxes(boxes)
    boxes = sorted(boxes, key=lambda box: box.area, reverse=True)[:max_candidates]
    candidates: list[Candidate] = []
    for box in boxes:
        candidates.append(
            Candidate(
                id=f"v{len(candidates) + 1}",
                bbox=box,
                source="visual-contour",
                type_hint="Unknown",
                confidence=0.38,
                source_refs=[f"visual:{len(candidates) + 1}"],
            )
        )
    return candidates


def build_ocr_candidates(composite_path: str | Path, min_area: int = 12, max_candidates: int = 120) -> list[Candidate]:
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return []
    try:
        image = Image.open(composite_path).convert("RGB")
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    candidates: list[Candidate] = []
    count = len(data.get("text", []))
    for index in range(count):
        text = str(data["text"][index] or "").strip()
        if not text:
            continue
        confidence = _parse_ocr_confidence(data.get("conf", [0])[index])
        if confidence < 0.2:
            continue
        box = BBox(
            int(data["left"][index]),
            int(data["top"][index]),
            int(data["width"][index]),
            int(data["height"][index]),
        )
        if box.area < min_area:
            continue
        candidates.append(
            Candidate(
                id=f"ocr{len(candidates) + 1}",
                bbox=box,
                source="ocr",
                type_hint="Text",
                confidence=max(0.45, min(0.9, confidence)),
                source_refs=[f"ocr:{len(candidates) + 1}"],
                text=text,
                style="source:ocr",
            )
        )
        if len(candidates) >= max_candidates:
            break
    return candidates


def fuse_candidates(candidates: list[Candidate]) -> list[Candidate]:
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (candidate.confidence, candidate.bbox.area),
        reverse=True,
    )
    fused: list[Candidate] = []
    for candidate in sorted_candidates:
        duplicate = _find_duplicate(candidate, fused)
        if duplicate is None:
            candidate.id = f"c{len(fused) + 1}"
            fused.append(candidate)
            continue
        duplicate.source_refs = sorted(set([*duplicate.source_refs, *candidate.source_refs]))
        duplicate.confidence = max(duplicate.confidence, candidate.confidence)
        if duplicate.type_hint == "Unknown" and candidate.type_hint != "Unknown":
            duplicate.type_hint = candidate.type_hint
        if not duplicate.text and candidate.text:
            duplicate.text = candidate.text
        if not duplicate.style and candidate.style:
            duplicate.style = candidate.style
        if not duplicate.asset and candidate.asset:
            duplicate.asset = candidate.asset
        if not duplicate.parent_hint and candidate.parent_hint:
            duplicate.parent_hint = candidate.parent_hint
        duplicate.metadata.setdefault("mergedSources", []).append(candidate.to_dict())
    return sorted(fused, key=lambda candidate: (candidate.bbox.area, candidate.confidence), reverse=True)


def infer_uiir_document(extract: PSDExtractResult, candidates: list[Candidate]) -> UIIRDocument:
    root = UINode(
        id="n1",
        type="Screen",
        bbox=BBox(0, 0, extract.width, extract.height),
        confidence=1.0,
        source_refs=["document"],
        children=[],
    )
    nodes = [_candidate_to_node(candidate, index + 2) for index, candidate in enumerate(candidates)]
    node_by_candidate = {candidate.id: node for candidate, node in zip(candidates, nodes)}
    candidate_by_layer_ref = _candidate_by_layer_ref(candidates)
    candidate_by_node = {node.id: candidate for candidate, node in zip(candidates, nodes)}

    for node in sorted(nodes, key=lambda item: item.bbox.area, reverse=True):
        candidate = candidate_by_node[node.id]
        parent = _find_parent(node, root, nodes, candidate, node_by_candidate, candidate_by_layer_ref)
        parent.children.append(node)

    next_node_id = _componentize_tree(root, start_index=len(nodes) + 2)
    _infer_layouts(root)
    _sort_tree(root)
    return UIIRDocument(
        version="0.1",
        source=extract.source.name,
        width=extract.width,
        height=extract.height,
        assets_root="assets/",
        root=root,
        candidates=candidates,
        metadata={
            "pipeline": "psd-layer+visual-candidates",
            "candidateCount": len(candidates),
            "componentized": True,
            "nextNodeId": next_node_id,
            "warnings": extract.warnings,
        },
    )


def detect_candidates(
    extract: PSDExtractResult,
    include_visual: bool = True,
    include_ocr: bool = False,
    min_area: int = 96,
) -> list[Candidate]:
    candidates = build_layer_candidates(extract.layers, min_area=min_area)
    if include_visual:
        candidates.extend(build_visual_candidates(extract.composite_path, min_area=min_area))
    if include_ocr:
        candidates.extend(build_ocr_candidates(extract.composite_path, min_area=max(12, min_area // 4)))
    return fuse_candidates(candidates)


def _candidate_to_node(candidate: Candidate, index: int) -> UINode:
    return UINode(
        id=f"n{index}",
        type=coerce_node_type(candidate.type_hint),
        bbox=candidate.bbox,
        confidence=candidate.confidence,
        source_refs=candidate.source_refs or [candidate.id],
        role=candidate.role,
        text=candidate.text,
        style=candidate.style,
        layout=candidate.layout,
        asset=candidate.asset,
        metadata={
            "candidateId": candidate.id,
            "source": candidate.source,
            "name": candidate.name,
            "psdParentId": candidate.metadata.get("psdParentId") or candidate.parent_hint,
            "psdPath": candidate.metadata.get("psdPath") or candidate.metadata.get("path"),
            "psdDepth": candidate.metadata.get("psdDepth") if candidate.metadata.get("psdDepth") is not None else candidate.metadata.get("depth"),
            "psdLayerKind": candidate.metadata.get("kind"),
            "isPsdGroup": candidate.metadata.get("isGroup"),
            "openaiComponentGroupId": candidate.metadata.get("openaiComponentGroupId"),
            "originalCandidateType": candidate.type_hint,
        },
    )


def _candidate_by_layer_ref(candidates: list[Candidate]) -> dict[str, Candidate]:
    by_ref: dict[str, Candidate] = {}
    for candidate in candidates:
        for ref in candidate.source_refs:
            if ref.startswith("layer:"):
                by_ref.setdefault(ref, candidate)
    return by_ref


def _find_parent(
    node: UINode,
    root: UINode,
    all_nodes: list[UINode],
    candidate: Candidate,
    node_by_candidate: dict[str, UINode],
    candidate_by_layer_ref: dict[str, Candidate],
) -> UINode:
    explicit = _explicit_parent(node, candidate, node_by_candidate, candidate_by_layer_ref)
    if explicit:
        return explicit
    candidates = [
        other
        for other in all_nodes
        if other is not node
        and other.type in PARENT_NODE_TYPES
        and other.bbox.area > node.bbox.area
        and other.bbox.contains_bbox(node.bbox, padding=3)
    ]
    if not candidates:
        return root
    return min(candidates, key=lambda item: item.bbox.area)


def _explicit_parent(
    node: UINode,
    candidate: Candidate,
    node_by_candidate: dict[str, UINode],
    candidate_by_layer_ref: dict[str, Candidate],
) -> UINode | None:
    hint = candidate.parent_hint
    if not hint:
        return None
    parent = node_by_candidate.get(hint)
    if parent is None:
        layer_parent = candidate_by_layer_ref.get(hint)
        if layer_parent is not None:
            parent = node_by_candidate.get(layer_parent.id)
    if parent is None or parent is node or not _can_parent_explicit(parent):
        return None
    if not parent.bbox.contains_bbox(node.bbox, padding=12) and parent.bbox.overlap_ratio(node.bbox) < 0.8:
        return None
    return parent


def _can_parent_explicit(node: UINode) -> bool:
    return node.type in PARENT_NODE_TYPES or node.type == "Background" or bool(node.metadata.get("isPsdGroup"))


def _componentize_tree(root: UINode, start_index: int) -> int:
    next_index = start_index
    for node in list(_walk_nodes(root)):
        if not node.children:
            continue
        node.children, next_index = _wrap_component_children(node.children, next_index)
    return next_index


def _wrap_component_children(children: list[UINode], next_index: int) -> tuple[list[UINode], int]:
    wrapped: list[UINode] = []
    used: set[int] = set()
    openai_groups = _openai_groups(children)
    for group in openai_groups:
        indexes = [children.index(child) for child in group if child in children]
        if any(index in used for index in indexes):
            continue
        component_type = _component_type_for_group(group) or "Container"
        component = _make_component_node(next_index, component_type, group, "openai_component_group")
        next_index += 1
        wrapped.append(component)
        used.update(indexes)

    for index, child in enumerate(children):
        if index in used:
            continue
        partner_index = _find_component_partner(child, children, used)
        if partner_index is not None:
            group = [child, children[partner_index]]
            component_type = _component_type_for_group(group) or "Button"
            component = _make_component_node(next_index, component_type, group, _grouping_reason(group))
            next_index += 1
            wrapped.append(component)
            used.update({index, partner_index})
            continue
        if _is_single_component_layer(child):
            component_type = "Button" if child.type == "Button" else child.type
            component = _make_component_node(next_index, component_type, [child], "single_component_layer")
            next_index += 1
            wrapped.append(component)
            used.add(index)
            continue
        wrapped.append(child)
        used.add(index)
    return wrapped, next_index


def _openai_groups(children: list[UINode]) -> list[list[UINode]]:
    groups: dict[str, list[UINode]] = {}
    for child in children:
        group_id = child.metadata.get("openaiComponentGroupId")
        if isinstance(group_id, str) and group_id.strip():
            groups.setdefault(group_id.strip(), []).append(child)
    return [group for group in groups.values() if len(group) >= 2]


def _find_component_partner(child: UINode, children: list[UINode], used: set[int]) -> int | None:
    if not _is_component_part(child):
        return None
    best_index = None
    best_score = 0.0
    for index, other in enumerate(children):
        if other is child or index in used or not _is_component_part(other):
            continue
        if not _types_complement(child, other):
            continue
        score = _component_pair_score(child, other)
        if score > best_score:
            best_index = index
            best_score = score
    if best_score >= 0.45:
        return best_index
    return None


def _component_pair_score(left: UINode, right: UINode) -> float:
    outer, inner = (left, right) if left.bbox.area >= right.bbox.area else (right, left)
    score = 0.0
    if outer.bbox.contains_point(*inner.bbox.center):
        score += 0.45
    if outer.bbox.contains_bbox(inner.bbox, padding=8):
        score += 0.35
    if outer.bbox.overlap_ratio(inner.bbox) > 0.35:
        score += 0.2
    if _name_family(left) and _name_family(left) == _name_family(right):
        score += 0.25
    if _name_indicates_component(left) or _name_indicates_component(right):
        score += 0.25
    return min(score, 1.0)


def _types_complement(left: UINode, right: UINode) -> bool:
    types = {left.type, right.type}
    if "Text" in types and types & {"Button", "Image", "Background", "Icon", "Unknown", "Container"}:
        return True
    if types == {"Icon", "Text"}:
        return True
    return False


def _is_component_part(node: UINode) -> bool:
    return node.type in {"Button", "Image", "Background", "Icon", "Text", "Unknown", "Container"} and not node.metadata.get("component")


def _is_single_component_layer(node: UINode) -> bool:
    if node.metadata.get("component") or node.children:
        return False
    return node.type in {"Button", "Input", "Toggle", "Slider"} and _name_indicates_component(node)


def _component_type_for_group(group: list[UINode]) -> str | None:
    types = {node.type for node in group}
    if types & {"Input", "Toggle", "Slider"}:
        for preferred in ("Input", "Toggle", "Slider"):
            if preferred in types:
                return preferred
    if "Button" in types or any(_name_indicates_component(node) for node in group):
        return "Button"
    if "Icon" in types and "Text" in types:
        return "Container"
    return "Container"


def _grouping_reason(group: list[UINode]) -> str:
    types = {node.type for node in group}
    if "Text" in types and types & {"Button", "Image", "Background", "Unknown"}:
        return "background_text_component"
    if types == {"Icon", "Text"}:
        return "icon_text_component"
    return "sibling_component"


def _make_component_node(index: int, component_type: str, children: list[UINode], reason: str) -> UINode:
    component_children = [_prepare_component_child(child, component_type) for child in children]
    bbox = _union_bbox([child.bbox for child in component_children])
    source_refs = sorted({ref for child in component_children for ref in child.source_refs})
    role = next((child.role for child in component_children if child.role), None)
    text = " ".join(child.text for child in component_children if child.text) or None
    return UINode(
        id=f"n{index}",
        type=coerce_node_type(component_type),
        bbox=bbox,
        confidence=max([child.confidence for child in component_children] or [0.75]),
        source_refs=source_refs,
        role=role,
        text=text,
        children=component_children,
        metadata={
            "component": True,
            "groupingReason": reason,
            "sourceChildIds": [child.id for child in component_children],
            "psdParentId": _common_metadata_value(component_children, "psdParentId"),
            "psdPath": _common_psd_prefix(component_children),
        },
    )


def _prepare_component_child(node: UINode, component_type: str) -> UINode:
    if component_type == "Button" and node.type == "Button":
        node.metadata.setdefault("originalType", node.type)
        if node.text and not node.asset:
            node.type = "Text"
        elif _name_has_background(node):
            node.type = "Background"
        else:
            node.type = "Image"
    return node


def _union_bbox(boxes: list[BBox]) -> BBox:
    if not boxes:
        return BBox(0, 0, 0, 0)
    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.right for box in boxes)
    bottom = max(box.bottom for box in boxes)
    return BBox.from_xyxy(left, top, right, bottom)


def _common_metadata_value(nodes: list[UINode], key: str):
    values = {node.metadata.get(key) for node in nodes if node.metadata.get(key)}
    return values.pop() if len(values) == 1 else None


def _common_psd_prefix(nodes: list[UINode]) -> str | None:
    paths = [str(node.metadata.get("psdPath") or "") for node in nodes if node.metadata.get("psdPath")]
    if not paths:
        return None
    split_paths = [path.split("/")[:-1] for path in paths]
    prefix: list[str] = []
    for parts in zip(*split_paths):
        if len(set(parts)) == 1:
            prefix.append(parts[0])
        else:
            break
    return "/".join(prefix) or None


def _name_text(node: UINode) -> str:
    return " ".join(
        str(value or "")
        for value in (
            node.metadata.get("name"),
            node.metadata.get("psdPath"),
            node.text,
            node.role,
        )
    ).lower()


def _name_indicates_component(node: UINode) -> bool:
    text = _name_text(node)
    return any(word in text for word in ("button", "btn", "按钮", "确定", "取消", "close", "buy", "equip", "ok"))


def _name_has_background(node: UINode) -> bool:
    text = _name_text(node)
    return any(word in text for word in ("bg", "background", "back", "底", "背景"))


def _name_family(node: UINode) -> str:
    text = str(node.metadata.get("name") or node.metadata.get("psdPath") or "").lower()
    for token in ("background", "button", "btn", "text", "label", "bg", "背景", "文字"):
        text = text.replace(token, "")
    return "".join(ch for ch in text if ch.isalnum())[:24]


def _infer_layouts(root: UINode) -> None:
    for node in _walk_nodes(root):
        if node.type == "Screen":
            continue
        if len(node.children) < 3:
            continue
        layout = _repeated_layout(node.children)
        if not layout:
            continue
        if not node.layout:
            node.layout = layout
        if node.type in {"Container", "Unknown"}:
            node.type = "Grid" if layout == "grid" else "List"
            node.confidence = max(node.confidence, 0.68)


def _walk_nodes(node: UINode):
    yield node
    for child in node.children:
        yield from _walk_nodes(child)


def _repeated_layout(children: list[UINode]) -> str | None:
    boxes = [child.bbox for child in children if child.bbox.area > 0 and child.type != "Background"]
    if len(boxes) < 3:
        return None
    widths = [box.w for box in boxes]
    heights = [box.h for box in boxes]
    if _relative_spread(widths) > 0.45 or _relative_spread(heights) > 0.45:
        return None
    xs = [box.x for box in boxes]
    ys = [box.y for box in boxes]
    x_groups = _cluster_positions(xs, tolerance=max(8, int(sum(widths) / len(widths) * 0.25)))
    y_groups = _cluster_positions(ys, tolerance=max(8, int(sum(heights) / len(heights) * 0.25)))
    if len(x_groups) >= 2 and len(y_groups) >= 2 and len(boxes) >= 4:
        return "grid"
    if len(x_groups) == 1 and len(y_groups) >= 3:
        return "vertical"
    if len(y_groups) == 1 and len(x_groups) >= 3:
        return "horizontal"
    return None


def _relative_spread(values: list[int]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    return (max(values) - min(values)) / mean


def _cluster_positions(values: list[int], tolerance: int) -> list[list[int]]:
    groups: list[list[int]] = []
    for value in sorted(values):
        if not groups or abs(value - groups[-1][-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _sort_tree(node: UINode) -> None:
    node.children.sort(key=lambda item: (item.bbox.y, item.bbox.x, item.bbox.area))
    for child in node.children:
        _sort_tree(child)


def _find_duplicate(candidate: Candidate, existing: list[Candidate]) -> Candidate | None:
    for other in existing:
        if candidate.bbox.iou(other.bbox) >= 0.88:
            return other
        if candidate.bbox.overlap_ratio(other.bbox) >= 0.92:
            if candidate.type_hint == other.type_hint or "Unknown" in {candidate.type_hint, other.type_hint}:
                return other
    return None


def _boxes_from_contours(contours, width: int, height: int, min_area: int, max_area_ratio: float) -> list[BBox]:
    import cv2  # type: ignore

    boxes: list[BBox] = []
    max_area = width * height * max_area_ratio
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box = BBox(x, y, w, h).clamp(width, height)
        if box.area < min_area or box.area > max_area:
            continue
        if box.w < 4 or box.h < 4:
            continue
        boxes.append(box)
    return boxes


def _dedupe_boxes(boxes: list[BBox]) -> list[BBox]:
    kept: list[BBox] = []
    for box in sorted(boxes, key=lambda item: item.area, reverse=True):
        if any(box.iou(other) > 0.75 or box.overlap_ratio(other) > 0.93 for other in kept):
            continue
        kept.append(box)
    return kept


def _parse_ocr_confidence(value) -> float:
    try:
        number = float(value)
    except Exception:
        return 0.0
    if number > 1:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def _style_to_attr(style: dict[str, Any] | None) -> str | None:
    if not style:
        return None
    parts: list[str] = []
    for key in ("fontFamily", "fontSize", "color", "align"):
        value = style.get(key)
        if value is not None and value != "":
            parts.append(f"{key}:{value}")
    return ";".join(parts) or None


def _alpha_component_candidates(path: Path, min_area: int, max_candidates: int) -> list[Candidate]:
    try:
        image = Image.open(path).convert("RGBA")
    except Exception:
        return []
    alpha = image.getchannel("A")
    if alpha.getextrema()[0] >= 250:
        return []
    width, height = image.size
    pixels = alpha.load()
    seen: set[tuple[int, int]] = set()
    boxes: list[BBox] = []
    for y, x in itertools.product(range(height), range(width)):
        if (x, y) in seen or pixels[x, y] <= 8:
            continue
        stack = [(x, y)]
        seen.add((x, y))
        min_x = max_x = x
        min_y = max_y = y
        while stack:
            px, py = stack.pop()
            min_x, max_x = min(min_x, px), max(max_x, px)
            min_y, max_y = min(min_y, py), max(max_y, py)
            for nx, ny in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height or (nx, ny) in seen:
                    continue
                seen.add((nx, ny))
                if pixels[nx, ny] > 8:
                    stack.append((nx, ny))
        box = BBox.from_xyxy(min_x, min_y, max_x + 1, max_y + 1)
        if box.area >= min_area:
            boxes.append(box)
    boxes = _dedupe_boxes(boxes)[:max_candidates]
    return [
        Candidate(
            id=f"v{index + 1}",
            bbox=box,
            source="visual-alpha",
            type_hint="Unknown",
            confidence=0.35,
            source_refs=[f"visual:{index + 1}"],
        )
        for index, box in enumerate(boxes)
    ]
