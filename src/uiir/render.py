from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def render_uiir_preview(uiir_json: str | Path, output_path: str | Path) -> Path:
    uiir_path = Path(uiir_json)
    data = json.loads(uiir_path.read_text(encoding="utf-8"))
    width = int(data["width"])
    height = int(data["height"])
    base_dir = uiir_path.parent
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas, "RGBA")
    _render_node(data.get("root", {}), canvas, draw, base_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def _render_node(node: dict[str, Any], canvas: Image.Image, draw: ImageDraw.ImageDraw, base_dir: Path) -> None:
    _render_single_node(node, canvas, draw, base_dir)
    for child in node.get("children", []) or []:
        _render_node(child, canvas, draw, base_dir)


def _render_single_node(node: dict[str, Any], canvas: Image.Image, draw: ImageDraw.ImageDraw, base_dir: Path) -> None:
    if node.get("type") == "Screen" or node.get("metadata", {}).get("component"):
        return
    bbox = node.get("bbox") or {}
    try:
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
    except Exception:
        return
    if w <= 0 or h <= 0:
        return

    asset = node.get("asset")
    if asset:
        asset_path = (base_dir / asset).resolve()
        if asset_path.exists():
            try:
                image = Image.open(asset_path).convert("RGBA")
                if image.size != (w, h):
                    image = image.resize((w, h), Image.Resampling.LANCZOS)
                canvas.alpha_composite(image, (x, y))
                return
            except Exception:
                pass

    color = _color_for_type(str(node.get("type") or "Unknown"))
    draw.rectangle((x, y, x + w, y + h), outline=color, fill=(*color[:3], 28), width=2)


def _color_for_type(node_type: str) -> tuple[int, int, int, int]:
    colors = {
        "Container": (239, 68, 68, 180),
        "Image": (245, 158, 11, 180),
        "Icon": (139, 92, 246, 180),
        "Text": (37, 99, 235, 180),
        "Button": (20, 184, 166, 180),
        "Input": (6, 182, 212, 180),
        "Toggle": (132, 204, 22, 180),
        "Slider": (234, 179, 8, 180),
        "ScrollView": (34, 197, 94, 180),
        "List": (34, 197, 94, 180),
        "Grid": (34, 197, 94, 180),
        "Background": (100, 116, 139, 180),
    }
    return colors.get(node_type, (249, 115, 22, 180))
