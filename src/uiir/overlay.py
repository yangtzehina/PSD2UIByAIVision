from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import Candidate


PALETTE = {
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


def draw_overlay(image_path: str | Path, candidates: list[Candidate], output_path: str | Path, max_labels: int = 220) -> Path:
    image = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = _load_font()
    width, height = image.size

    for candidate in candidates[:max_labels]:
        color = PALETTE.get(candidate.type_hint, PALETTE["Unknown"])
        box = candidate.bbox.clamp(width, height)
        if box.is_empty:
            continue
        draw.rectangle([box.x, box.y, box.right, box.bottom], outline=color, width=2)
        label = f"{candidate.id}:{candidate.type_hint}"
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_w = label_bbox[2] - label_bbox[0] + 8
        label_h = label_bbox[3] - label_bbox[1] + 6
        label_x0 = max(0, min(box.x, width - 1))
        label_x1 = max(label_x0 + 1, min(width, label_x0 + label_w))
        if box.y >= label_h:
            label_y0 = box.y - label_h
            label_y1 = box.y
        else:
            label_y0 = min(height - 1, box.bottom)
            label_y1 = min(height, label_y0 + label_h)
        if label_y1 > label_y0:
            draw.rectangle([label_x0, label_y0, label_x1, label_y1], fill=color)
            draw.text((label_x0 + 4, label_y0 + 2), label, fill="white", font=font)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", 12)
    except Exception:
        return ImageFont.load_default()
