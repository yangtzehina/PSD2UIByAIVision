from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


NODE_TYPES = (
    "Screen",
    "Container",
    "Image",
    "Icon",
    "Text",
    "Button",
    "Input",
    "Toggle",
    "Slider",
    "ScrollView",
    "List",
    "Grid",
    "Background",
    "Unknown",
)

PARENT_NODE_TYPES = {"Screen", "Container", "Button", "Input", "Toggle", "Slider", "ScrollView", "List", "Grid"}


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_xyxy(cls, left: int | float, top: int | float, right: int | float, bottom: int | float) -> "BBox":
        x = int(round(left))
        y = int(round(top))
        r = int(round(right))
        b = int(round(bottom))
        return cls(x=x, y=y, w=max(0, r - x), h=max(0, b - y))

    @classmethod
    def from_any(cls, value: "BBox | dict[str, Any] | list[int] | tuple[int, ...]") -> "BBox":
        if isinstance(value, BBox):
            return value
        if isinstance(value, dict):
            return cls(int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"]))
        if len(value) == 4:
            x, y, w, h = value
            return cls(int(x), int(y), int(w), int(h))
        raise ValueError(f"Cannot coerce bbox from {value!r}")

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0

    def to_attr(self) -> str:
        return f"{self.x},{self.y},{self.w},{self.h}"

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def contains_point(self, x: float, y: float) -> bool:
        return self.x <= x <= self.right and self.y <= y <= self.bottom

    def contains_bbox(self, other: "BBox", padding: int = 0) -> bool:
        return (
            self.x - padding <= other.x
            and self.y - padding <= other.y
            and self.right + padding >= other.right
            and self.bottom + padding >= other.bottom
        )

    def intersection(self, other: "BBox") -> int:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.right, other.right)
        y2 = min(self.bottom, other.bottom)
        return max(0, x2 - x1) * max(0, y2 - y1)

    def iou(self, other: "BBox") -> float:
        inter = self.intersection(other)
        union = self.area + other.area - inter
        if union <= 0:
            return 0.0
        return inter / union

    def overlap_ratio(self, other: "BBox") -> float:
        smaller = min(self.area, other.area)
        if smaller <= 0:
            return 0.0
        return self.intersection(other) / smaller

    def clamp(self, width: int, height: int) -> "BBox":
        x = max(0, min(self.x, width))
        y = max(0, min(self.y, height))
        right = max(x, min(self.right, width))
        bottom = max(y, min(self.bottom, height))
        return BBox.from_xyxy(x, y, right, bottom)


@dataclass
class LayerRecord:
    id: str
    name: str
    path: str
    kind: str
    bbox: BBox
    visible: bool = True
    opacity: float = 1.0
    is_group: bool = False
    parent_id: str | None = None
    depth: int = 0
    blend_mode: str | None = None
    text: str | None = None
    style: dict[str, Any] = field(default_factory=dict)
    asset: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data


@dataclass
class Candidate:
    id: str
    bbox: BBox
    source: str
    type_hint: str = "Unknown"
    confidence: float = 0.5
    source_refs: list[str] = field(default_factory=list)
    name: str | None = None
    text: str | None = None
    style: str | None = None
    role: str | None = None
    layout: str | None = None
    asset: str | None = None
    parent_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_type(self) -> str:
        return self.type_hint if self.type_hint in NODE_TYPES else "Unknown"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        copied = dict(data)
        copied["bbox"] = BBox.from_any(copied["bbox"])
        return cls(**copied)


@dataclass
class UINode:
    id: str
    type: str
    bbox: BBox
    confidence: float
    source_refs: list[str]
    role: str | None = None
    text: str | None = None
    style: str | None = None
    layout: str | None = None
    asset: str | None = None
    interaction: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list["UINode"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in NODE_TYPES:
            self.type = "Unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "bbox": self.bbox.to_dict(),
            "confidence": round(float(self.confidence), 4),
            "sourceRefs": list(self.source_refs),
            "role": self.role,
            "text": self.text,
            "style": self.style,
            "layout": self.layout,
            "asset": self.asset,
            "interaction": self.interaction,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class UIIRDocument:
    version: str
    source: str
    width: int
    height: int
    assets_root: str
    root: UINode
    candidates: list[Candidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "assetsRoot": self.assets_root,
            "root": self.root.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": self.metadata,
        }


def relpath(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
