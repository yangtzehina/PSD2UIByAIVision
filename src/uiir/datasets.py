from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .models import BBox, UIIRDocument, UINode
from .xml_writer import write_json


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
HIERARCHY_SUFFIXES = (
    "_view_hierarchy",
    "-view-hierarchy",
    ".view_hierarchy",
    "_hierarchy",
    "-hierarchy",
    ".hierarchy",
    "_views",
    "-views",
)

ANDROID_CLASS_TYPES = {
    "textview": "Text",
    "button": "Button",
    "imagebutton": "Button",
    "edittext": "Input",
    "autocomplete": "Input",
    "textinputedittext": "Input",
    "checkbox": "Toggle",
    "checkedtextview": "Toggle",
    "radiobutton": "Toggle",
    "switch": "Toggle",
    "togglebutton": "Toggle",
    "seekbar": "Slider",
    "ratingbar": "Slider",
    "imageview": "Image",
    "toolbar": "Container",
    "actionbar": "Container",
    "linearlayout": "Container",
    "framelayout": "Container",
    "relativelayout": "Container",
    "constraintlayout": "Container",
    "coordinatorlayout": "Container",
    "viewgroup": "Container",
    "recyclerview": "List",
    "listview": "List",
    "expandablelistview": "List",
    "gridview": "Grid",
    "scrollview": "ScrollView",
    "horizontalscrollview": "ScrollView",
    "nestedscrollview": "ScrollView",
    "viewpager": "ScrollView",
}


def import_rico_dataset(input_dir: str | Path, output_dir: str | Path, limit: int | None = None) -> dict[str, Any]:
    """Convert local Rico-style screenshots and view hierarchies into UIIR goldens.

    The importer only reads files already present under ``input_dir``. It pairs JSON
    hierarchy files with screenshots by stem, writes UIIR JSON under
    ``output_dir/goldens/<sample>/uiir.json``, and records skipped hierarchies in a
    root manifest.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be None or a non-negative integer")

    source_root = Path(input_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()
    golden_root = out_root / "goldens"
    out_root.mkdir(parents=True, exist_ok=True)
    golden_root.mkdir(parents=True, exist_ok=True)

    images = _index_images(source_root)
    samples: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    used_names: set[str] = set()

    for hierarchy_path in _hierarchy_paths(source_root):
        if limit is not None and len(samples) >= limit:
            break

        screenshot_path = _matching_screenshot(hierarchy_path, images)
        if screenshot_path is None:
            skipped.append(
                {
                    "hierarchy": _relative_or_absolute(hierarchy_path, source_root),
                    "reason": "missing_screenshot",
                }
            )
            continue

        try:
            hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
            root_view = _find_hierarchy_root(hierarchy)
            if root_view is None:
                skipped.append(
                    {
                        "hierarchy": _relative_or_absolute(hierarchy_path, source_root),
                        "reason": "not_view_hierarchy",
                    }
                )
                continue
            with Image.open(screenshot_path) as image:
                width, height = image.size
        except Exception as exc:
            skipped.append(
                {
                    "hierarchy": _relative_or_absolute(hierarchy_path, source_root),
                    "reason": f"read_failed: {exc}",
                }
            )
            continue

        sample_name = _sample_name(hierarchy_path, screenshot_path, source_root, used_names)
        sample_dir = golden_root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        local_screenshot = _copy_screenshot(screenshot_path, sample_dir)
        document = _document_from_hierarchy(
            root_view,
            hierarchy_path=hierarchy_path,
            screenshot_path=screenshot_path,
            local_screenshot=local_screenshot,
            source_root=source_root,
            width=width,
            height=height,
        )
        uiir_path = write_json(document, sample_dir / "uiir.json")
        node_count = _count_nodes(document.root)
        samples.append(
            {
                "name": sample_name,
                "hierarchy": _relative_or_absolute(hierarchy_path, source_root),
                "screenshot": _relative_or_absolute(screenshot_path, source_root),
                "copied_screenshot": _relative_or_absolute(local_screenshot, out_root),
                "uiir_json": _relative_or_absolute(uiir_path, out_root),
                "width": width,
                "height": height,
                "node_count": node_count,
            }
        )

    manifest = {
        "version": "0.1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": "rico-local",
        "input_dir": source_root.as_posix(),
        "output_dir": out_root.as_posix(),
        "golden_root": golden_root.as_posix(),
        "limit": limit,
        "count": len(samples),
        "skipped_count": len(skipped),
        "samples": samples,
        "skipped": skipped,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


@dataclass
class _NodeBuilder:
    width: int
    height: int
    index: int = 1
    path: list[int] = field(default_factory=list)

    def next_id(self) -> str:
        self.index += 1
        return f"n{self.index}"

    def convert_children(self, node: dict[str, Any]) -> list[UINode]:
        converted: list[UINode] = []
        for child_index, child in enumerate(_children(node)):
            if not isinstance(child, dict):
                continue
            self.path.append(child_index)
            converted.extend(self.convert_node_or_promote(child))
            self.path.pop()
        return converted

    def convert_node_or_promote(self, node: dict[str, Any]) -> list[UINode]:
        if _is_invisible(node):
            return []

        bbox = _bbox_from_node(node)
        children = self.convert_children(node)
        if bbox is None:
            return children

        bbox = bbox.clamp(self.width, self.height)
        if bbox.is_empty:
            return children

        node_type = map_android_class_to_uiir_type(_class_name(node), node)
        ui_node = UINode(
            id=self.next_id(),
            type=node_type,
            bbox=bbox,
            confidence=0.95,
            source_refs=[f"rico:{'.'.join(str(part) for part in self.path) or 'root'}"],
            role=_role(node, node_type),
            text=_node_text(node),
            layout=_layout(node),
            interaction=_interaction(node, node_type),
            metadata=_metadata(node),
            children=children,
        )
        return [ui_node]


def map_android_class_to_uiir_type(class_name: str | None, node: dict[str, Any] | None = None) -> str:
    """Map a common Android view class name to the closest UIIR node type."""

    normalized = _class_tail(class_name).lower()
    node = node or {}
    if _bool_value(node.get("scrollable")):
        return "ScrollView"
    if _bool_value(node.get("checkable")) or _bool_value(node.get("checked")):
        return "Toggle"
    if _bool_value(node.get("editable")) or _has_truthy_key(node, ("isEditable", "password")):
        return "Input"
    if normalized in ANDROID_CLASS_TYPES:
        return ANDROID_CLASS_TYPES[normalized]
    if "recyclerview" in normalized or "listview" in normalized:
        return "List"
    if "grid" in normalized:
        return "Grid"
    if "scroll" in normalized or "viewpager" in normalized:
        return "ScrollView"
    if "imagebutton" in normalized:
        return "Button"
    if "button" in normalized:
        return "Button"
    if "edittext" in normalized or "input" in normalized:
        return "Input"
    if "switch" in normalized or "checkbox" in normalized or "radio" in normalized:
        return "Toggle"
    if "seekbar" in normalized or "slider" in normalized:
        return "Slider"
    if "text" in normalized:
        return "Text"
    if "image" in normalized:
        return "Image"
    if normalized in {"view", "space"}:
        return "Background"
    if normalized:
        return "Container"
    return "Unknown"


def _document_from_hierarchy(
    root_view: dict[str, Any],
    *,
    hierarchy_path: Path,
    screenshot_path: Path,
    local_screenshot: Path,
    source_root: Path,
    width: int,
    height: int,
) -> UIIRDocument:
    builder = _NodeBuilder(width=width, height=height)
    root = UINode(
        id="n1",
        type="Screen",
        bbox=BBox(0, 0, width, height),
        confidence=1,
        source_refs=["rico:screenshot", "rico:hierarchy"],
        metadata={
            "source": "rico-local",
            "hierarchy": _relative_or_absolute(hierarchy_path, source_root),
            "screenshot": _relative_or_absolute(screenshot_path, source_root),
        },
        children=builder.convert_node_or_promote(root_view),
    )
    return UIIRDocument(
        version="0.1",
        source=local_screenshot.name,
        width=width,
        height=height,
        assets_root="assets/",
        root=root,
        metadata={
            "documentKind": "screen",
            "dataset": {
                "name": "rico",
                "importer": "import_rico_dataset",
                "hierarchy": _relative_or_absolute(hierarchy_path, source_root),
                "screenshot": _relative_or_absolute(screenshot_path, source_root),
                "localScreenshot": local_screenshot.name,
            },
        },
    )


def _hierarchy_paths(root: Path) -> list[Path]:
    return sorted((path for path in root.rglob("*.json") if path.is_file()), key=lambda path: path.as_posix().lower())


def _index_images(root: Path) -> dict[str, list[Path]]:
    images: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.setdefault(path.stem.lower(), []).append(path)
    return images


def _matching_screenshot(hierarchy_path: Path, images: dict[str, list[Path]]) -> Path | None:
    for stem in _candidate_image_stems(hierarchy_path.stem):
        candidates = images.get(stem.lower())
        if candidates:
            return sorted(candidates, key=lambda candidate: _image_match_score(hierarchy_path, candidate))[0]
    return None


def _candidate_image_stems(stem: str) -> list[str]:
    stems = [stem]
    lowered = stem.lower()
    for suffix in HIERARCHY_SUFFIXES:
        if lowered.endswith(suffix):
            stems.append(stem[: -len(suffix)])
    return list(dict.fromkeys(item for item in stems if item))


def _image_match_score(hierarchy_path: Path, image_path: Path) -> tuple[int, int, str]:
    try:
        common = len(set(hierarchy_path.parent.parts) & set(image_path.parent.parts))
    except Exception:
        common = 0
    same_directory = hierarchy_path.parent == image_path.parent
    return (0 if same_directory else 1, -common, image_path.as_posix().lower())


def _find_hierarchy_root(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    if _looks_like_view_node(data):
        return data
    for key in ("root", "activity", "view_hierarchy", "viewHierarchy", "hierarchy", "tree"):
        value = data.get(key)
        if isinstance(value, dict):
            found = _find_hierarchy_root(value)
            if found is not None:
                return found
    return None


def _looks_like_view_node(value: dict[str, Any]) -> bool:
    return bool(_class_name(value) or _bbox_from_node(value) or _children(value))


def _copy_screenshot(screenshot_path: Path, sample_dir: Path) -> Path:
    suffix = screenshot_path.suffix.lower()
    target = sample_dir / f"screenshot{suffix if suffix in IMAGE_EXTENSIONS else '.png'}"
    shutil.copyfile(screenshot_path, target)
    return target


def _sample_name(hierarchy_path: Path, screenshot_path: Path, source_root: Path, used_names: set[str]) -> str:
    base = _safe_name(screenshot_path.stem or _canonical_hierarchy_stem(hierarchy_path.stem))
    name = base
    if name in used_names:
        digest = hashlib.sha1(_relative_or_absolute(hierarchy_path, source_root).encode("utf-8")).hexdigest()[:8]
        name = f"{base}-{digest}"
    used_names.add(name)
    return name


def _canonical_hierarchy_stem(stem: str) -> str:
    lowered = stem.lower()
    for suffix in HIERARCHY_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip(".-")
    return safe or "sample"


def _class_name(node: dict[str, Any]) -> str | None:
    for key in ("class", "className", "class_name", "type"):
        value = node.get(key)
        if value:
            return str(value)
    return None


def _class_tail(class_name: str | None) -> str:
    if not class_name:
        return ""
    tail = str(class_name).split("$")[-1].split(".")[-1]
    return re.sub(r"[^A-Za-z0-9_]+", "", tail)


def _children(node: dict[str, Any]) -> list[Any]:
    for key in ("children", "child", "nodes"):
        value = node.get(key)
        if isinstance(value, list):
            return value
    return []


def _bbox_from_node(node: dict[str, Any]) -> BBox | None:
    for key in ("bounds", "visibleBounds", "screenBounds", "boundsInScreen", "bbox"):
        if key in node:
            bbox = _bbox_from_any(node.get(key))
            if bbox is not None:
                return bbox
    return None


def _bbox_from_any(value: Any) -> BBox | None:
    if isinstance(value, BBox):
        return value
    if isinstance(value, dict):
        if all(key in value for key in ("x", "y", "w", "h")):
            return BBox(int(value["x"]), int(value["y"]), int(value["w"]), int(value["h"]))
        if all(key in value for key in ("x", "y", "width", "height")):
            return BBox(int(value["x"]), int(value["y"]), int(value["width"]), int(value["height"]))
        if all(key in value for key in ("left", "top", "right", "bottom")):
            return BBox.from_xyxy(value["left"], value["top"], value["right"], value["bottom"])
    if isinstance(value, str):
        numbers = [int(match) for match in re.findall(r"-?\d+", value)]
        if len(numbers) >= 4:
            return BBox.from_xyxy(numbers[0], numbers[1], numbers[2], numbers[3])
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        left, top, right, bottom = value[:4]
        return BBox.from_xyxy(left, top, right, bottom)
    return None


def _is_invisible(node: dict[str, Any]) -> bool:
    for key in ("visible", "visible-to-user", "visibleToUser", "shown"):
        if key in node and not _bool_value(node.get(key)):
            return True
    return False


def _node_text(node: dict[str, Any]) -> str | None:
    for key in ("text", "label", "content-desc", "contentDescription", "description", "hint"):
        value = node.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _role(node: dict[str, Any], node_type: str) -> str | None:
    if _bool_value(node.get("clickable")):
        return "button" if node_type == "Button" else "action"
    if _bool_value(node.get("focusable")):
        return "focusable"
    if _bool_value(node.get("checkable")):
        return "toggle"
    return None


def _layout(node: dict[str, Any]) -> str | None:
    orientation = node.get("orientation")
    if orientation is None:
        return None
    if isinstance(orientation, str):
        return orientation.strip().lower() or None
    return str(orientation)


def _interaction(node: dict[str, Any], node_type: str) -> str | None:
    if node_type == "Input":
        return "input"
    if node_type == "Toggle":
        return "toggle"
    if node_type == "ScrollView" or _bool_value(node.get("scrollable")):
        return "scroll"
    if _bool_value(node.get("clickable")):
        return "tap"
    return None


def _metadata(node: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source_key, target_key in (
        ("class", "androidClass"),
        ("className", "androidClass"),
        ("resource-id", "resourceId"),
        ("resourceId", "resourceId"),
        ("package", "package"),
        ("content-desc", "contentDescription"),
        ("contentDescription", "contentDescription"),
        ("clickable", "clickable"),
        ("enabled", "enabled"),
        ("focusable", "focusable"),
        ("scrollable", "scrollable"),
        ("checkable", "checkable"),
        ("checked", "checked"),
    ):
        if source_key in node and target_key not in metadata:
            metadata[target_key] = node[source_key]
    return metadata


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _has_truthy_key(node: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(_bool_value(node.get(key)) for key in keys)


def _count_nodes(node: UINode) -> int:
    return 1 + sum(_count_nodes(child) for child in node.children)


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
