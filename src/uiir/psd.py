from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from .heuristics import is_comment_layer
from .models import BBox, LayerRecord, relpath


@dataclass
class PSDExtractResult:
    source: Path
    width: int
    height: int
    composite_path: Path
    assets_root: Path
    layers: list[LayerRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_metadata(self, base: Path) -> dict[str, Any]:
        return {
            "source": self.source.name,
            "width": self.width,
            "height": self.height,
            "composite": relpath(self.composite_path, base),
            "assetsRoot": relpath(self.assets_root, base),
            "warnings": self.warnings,
            "layers": [layer.to_dict() for layer in self.layers],
        }


def extract_psd(psd_path: str | Path, output_dir: str | Path) -> PSDExtractResult:
    try:
        from psd_tools import PSDImage
    except ImportError as exc:
        raise RuntimeError("psd-tools is required. Install with: python3 -m pip install -e .") from exc

    source = Path(psd_path).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_root = out_dir / "assets"
    layer_asset_root = assets_root / "layers"
    layer_asset_root.mkdir(parents=True, exist_ok=True)

    psd = PSDImage.open(source)
    width = int(getattr(psd, "width"))
    height = int(getattr(psd, "height"))
    composite_path = out_dir / "composite.png"
    warnings: list[str] = []

    try:
        composite = psd.composite()
        if composite is None:
            raise ValueError("PSD composite returned no image")
        composite.convert("RGBA").save(composite_path)
    except Exception as exc:
        warnings.append(f"Could not render PSD composite: {exc}")
        Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(composite_path)

    layers: list[LayerRecord] = []
    counters = {"value": 0}
    for child in _iter_children(psd):
        _walk_layer(
            child,
            parent_id=None,
            depth=0,
            path_parts=[],
            records=layers,
            counters=counters,
            layer_asset_root=layer_asset_root,
            output_dir=out_dir,
        )

    result = PSDExtractResult(
        source=source,
        width=width,
        height=height,
        composite_path=composite_path,
        assets_root=assets_root,
        layers=layers,
        warnings=warnings,
    )
    metadata_path = out_dir / "layer_metadata.json"
    metadata_path.write_text(json.dumps(result.to_metadata(out_dir), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _iter_children(layer: Any) -> Iterable[Any]:
    try:
        yield from layer
    except TypeError:
        return


def _walk_layer(
    layer: Any,
    parent_id: str | None,
    depth: int,
    path_parts: list[str],
    records: list[LayerRecord],
    counters: dict[str, int],
    layer_asset_root: Path,
    output_dir: Path,
) -> None:
    counters["value"] += 1
    layer_id = f"layer:{counters['value']}"
    name = str(getattr(layer, "name", "") or f"Layer {counters['value']}")
    path = "/".join([*path_parts, name])
    kind = str(getattr(layer, "kind", "") or "")
    is_group = _is_group(layer)
    visible = bool(getattr(layer, "visible", True))
    opacity = _normalize_opacity(getattr(layer, "opacity", 255))
    bbox = _extract_bbox(layer)
    warnings: list[str] = []
    asset: str | None = None

    if visible and not is_group and not bbox.is_empty and not is_comment_layer(name):
        asset = _export_layer_image(layer, layer_id, layer_asset_root, output_dir, warnings)

    record = LayerRecord(
        id=layer_id,
        name=name,
        path=path,
        kind=kind,
        bbox=bbox,
        visible=visible,
        opacity=opacity,
        is_group=is_group,
        parent_id=parent_id,
        depth=depth,
        blend_mode=str(getattr(layer, "blend_mode", "") or "") or None,
        text=_extract_text(layer),
        style=_extract_text_style(layer),
        asset=asset,
        warnings=warnings,
    )
    records.append(record)

    for child in _iter_children(layer):
        _walk_layer(
            child,
            parent_id=layer_id,
            depth=depth + 1,
            path_parts=[*path_parts, name],
            records=records,
            counters=counters,
            layer_asset_root=layer_asset_root,
            output_dir=output_dir,
        )


def _is_group(layer: Any) -> bool:
    checker = getattr(layer, "is_group", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    return bool(checker)


def _normalize_opacity(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return 1.0
    if number > 1:
        number = number / 255.0
    return max(0.0, min(1.0, number))


def _extract_bbox(layer: Any) -> BBox:
    left = getattr(layer, "left", None)
    top = getattr(layer, "top", None)
    right = getattr(layer, "right", None)
    bottom = getattr(layer, "bottom", None)
    if None not in (left, top, right, bottom):
        return BBox.from_xyxy(left, top, right, bottom)

    bbox = getattr(layer, "bbox", None)
    if bbox is None:
        return BBox(0, 0, 0, 0)
    for names in (("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")):
        values = [getattr(bbox, name, None) for name in names]
        if None not in values:
            return BBox.from_xyxy(*values)
    try:
        values = list(bbox)
        if len(values) == 4:
            return BBox.from_xyxy(*values)
    except Exception:
        pass
    return BBox(0, 0, 0, 0)


def _extract_text(layer: Any) -> str | None:
    for attr in ("text", "string"):
        value = getattr(layer, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    engine_dict = getattr(layer, "engine_dict", None)
    if engine_dict:
        try:
            text = str(engine_dict.get("Editor", {}).get("Text", "")).strip()
            return text or None
        except Exception:
            return None
    return None


def _extract_text_style(layer: Any) -> dict[str, Any]:
    engine_dict = getattr(layer, "engine_dict", None)
    if not engine_dict:
        return {}
    data = _to_plain(engine_dict)
    style: dict[str, Any] = {}
    _first_number(data, {"FontSize", "FntSz", "ImpliedFontSize"}, style, "fontSize")
    _first_string(data, {"Font", "FontName", "Name"}, style, "fontFamily")
    _first_string(data, {"Justification", "Justify", "Alignment"}, style, "align")
    color = _first_color(data)
    if color:
        style["color"] = color
    return style


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    for attr in ("to_dict", "items"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                converted = dict(method()) if attr == "items" else method()
                return _to_plain(converted)
            except Exception:
                pass
    return value


def _walk_plain(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_plain(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_plain(item)


def _first_number(data: Any, keys: set[str], style: dict[str, Any], target: str) -> None:
    for obj in _walk_plain(data):
        for key, value in obj.items():
            if key in keys:
                try:
                    style[target] = round(float(value), 2)
                    return
                except Exception:
                    continue


def _first_string(data: Any, keys: set[str], style: dict[str, Any], target: str) -> None:
    for obj in _walk_plain(data):
        for key, value in obj.items():
            if key in keys and isinstance(value, str) and value.strip():
                style[target] = value.strip()
                return


def _first_color(data: Any) -> str | None:
    for obj in _walk_plain(data):
        for key, value in obj.items():
            if "color" not in key.lower():
                continue
            channels = value
            if isinstance(value, dict):
                channels = value.get("Values") or value.get("values") or value.get("RGB")
            if not isinstance(channels, (list, tuple)) or len(channels) < 3:
                continue
            try:
                rgb = [int(max(0, min(255, round(float(item))))) for item in channels[:3]]
            except Exception:
                continue
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    return None


def _export_layer_image(layer: Any, layer_id: str, layer_asset_root: Path, output_dir: Path, warnings: list[str]) -> str | None:
    try:
        image = layer.composite()
        if image is None:
            warnings.append("Layer composite returned no image")
            return None
        safe = _safe_filename(layer_id.replace(":", "_"))
        path = layer_asset_root / f"{safe}.png"
        image.convert("RGBA").save(path)
        return relpath(path, output_dir)
    except Exception as exc:
        warnings.append(f"Could not export layer image: {exc}")
        return None


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value.strip("._") or "layer"
