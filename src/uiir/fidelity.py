from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any


REPORT_VERSION = "0.1"
LAYER_METADATA = "layer_metadata.json"
UIIR_JSON = "uiir.json"
CANDIDATES_JSON = "candidates.json"
REPORT_JSON = "parser_fidelity.json"

_EFFECT_COLLECTION_KEYS = {"effects", "layereffects", "layereffect", "layerfx", "fx"}
_EFFECT_SINGLE_KEYS = {
    "bevel",
    "coloroverlay",
    "dropshadow",
    "gradientoverlay",
    "innerglow",
    "innershadow",
    "outerglow",
    "patternoverlay",
    "satin",
    "stroke",
}
_SMART_OBJECT_KEYS = {
    "issmartobject",
    "smartobject",
    "smartobjectlayer",
    "smartobjectlinked",
    "smartobjectembedded",
    "placedlayer",
}


def build_parser_fidelity_report(
    extract_output: str | os.PathLike[str] | Any,
    output_dir: str | os.PathLike[str] | None = None,
    probe_photoshopapi: bool = False,
) -> dict[str, Any]:
    """Build a parser fidelity summary for an extract output directory.

    The function reads layer_metadata.json, uiir.json, and candidates.json from
    the extract output, writes parser_fidelity.json, and returns the same data.
    If ``probe_photoshopapi`` is enabled, the probe only checks whether a module
    can be found and records no import errors, paths, or module origins.
    """

    input_dir = _extract_output_dir(extract_output)
    report_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else input_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "layer_metadata": _artifact_path(extract_output, input_dir, "layers_json", LAYER_METADATA),
        "uiir": _artifact_path(extract_output, input_dir, "uiir_json", UIIR_JSON),
        "candidates": _artifact_path(extract_output, input_dir, "candidates_json", CANDIDATES_JSON),
    }
    missing_inputs = [name for name, path in paths.items() if not path.exists()]

    layer_metadata = _load_json(paths["layer_metadata"], default={})
    uiir = _load_json(paths["uiir"], default={})
    candidates_data = _load_json(paths["candidates"], default=[])

    layers = _items(layer_metadata.get("layers") if isinstance(layer_metadata, dict) else [])
    candidates = _items(candidates_data.get("items") if isinstance(candidates_data, dict) else candidates_data)
    uiir_nodes = _flatten_nodes(uiir.get("root") if isinstance(uiir, dict) else None)

    layer_ids = {str(layer.get("id")) for layer in layers if layer.get("id") is not None}
    candidate_layer_refs = _layer_refs_from_candidates(candidates)
    uiir_layer_refs = _layer_refs_from_nodes(uiir_nodes)
    candidate_known_refs = candidate_layer_refs & layer_ids if layer_ids else set(candidate_layer_refs)
    uiir_known_refs = uiir_layer_refs & layer_ids if layer_ids else set(uiir_layer_refs)

    text_layers = [layer for layer in layers if _is_text_layer(layer)]
    styled_text_layers = [layer for layer in text_layers if _has_style(layer)]
    group_layers = [layer for layer in layers if _is_group_layer(layer)]
    smart_object_layers = [layer for layer in layers if _is_smart_object_ish(layer)]
    visible_layers = [layer for layer in layers if _value_for_keys(layer, "visible") is not False]
    nonempty_bbox_layers = [layer for layer in layers if _has_nonempty_bbox(layer)]
    rasterizable_layers = [
        layer
        for layer in layers
        if _value_for_keys(layer, "visible") is not False
        and not _is_group_layer(layer)
        and _has_nonempty_bbox(layer)
    ]

    layer_asset_refs = _asset_refs_from_layers(layers)
    candidate_asset_refs = _asset_refs_from_items(candidates)
    uiir_asset_refs = _asset_refs_from_items(uiir_nodes)
    unique_asset_refs = layer_asset_refs | candidate_asset_refs | uiir_asset_refs

    layer_effect_counts = [_layer_effect_count(layer) for layer in layers]
    layers_with_effects = sum(1 for count in layer_effect_counts if count > 0)
    effect_entries = sum(layer_effect_counts)

    top_level_warnings = _warning_count(layer_metadata.get("warnings") if isinstance(layer_metadata, dict) else None)
    layer_warning_counts = [_warning_count(_value_for_keys(layer, "warnings", "warning")) for layer in layers]
    layers_with_warnings = sum(1 for count in layer_warning_counts if count > 0)
    layer_warnings = sum(layer_warning_counts)
    total_warnings = top_level_warnings + layer_warnings

    counts = {
        "layers": len(layers),
        "visible_layers": len(visible_layers),
        "hidden_layers": max(0, len(layers) - len(visible_layers)),
        "groups": len(group_layers),
        "text_layers": len(text_layers),
        "styled_text_layers": len(styled_text_layers),
        "smart_object_ish_layers": len(smart_object_layers),
        "smart_object_like_layers": len(smart_object_layers),
        "assets": len(unique_asset_refs),
        "layer_assets": len(layer_asset_refs),
        "candidate_assets": len(candidate_asset_refs),
        "uiir_assets": len(uiir_asset_refs),
        "layer_effect_layers": layers_with_effects,
        "layer_effects": effect_entries,
        "warning_layers": layers_with_warnings,
        "warnings": total_warnings,
        "candidates": len(candidates),
        "uiir_nodes": len(uiir_nodes),
        "uiir_non_screen_nodes": sum(1 for node in uiir_nodes if node.get("type") != "Screen"),
    }

    report_path = report_dir / REPORT_JSON
    report: dict[str, Any] = {
        "version": REPORT_VERSION,
        "extract_output": input_dir.as_posix(),
        "output_dir": report_dir.as_posix(),
        "paths": {
            "layer_metadata": paths["layer_metadata"].as_posix(),
            "uiir": paths["uiir"].as_posix(),
            "candidates": paths["candidates"].as_posix(),
            "report": report_path.as_posix(),
        },
        "missing_inputs": missing_inputs,
        "counts": counts,
        "layer_effects": {
            "layers": layers_with_effects,
            "entries": effect_entries,
        },
        "warnings": {
            "top_level": top_level_warnings,
            "layer_level": layer_warnings,
            "layers": layers_with_warnings,
            "total": total_warnings,
        },
        "psd_tools_coverage": {
            "coverage_basis": "layer_metadata.layers",
            "layer_id_count": len(layer_ids),
            "layers_with_bbox": sum(1 for layer in layers if isinstance(layer.get("bbox"), dict)),
            "nonempty_bbox_layers": len(nonempty_bbox_layers),
            "rasterizable_layers": len(rasterizable_layers),
            "layers_with_assets": len(layer_asset_refs),
            "candidate_layer_refs": len(candidate_layer_refs),
            "candidate_known_layer_refs": len(candidate_known_refs),
            "uiir_layer_refs": len(uiir_layer_refs),
            "uiir_known_layer_refs": len(uiir_known_refs),
            "candidate_layer_coverage": _ratio(len(candidate_known_refs), len(layers)),
            "uiir_layer_coverage": _ratio(len(uiir_known_refs), len(layers)),
            "asset_coverage": _ratio(len(layer_asset_refs), len(rasterizable_layers)),
            "text_style_coverage": _ratio(len(styled_text_layers), len(text_layers)),
        },
    }
    if probe_photoshopapi:
        report["photoshopapi_probe"] = _probe_photoshopapi()

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _extract_output_dir(extract_output: str | os.PathLike[str] | Any) -> Path:
    value = _attr_or_key(extract_output, "output_dir")
    if value is None:
        for attr in ("layers_json", "uiir_json", "candidates_json"):
            artifact = _attr_or_key(extract_output, attr)
            if artifact is not None:
                return Path(artifact).expanduser().resolve().parent
        if not isinstance(extract_output, (str, os.PathLike)):
            raise TypeError("extract_output must be an output directory, artifact path, or object with output_dir/artifact paths.")
        value = extract_output
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path.parent
    return path


def _artifact_path(extract_output: Any, input_dir: Path, attr: str, fallback_name: str) -> Path:
    value = _attr_or_key(extract_output, attr)
    if value is None:
        return input_dir / fallback_name
    return Path(value).expanduser().resolve()


def _attr_or_key(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _flatten_nodes(root: Any) -> list[dict[str, Any]]:
    if not isinstance(root, dict):
        return []
    nodes = [root]
    for child in root.get("children", []) or []:
        nodes.extend(_flatten_nodes(child))
    return nodes


def _layer_refs_from_candidates(candidates: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for candidate in candidates:
        refs.update(_layer_refs(candidate.get("source_refs")))
        refs.update(_layer_refs(candidate.get("sourceRefs")))
        metadata = candidate.get("metadata")
        if isinstance(metadata, dict):
            refs.update(_layer_refs(metadata.get("source_refs")))
            refs.update(_layer_refs(metadata.get("sourceRefs")))
    return refs


def _layer_refs_from_nodes(nodes: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for node in nodes:
        refs.update(_layer_refs(node.get("sourceRefs")))
        refs.update(_layer_refs(node.get("source_refs")))
    return refs


def _layer_refs(value: Any) -> set[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return {str(item) for item in values if isinstance(item, str) and item.startswith("layer:")}


def _asset_refs_from_layers(layers: list[dict[str, Any]]) -> set[str]:
    return _asset_refs_from_items(layers)


def _asset_refs_from_items(items: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for item in items:
        asset = item.get("asset")
        if isinstance(asset, str) and asset.strip():
            refs.add(asset.strip())
    return refs


def _is_text_layer(layer: dict[str, Any]) -> bool:
    text = _value_for_keys(layer, "text", "string")
    if isinstance(text, str) and text.strip():
        return True
    kind = str(_value_for_keys(layer, "kind", "type") or "").strip().lower()
    normalized_kind = kind.replace("_", " ").replace("-", " ")
    return normalized_kind in {"type", "text", "text layer"} or "text" in normalized_kind


def _has_style(layer: dict[str, Any]) -> bool:
    return _has_value(_value_for_keys(layer, "style", "textStyle", "text_style"))


def _is_group_layer(layer: dict[str, Any]) -> bool:
    if _truthy(_value_for_keys(layer, "is_group", "isGroup")):
        return True
    kind = str(_value_for_keys(layer, "kind") or "").lower()
    return "group" in kind or "folder" in kind


def _is_smart_object_ish(layer: dict[str, Any]) -> bool:
    for key, value in layer.items():
        normalized_key = _normalize_key(key)
        if normalized_key in _SMART_OBJECT_KEYS and _truthy(value):
            return True
    kind = str(_value_for_keys(layer, "kind") or "").lower()
    name = str(_value_for_keys(layer, "name", "path") or "").lower()
    return any(token in kind for token in ("smart", "placed", "object")) or "smart object" in name or "smart-object" in name


def _has_nonempty_bbox(layer: dict[str, Any]) -> bool:
    bbox = layer.get("bbox")
    if not isinstance(bbox, dict):
        return False
    try:
        return int(bbox.get("w", 0)) > 0 and int(bbox.get("h", 0)) > 0
    except Exception:
        return False


def _layer_effect_count(layer: dict[str, Any]) -> int:
    count = 0
    for key, value in layer.items():
        normalized_key = _normalize_key(key)
        if normalized_key in _EFFECT_COLLECTION_KEYS:
            count += _entry_count(value)
        elif normalized_key in _EFFECT_SINGLE_KEYS and _has_value(value):
            count += 1
    return count


def _entry_count(value: Any) -> int:
    if isinstance(value, dict):
        return sum(1 for item in value.values() if _has_value(item)) or (1 if _has_value(value) else 0)
    if isinstance(value, list):
        return sum(1 for item in value if _has_value(item))
    return 1 if _has_value(value) else 0


def _warning_count(value: Any) -> int:
    if isinstance(value, list):
        return sum(1 for item in value if _has_value(item))
    return 1 if _has_value(value) else 0


def _value_for_keys(item: dict[str, Any], *keys: str) -> Any:
    wanted = {_normalize_key(key) for key in keys}
    for key, value in item.items():
        if _normalize_key(key) in wanted:
            return value
    return None


def _normalize_key(key: Any) -> str:
    return "".join(character for character in str(key).lower() if character.isalnum())


def _has_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return any(_has_value(item) for item in value)
    if isinstance(value, dict):
        return any(_has_value(item) for item in value.values())
    return True


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 5)


def _probe_photoshopapi() -> dict[str, Any]:
    for module_name in ("photoshopapi", "PhotoshopAPI"):
        try:
            available = importlib.util.find_spec(module_name) is not None
        except Exception:
            available = False
        if available:
            return {
                "requested": True,
                "status": "available",
                "available": True,
                "module": module_name,
            }
    return {
        "requested": True,
        "status": "not_available",
        "available": False,
        "module": None,
    }
