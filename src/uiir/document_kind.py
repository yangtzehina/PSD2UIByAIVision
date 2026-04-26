from __future__ import annotations

from pathlib import Path

from .models import LayerRecord


DOCUMENT_KINDS = ("auto", "screen", "asset_sheet")


def resolve_document_kind(requested: str, source: str | Path, layers: list[LayerRecord]) -> str:
    normalized = (requested or "auto").strip().lower().replace("-", "_")
    if normalized not in DOCUMENT_KINDS:
        raise ValueError(f"Unsupported document kind {requested!r}; expected auto, screen, or asset_sheet")
    if normalized != "auto":
        return normalized
    return classify_document_kind(source, layers)


def classify_document_kind(source: str | Path, layers: list[LayerRecord]) -> str:
    stem = Path(source).stem.lower().replace("-", " ").replace("_", " ")
    score = 0
    if stem in {"ui", "gui", "gui design", "game ui p 1", "game ui", "2d ui kit"}:
        score += 3
    if any(token in stem for token in ("sprite", "sheet", "kit", "atlas", "elements")):
        score += 2

    names = " / ".join(f"{layer.name} {layer.path}" for layer in layers[:400]).lower()
    palette_terms = (
        "buttons",
        "icons",
        "slider",
        "scrollbar",
        "listbox",
        "inputbox",
        "radio button",
        "common",
        "cursors",
        "items copia",
    )
    score += sum(1 for term in palette_terms if term in names)
    top_level_groups = {layer.name.lower() for layer in layers if layer.is_group and layer.depth <= 1}
    if {"common", "menu"} & top_level_groups and any(term in names for term in ("buttons", "icons", "slider")):
        score += 2
    return "asset_sheet" if score >= 4 else "screen"
