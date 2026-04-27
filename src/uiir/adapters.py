from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detect import build_visual_candidates
from .models import Candidate, relpath


ADAPTER_CANDIDATES_FILE = "adapter_candidates.json"
ADAPTER_MANIFEST_FILE = "adapter_manifest.json"


@dataclass(frozen=True)
class VisionAdapterSpec:
    name: str
    description: str
    status: str
    license_note: str
    dependency_note: str


@dataclass(frozen=True)
class VisionAdapterRun:
    adapter: str
    status: str
    output_dir: Path
    candidates_path: Path
    manifest_path: Path
    candidates: list[Candidate]
    manifest: dict[str, Any]


VISION_ADAPTERS: dict[str, VisionAdapterSpec] = {
    "uied": VisionAdapterSpec(
        name="uied",
        description="Local OpenCV/UIED-style visual component proposals from the extract composite.",
        status="supported",
        license_note="Uses this project's bundled heuristic implementation; no external model weights.",
        dependency_note="Uses optional OpenCV/numpy when installed and falls back to Pillow alpha components.",
    ),
    "omniparser": VisionAdapterSpec(
        name="omniparser",
        description="Reserved adapter slot for OmniParser-style UI parsing.",
        status="skipped",
        license_note="OmniParser is not bundled; review upstream license terms before enabling locally.",
        dependency_note="Requires external code and model weights. This registry never downloads weights.",
    ),
    "sam": VisionAdapterSpec(
        name="sam",
        description="Reserved adapter slot for Segment Anything model proposals.",
        status="skipped",
        license_note="SAM model/code license must be reviewed separately before local integration.",
        dependency_note="Requires heavyweight model weights. This registry never downloads weights.",
    ),
    "paddleocr": VisionAdapterSpec(
        name="paddleocr",
        description="Reserved adapter slot for PaddleOCR text-region proposals.",
        status="skipped",
        license_note="PaddleOCR and model license terms must be reviewed separately before enabling.",
        dependency_note="Requires PaddleOCR runtime/model assets. This registry never downloads weights.",
    ),
}
ADAPTER_REGISTRY = VISION_ADAPTERS


def list_vision_adapters() -> dict[str, VisionAdapterSpec]:
    return dict(VISION_ADAPTERS)


def run_vision_adapter(
    adapter: str,
    extract_output: str | Path | object,
    *,
    output_dir: str | Path | None = None,
    composite_path: str | Path | None = None,
    min_area: int = 96,
    max_candidates: int = 180,
) -> VisionAdapterRun:
    name = _normalize_adapter_name(adapter)
    spec = VISION_ADAPTERS[name]
    out_dir = _resolve_output_dir(extract_output, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_composite = _resolve_composite_path(extract_output, out_dir, composite_path)
    candidates_path = out_dir / ADAPTER_CANDIDATES_FILE
    manifest_path = out_dir / ADAPTER_MANIFEST_FILE

    if spec.status != "supported":
        candidates: list[Candidate] = []
        _write_candidates(candidates_path, candidates)
        manifest = _base_manifest(spec, out_dir, resolved_composite)
        manifest.update(
            {
                "status": "skipped",
                "candidate_count": 0,
                "reason": "adapter_not_bundled",
                "outputs": {"candidates": relpath(candidates_path, out_dir)},
            }
        )
        _write_manifest(manifest_path, manifest)
        return VisionAdapterRun(name, "skipped", out_dir, candidates_path, manifest_path, candidates, manifest)

    candidates = _run_uied(resolved_composite, min_area=min_area, max_candidates=max_candidates)
    _write_candidates(candidates_path, candidates)
    manifest = _base_manifest(spec, out_dir, resolved_composite)
    manifest.update(
        {
            "status": "ok",
            "candidate_count": len(candidates),
            "min_area": min_area,
            "max_candidates": max_candidates,
            "outputs": {"candidates": relpath(candidates_path, out_dir)},
        }
    )
    _write_manifest(manifest_path, manifest)
    return VisionAdapterRun(name, "ok", out_dir, candidates_path, manifest_path, candidates, manifest)


def _run_uied(composite_path: Path, *, min_area: int, max_candidates: int) -> list[Candidate]:
    candidates = build_visual_candidates(composite_path, min_area=min_area, max_candidates=max_candidates)
    for index, candidate in enumerate(candidates, start=1):
        candidate.id = f"uied{index}"
        candidate.source = "adapter:uied"
        candidate.source_refs = [f"uied:{index}"]
        candidate.metadata = {
            **candidate.metadata,
            "adapter": "uied",
            "adapterSource": "build_visual_candidates",
        }
    return candidates


def _normalize_adapter_name(adapter: str) -> str:
    name = str(adapter or "").strip().lower()
    if name not in VISION_ADAPTERS:
        supported = ", ".join(sorted(VISION_ADAPTERS))
        raise ValueError(f"Unknown vision adapter {adapter!r}. Available adapters: {supported}")
    return name


def _resolve_output_dir(extract_output: str | Path | object, output_dir: str | Path | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    if isinstance(extract_output, (str, Path)):
        path = Path(extract_output).expanduser().resolve()
        return path if path.is_dir() else path.parent
    composite = getattr(extract_output, "composite_path", None)
    if composite is not None:
        return Path(composite).expanduser().resolve().parent
    source = getattr(extract_output, "source", None)
    if source is not None:
        return Path(source).expanduser().resolve().parent
    return Path.cwd()


def _resolve_composite_path(extract_output: str | Path | object, output_dir: Path, composite_path: str | Path | None) -> Path:
    if composite_path is not None:
        return Path(composite_path).expanduser().resolve()
    if not isinstance(extract_output, (str, Path)):
        value = getattr(extract_output, "composite_path", None)
        if value is not None:
            return Path(value).expanduser().resolve()
    path = Path(extract_output).expanduser().resolve() if isinstance(extract_output, (str, Path)) else output_dir
    if path.is_file():
        return path
    metadata_path = path / "layer_metadata.json"
    if metadata_path.exists():
        composite = _composite_from_metadata(metadata_path)
        if composite is not None:
            return composite
    return path / "composite.png"


def _composite_from_metadata(metadata_path: Path) -> Path | None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(metadata, dict) or not metadata.get("composite"):
        return None
    composite = Path(str(metadata["composite"]))
    if not composite.is_absolute():
        composite = metadata_path.parent / composite
    return composite.expanduser().resolve()


def _base_manifest(spec: VisionAdapterSpec, output_dir: Path, composite_path: Path) -> dict[str, Any]:
    return {
        "adapter": spec.name,
        "description": spec.description,
        "composite": relpath(composite_path, output_dir),
        "license_note": spec.license_note,
        "dependency_note": spec.dependency_note,
        "downloads": [],
        "weights_downloaded": False,
    }


def _write_candidates(path: Path, candidates: list[Candidate]) -> None:
    path.write_text(
        json.dumps([candidate.to_dict() for candidate in candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
