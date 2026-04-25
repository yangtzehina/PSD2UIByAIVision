from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .pipeline import ExtractOptions, run_extract


PSD_EXTENSIONS = (".psd", ".psb")


@dataclass
class BatchItem:
    source: str
    output_dir: str
    ok: bool
    seconds: float
    expected: str = "ok"
    skipped: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def run_batch(input_dir: str | Path, output_dir: str | Path, options: ExtractOptions, limit: int | None = None) -> dict[str, Any]:
    source_dir = Path(input_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    psd_paths = _find_psd_files(source_dir)
    expectations = _fixture_expectations(source_dir)
    if limit is not None:
        psd_paths = psd_paths[: max(0, limit)]

    items: list[BatchItem] = []
    used_slugs: set[str] = set()
    started = time.perf_counter()
    for index, psd_path in enumerate(psd_paths, start=1):
        item_out = out_dir / _unique_slug(psd_path, used_slugs, index)
        item_started = time.perf_counter()
        expected = expectations.get(psd_path.resolve().as_posix(), "ok")
        try:
            artifacts = run_extract(psd_path, item_out, options)
            item = BatchItem(
                source=psd_path.as_posix(),
                output_dir=item_out.as_posix(),
                ok=True,
                seconds=round(time.perf_counter() - item_started, 3),
                expected=expected,
                artifacts={
                    "composite": artifacts.composite.as_posix(),
                    "overlay": artifacts.overlay.as_posix(),
                    "layers": artifacts.layers_json.as_posix(),
                    "candidates": artifacts.candidates_json.as_posix(),
                    "json": artifacts.uiir_json.as_posix(),
                    "xml": artifacts.uiir_xml.as_posix(),
                },
            )
        except Exception as exc:
            skipped = expected == "skip"
            item = BatchItem(
                source=psd_path.as_posix(),
                output_dir=item_out.as_posix(),
                ok=False,
                seconds=round(time.perf_counter() - item_started, 3),
                expected=expected,
                skipped=skipped,
                error=str(exc),
            )
        items.append(item)

    report = {
        "input": source_dir.as_posix(),
        "output": out_dir.as_posix(),
        "count": len(items),
        "ok": sum(1 for item in items if item.ok),
        "skipped": sum(1 for item in items if item.skipped),
        "failed": sum(1 for item in items if not item.ok and not item.skipped),
        "seconds": round(time.perf_counter() - started, 3),
        "items": [asdict(item) for item in items],
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _find_psd_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in PSD_EXTENSIONS:
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(
        [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in PSD_EXTENSIONS],
        key=lambda path: path.as_posix().lower(),
    )


def _fixture_expectations(root: Path) -> dict[str, str]:
    manifest = root / "fixtures.manifest.json"
    if not manifest.exists():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return {}
    fixture_root = Path(data.get("root") or root).expanduser().resolve()
    expectations: dict[str, str] = {}
    for item in data.get("files", []) or []:
        if not isinstance(item, dict):
            continue
        output_path = item.get("output_path")
        if not output_path:
            continue
        expected = str(item.get("expected") or "ok")
        expectations[(fixture_root / output_path).resolve().as_posix()] = expected
    return expectations


def _unique_slug(path: Path, used: set[str], index: int) -> str:
    base = re.sub(r"[^a-zA-Z0-9_.-]+", "_", path.stem).strip("._") or f"item_{index}"
    slug = base
    suffix = 2
    while slug in used:
        slug = f"{base}_{suffix}"
        suffix += 1
    used.add(slug)
    return slug
