from __future__ import annotations

import json
import os
import hashlib
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PSD_EXTENSIONS = (".psd", ".psb")


@dataclass(frozen=True)
class FixtureSource:
    name: str
    repo: str | None = None
    path: str | None = None
    ref: str = "main"
    max_files: int = 8
    url: str | None = None
    file_name: str | None = None
    license: str | None = None
    source_url: str | None = None
    attribution: str | None = None
    expected: str = "ok"


@dataclass
class FixtureFile:
    source: str
    repo: str | None
    repo_path: str | None
    output_path: str
    size: int | None
    sha: str | None
    sha256: str | None
    license: str | None
    source_url: str | None
    attribution: str | None
    expected: str
    downloaded: bool
    archive_path: str | None = None


PARSER_SMOKE_SOURCES = [
    FixtureSource("psd-tools", repo="psd-tools/psd-tools", path="tests/psd_files", ref="main", max_files=12, source_url="https://github.com/psd-tools/psd-tools/tree/main/tests/psd_files"),
    FixtureSource("ag-psd", repo="Agamnentzar/ag-psd", path="test/read-write", ref="master", max_files=8, source_url="https://github.com/Agamnentzar/ag-psd/tree/master/test/read-write"),
    FixtureSource("webtoon-psd", repo="webtoon/psd", path="packages/psd/tests/integration/fixtures", ref="main", max_files=6, source_url="https://github.com/webtoon/psd/tree/main/packages/psd/tests/integration/fixtures"),
    FixtureSource("baum2", repo="kyubuns/Baum2", path="SamplePsd", ref="master", max_files=2, source_url="https://github.com/kyubuns/Baum2/tree/master/SamplePsd"),
]

GAME_UI_SMOKE_SOURCES = [
    FixtureSource(
        "opengameart-rpg-game-ui",
        url="https://opengameart.org/sites/default/files/interface.psd",
        file_name="interface.psd",
        license="CC0",
        source_url="https://opengameart.org/content/rpg-game-ui",
        attribution="Wyrmheart",
    ),
    FixtureSource(
        "opengameart-user-interface",
        url="https://opengameart.org/sites/default/files/GUI%20Design.psd",
        file_name="GUI Design.psd",
        license="CC-BY 3.0",
        source_url="https://opengameart.org/content/user-interface-0",
        attribution="ドリームキャスト",
    ),
    FixtureSource(
        "opengameart-2d-ui-kit",
        url="https://opengameart.org/sites/default/files/game-ui-p-1.psd",
        file_name="game-ui-p-1.psd",
        license="CC0",
        source_url="https://opengameart.org/content/2d-ui-kit",
        attribution="MontaG97",
    ),
    FixtureSource(
        "opengameart-golden-ui",
        url="https://lpc.opengameart.org/sites/default/files/ui.psd",
        file_name="ui.psd",
        license="CC0",
        source_url="https://lpc.opengameart.org/content/golden-ui",
        attribution="Buch",
    ),
    FixtureSource(
        "opengameart-ui-elements",
        url="https://opengameart.org/sites/default/files/Some-ui-stuff.zip",
        file_name="Some-ui-stuff.zip",
        license="CC0",
        source_url="https://opengameart.org/content/ui-elements-0",
        attribution="kindland",
    ),
]

OPENAI_SMOKE_SOURCES = [
    GAME_UI_SMOKE_SOURCES[0],
    GAME_UI_SMOKE_SOURCES[3],
]

FIXTURE_PRESETS: dict[str, list[FixtureSource]] = {
    "parser-smoke": PARSER_SMOKE_SOURCES,
    "game-ui-smoke": GAME_UI_SMOKE_SOURCES,
    "openai-smoke": OPENAI_SMOKE_SOURCES,
    "all-smoke": [*PARSER_SMOKE_SOURCES, *GAME_UI_SMOKE_SOURCES],
}


def list_fixture_sets() -> dict[str, list[dict[str, Any]]]:
    return {name: [asdict(source) for source in sources] for name, sources in FIXTURE_PRESETS.items()}


def download_fixture_set(set_name: str, output_dir: str | Path, limit: int | None = None, overwrite: bool = False) -> dict[str, Any]:
    if set_name not in FIXTURE_PRESETS:
        available = ", ".join(sorted(FIXTURE_PRESETS))
        raise ValueError(f"Unknown fixture set {set_name!r}. Available sets: {available}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[FixtureFile] = []
    warnings: list[str] = []
    remaining = limit

    for source in FIXTURE_PRESETS[set_name]:
        source_limit = source.max_files if remaining is None else min(source.max_files, remaining)
        if source_limit <= 0:
            break
        try:
            entries = _source_entries(source, out_dir, overwrite)[:source_limit]
        except Exception as exc:
            warnings.append(f"{source.name}: could not list fixtures: {exc}")
            continue
        for entry in entries:
            target = out_dir / source.name / _safe_repo_path(entry["path"])
            try:
                if entry.get("local_path"):
                    target = Path(entry["local_path"])
                    downloaded = bool(entry.get("downloaded"))
                else:
                    downloaded = _download(entry["download_url"], target, overwrite=overwrite)
            except Exception as exc:
                warnings.append(f"{source.name}: could not download {entry['path']}: {exc}")
                continue
            files.append(
                FixtureFile(
                    source=source.name,
                    repo=source.repo,
                    repo_path=entry.get("repo_path") or entry["path"],
                    output_path=target.relative_to(out_dir).as_posix(),
                    size=target.stat().st_size if target.exists() else entry.get("size"),
                    sha=entry.get("sha"),
                    sha256=_sha256(target) if target.exists() else None,
                    license=source.license,
                    source_url=entry.get("source_url") or source.source_url,
                    attribution=source.attribution,
                    expected=entry.get("expected") or source.expected,
                    downloaded=downloaded,
                    archive_path=entry.get("archive_path"),
                )
            )
        if remaining is not None:
            remaining -= len(entries)

    manifest = {
        "set": set_name,
        "root": out_dir.as_posix(),
        "count": len(files),
        "warnings": warnings,
        "sources": [asdict(source) for source in FIXTURE_PRESETS[set_name]],
        "files": [asdict(item) for item in files],
    }
    (out_dir / "fixtures.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _source_entries(source: FixtureSource, out_dir: Path, overwrite: bool) -> list[dict[str, Any]]:
    if source.url:
        return _direct_entries(source, out_dir, overwrite)
    return _list_psd_entries(source)


def _direct_entries(source: FixtureSource, out_dir: Path, overwrite: bool) -> list[dict[str, Any]]:
    if not source.url:
        return []
    file_name = source.file_name or Path(urllib.parse.unquote(urllib.parse.urlparse(source.url).path)).name
    if not file_name:
        raise ValueError(f"{source.name}: direct fixture URL has no file name")
    raw_target = out_dir / source.name / _safe_repo_path(file_name)
    downloaded = _download(source.url, raw_target, overwrite=overwrite)
    if raw_target.suffix.lower() == ".zip":
        return _zip_entries(source, raw_target, out_dir, downloaded, overwrite)
    if raw_target.suffix.lower() not in PSD_EXTENSIONS:
        raise ValueError(f"{source.name}: direct fixture is not a PSD/PSB/ZIP: {raw_target.name}")
    return [
        {
            "path": raw_target.name,
            "local_path": raw_target.as_posix(),
            "size": raw_target.stat().st_size,
            "sha": None,
            "downloaded": downloaded,
            "source_url": source.source_url,
            "expected": source.expected,
        }
    ]


def _zip_entries(source: FixtureSource, archive_path: Path, out_dir: Path, downloaded: bool, overwrite: bool) -> list[dict[str, Any]]:
    extract_root = out_dir / source.name / "extracted"
    entries: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(PSD_EXTENSIONS):
                continue
            safe_member = _safe_zip_member(member.filename)
            target = extract_root / safe_member
            if not target.exists() or overwrite:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source_file:
                    target.write_bytes(source_file.read())
            entries.append(
                {
                    "path": f"extracted/{safe_member}",
                    "local_path": target.as_posix(),
                    "size": target.stat().st_size,
                    "sha": None,
                    "downloaded": downloaded,
                    "source_url": source.source_url,
                    "expected": source.expected,
                    "archive_path": archive_path.relative_to(out_dir).as_posix(),
                }
            )
    if not entries:
        raise ValueError(f"{source.name}: ZIP did not contain PSD/PSB files")
    return sorted(entries, key=lambda item: item["path"].lower())


def _list_psd_entries(source: FixtureSource) -> list[dict[str, Any]]:
    if not source.repo or not source.path:
        return []
    entries: list[dict[str, Any]] = []
    _walk_github_contents(source, source.path, entries)
    return sorted(entries, key=lambda item: item["path"].lower())


def _walk_github_contents(source: FixtureSource, path: str, entries: list[dict[str, Any]]) -> None:
    payload = _github_json(_contents_url(source.repo, path, source.ref))
    items = payload if isinstance(payload, list) else [payload]
    for item in sorted(items, key=lambda value: value.get("path", "").lower()):
        kind = item.get("type")
        item_path = item.get("path", "")
        if kind == "dir":
            _walk_github_contents(source, item_path, entries)
            continue
        if kind != "file" or not item_path.lower().endswith(PSD_EXTENSIONS):
            continue
        download_url = item.get("download_url")
        if download_url:
            entries.append(item)


def _contents_url(repo: str, path: str, ref: str) -> str:
    quoted = urllib.parse.quote(path.strip("/"))
    return f"https://api.github.com/repos/{repo}/contents/{quoted}?ref={urllib.parse.quote(ref)}"


def _github_json(url: str) -> Any:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, target: Path, overwrite: bool) -> bool:
    if target.exists() and not overwrite:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())
    return True


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "psd-uiir-fixture-downloader"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _safe_repo_path(path: str) -> str:
    return "/".join(part.replace(":", "_") for part in path.split("/") if part and part not in {".", ".."})


def _safe_zip_member(path: str) -> str:
    parts = []
    for part in Path(path).parts:
        if part in {"", ".", ".."} or part.endswith(":"):
            continue
        parts.append(part.replace(":", "_"))
    return "/".join(parts) or "fixture.psd"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
