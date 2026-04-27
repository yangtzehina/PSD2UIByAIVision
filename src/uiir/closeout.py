from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .adapters import run_vision_adapter
from .batch import run_batch
from .compare_openai import IterateOptions, run_iterate_openai
from .evaluate import evaluate_outputs
from .fidelity import build_parser_fidelity_report
from .focus import build_focus_tiles
from .graph import build_ui_graph
from .pipeline import ExtractOptions
from .render_review import review_render


DEFAULT_SENSITIVE_PATTERNS = (
    r"Bearer\s+[A-Za-z0-9._\-]{20,}",
    r"(?:OPENAI|UIIR|PROVIDER|THIRD_PARTY)_[A-Z0-9_]*API_KEY\s*=\s*['\"]?[A-Za-z0-9._\-]{12,}",
    r"sk-[A-Za-z0-9]{20,}",
)
PIXEL_BASELINE = 0.88996
PIXEL_MIN_RATIO = 0.95


@dataclass(frozen=True)
class CloseoutOptions:
    output_dir: str | Path = "out/closeout"
    fixture_dir: str | Path = "fixtures/game-ui-smoke"
    openai_fixture_dir: str | Path = "fixtures/openai-smoke"
    golden_root: str | Path | None = "goldens/local"
    model: str = "gpt-5.5"
    detail: str = "original"
    provider_name: str = "third-party"
    api_key_env: str = "UIIR_PROVIDER_API_KEY"
    base_url: str | None = None
    api_mode: str = "chat-completions"
    limit: int = 2
    run_provider_smoke: bool = False
    skip_inspector_build: bool = False
    skip_adapter: bool = False
    dry_run: bool = False
    sensitive_patterns: tuple[str, ...] = DEFAULT_SENSITIVE_PATTERNS


CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]


def run_closeout(options: CloseoutOptions, command_runner: CommandRunner | None = None) -> dict[str, Any]:
    root = _repo_root()
    out_dir = Path(options.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command_runner = command_runner or _run_command

    report: dict[str, Any] = {
        "version": "0.1",
        "created_at": _now(),
        "status": "planned" if options.dry_run else "ok",
        "output_dir": out_dir.as_posix(),
        "baseline_pixel": PIXEL_BASELINE,
        "pixel_min_ratio": PIXEL_MIN_RATIO,
        "inputs": {
            "fixture_dir": Path(options.fixture_dir).expanduser().as_posix(),
            "openai_fixture_dir": Path(options.openai_fixture_dir).expanduser().as_posix(),
            "golden_root": Path(options.golden_root).expanduser().as_posix() if options.golden_root else None,
        },
        "provider": {
            "provider_name": options.provider_name,
            "api_key_env": options.api_key_env,
            "api_key_present": bool(os.getenv(options.api_key_env)),
            "base_url_present": bool(options.base_url),
            "api_mode": options.api_mode,
            "model": options.model,
            "run_provider_smoke": options.run_provider_smoke,
        },
        "commands": [],
        "baseline": None,
        "diagnostics": [],
        "openai_skip": None,
        "provider_smoke": None,
        "sensitive_scan": None,
        "gates": {},
    }

    _append_command(report, "py_compile", [sys.executable, "-m", "compileall", "-q", "src/uiir"], root, options, command_runner)
    _append_command(report, "unittest", [sys.executable, "-m", "unittest", "discover", "-s", "tests"], root, options, command_runner)
    if not options.skip_inspector_build:
        _append_command(report, "inspector_build", ["npm", "run", "build", "--", "--base", "/PSD2UIByAIVision/"], root / "inspector", options, command_runner)

    if options.dry_run:
        report["sensitive_scan"] = _sensitive_scan(root, options.sensitive_patterns)
        report["gates"] = _closeout_gates(report)
        _write_closeout_report(out_dir, report)
        return report

    fixture_dir = Path(options.fixture_dir).expanduser().resolve()
    baseline_dir = out_dir / "game-ui-smoke"
    batch_report = run_batch(fixture_dir, baseline_dir, ExtractOptions(use_openai=False))
    metrics = evaluate_outputs(baseline_dir, report_path=baseline_dir / "metrics.json")
    report["baseline"] = {"batch": _baseline_batch_summary(batch_report), "metrics": _metrics_summary(metrics)}

    report["diagnostics"] = _run_diagnostics(batch_report, out_dir, skip_adapter=options.skip_adapter)
    report["openai_skip"] = _run_openai_skip(options, out_dir)
    if options.run_provider_smoke:
        report["provider_smoke"] = _run_provider_smoke(options, out_dir)
    report["sensitive_scan"] = _sensitive_scan(root, options.sensitive_patterns)
    report["gates"] = _closeout_gates(report)
    if not all(report["gates"].values()):
        report["status"] = "failed"
    _write_closeout_report(out_dir, report)
    return report


def _append_command(
    report: dict[str, Any],
    label: str,
    command: list[str],
    cwd: Path,
    options: CloseoutOptions,
    runner: CommandRunner,
) -> None:
    display_command = _redact_command(command, options)
    if options.dry_run:
        report["commands"].append({"label": label, "status": "planned", "command": display_command, "cwd": cwd.as_posix()})
        return
    started = time.perf_counter()
    result = runner(command, cwd)
    report["commands"].append(
        {
            "label": label,
            "status": "ok" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "seconds": round(time.perf_counter() - started, 3),
            "command": display_command,
            "cwd": cwd.as_posix(),
            "stdout_tail": _tail(result.stdout),
            "stderr_tail": _tail(result.stderr),
        }
    )


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def _run_diagnostics(batch_report: dict[str, Any], out_dir: Path, *, skip_adapter: bool) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    adapter_root = out_dir / "adapter-uied"
    for item in batch_report.get("items", []):
        if not item.get("ok"):
            continue
        output_dir = Path(item["output_dir"])
        name = output_dir.name
        entry: dict[str, Any] = {"sample": name, "output_dir": output_dir.as_posix()}
        try:
            graph = build_ui_graph(output_dir)
            entry["graph"] = {"nodes": graph.node_count, "edges": graph.edge_count}
        except Exception as exc:
            entry["graph"] = {"error": str(exc)}
        try:
            render = review_render(output_dir)
            entry["render_review"] = {"status": render.get("status"), "issue_count": render.get("issue_count", 0)}
        except Exception as exc:
            entry["render_review"] = {"error": str(exc)}
        try:
            focus = build_focus_tiles(output_dir)
            entry["focus_tiles"] = {"count": focus.get("count", 0), "path": (output_dir / "focus_tiles.json").as_posix()}
        except Exception as exc:
            entry["focus_tiles"] = {"error": str(exc)}
        try:
            fidelity = build_parser_fidelity_report(output_dir)
            entry["parser_fidelity"] = {
                "layer_count": fidelity.get("layer_count"),
                "text_layers": fidelity.get("text_layers"),
                "smart_object_ish_layers": fidelity.get("smart_object_ish_layers"),
                "asset_coverage": fidelity.get("asset_coverage"),
            }
        except Exception as exc:
            entry["parser_fidelity"] = {"error": str(exc)}
        if skip_adapter:
            entry["adapter_uied"] = {"status": "skipped"}
        else:
            try:
                adapter = run_vision_adapter("uied", output_dir, output_dir=adapter_root / name)
                entry["adapter_uied"] = {"status": adapter.status, "candidate_count": adapter.candidate_count}
            except Exception as exc:
                entry["adapter_uied"] = {"status": "failed", "error": str(exc)}
        diagnostics.append(entry)
    return diagnostics


def _run_openai_skip(options: CloseoutOptions, out_dir: Path) -> dict[str, Any]:
    return run_iterate_openai(
        Path(options.openai_fixture_dir).expanduser().resolve(),
        out_dir / "provider-skip",
        IterateOptions(
            model=options.model,
            detail=options.detail,
            limit=options.limit,
            provider_name=options.provider_name,
            api_key_env="UIIR_CLOSEOUT_MISSING_API_KEY",
            api_mode=options.api_mode,
            prompt_versions=("semantic_v3", "relation_v1"),
            policies=("audit", "strict"),
            graph_overlay=True,
            render_review=True,
            curation_report=True,
            focus_tiles=True,
            parser_fidelity=True,
        ),
    )


def _run_provider_smoke(options: CloseoutOptions, out_dir: Path) -> dict[str, Any]:
    golden = Path(options.golden_root).expanduser().resolve() if options.golden_root and Path(options.golden_root).expanduser().exists() else None
    report = run_iterate_openai(
        Path(options.openai_fixture_dir).expanduser().resolve(),
        out_dir / "provider-smoke",
        IterateOptions(
            model=options.model,
            detail=options.detail,
            limit=options.limit,
            provider_name=options.provider_name,
            api_key_env=options.api_key_env,
            base_url=options.base_url,
            api_mode=options.api_mode,
            golden_root=golden,
            prompt_versions=("semantic_v3", "relation_v1"),
            policies=("audit", "strict"),
            graph_overlay=True,
            render_review=True,
            curation_report=True,
            focus_tiles=True,
            parser_fidelity=True,
        ),
    )
    return _redact_provider_report(report)


def _sensitive_scan(root: Path, patterns: tuple[str, ...]) -> dict[str, Any]:
    compiled = [(index + 1, re.compile(pattern)) for index, pattern in enumerate(patterns)]
    files = _tracked_files(root)
    matches = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern_index, pattern in compiled:
                if pattern.search(line):
                    matches.append({"file": path.relative_to(root).as_posix(), "line": line_no, "pattern_index": pattern_index})
    return {
        "status": "ok" if not matches else "failed",
        "tracked_files_scanned": len(files),
        "pattern_count": len(patterns),
        "match_count": len(matches),
        "matches": matches,
    }


def _tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(["git", "ls-files"], cwd=root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return []
    return [root / line.strip() for line in result.stdout.splitlines() if line.strip()]


def _closeout_gates(report: dict[str, Any]) -> dict[str, bool]:
    command_ok = all(item.get("status") in {"ok", "planned"} for item in report.get("commands", []))
    baseline = report.get("baseline") or {}
    batch = baseline.get("batch") or {}
    metrics = baseline.get("metrics") or {}
    pixel = metrics.get("avg_pixel_similarity")
    return {
        "commands_ok": command_ok,
        "baseline_failed_zero": report.get("status") == "planned" or batch.get("failed") == 0,
        "schema_all_ok": report.get("status") == "planned" or metrics.get("schema_ok") == metrics.get("count"),
        "pixel_not_degraded": report.get("status") == "planned" or pixel is None or pixel >= PIXEL_BASELINE * PIXEL_MIN_RATIO,
        "openai_skip_ok": report.get("status") == "planned" or (report.get("openai_skip") or {}).get("status") == "skipped",
        "provider_smoke_ok": report.get("status") == "planned" or not report.get("provider", {}).get("run_provider_smoke") or (report.get("provider_smoke") or {}).get("status") == "ok",
        "sensitive_scan_ok": (report.get("sensitive_scan") or {}).get("status") == "ok",
    }


def _baseline_batch_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": report.get("input"),
        "output": report.get("output"),
        "count": report.get("count"),
        "ok": report.get("ok"),
        "skipped": report.get("skipped"),
        "failed": report.get("failed"),
        "seconds": report.get("seconds"),
    }


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": metrics.get("count"),
        "schema_ok": metrics.get("schema_ok"),
        "avg_pixel_similarity": metrics.get("avg_pixel_similarity"),
        "avg_type_f1": metrics.get("avg_type_f1"),
        "avg_relation_f1": metrics.get("avg_relation_f1"),
        "avg_component_group_f1": metrics.get("avg_component_group_f1"),
    }


def _write_closeout_report(out_dir: Path, report: dict[str, Any]) -> None:
    (out_dir / "closeout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "closeout_report.md").write_text(_closeout_markdown(report), encoding="utf-8")


def _closeout_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# UIIR Closeout Report",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Output: `{report.get('output_dir')}`",
        f"- Provider smoke: `{'enabled' if report.get('provider', {}).get('run_provider_smoke') else 'disabled'}`",
        "",
        "## Gates",
    ]
    for key, value in (report.get("gates") or {}).items():
        lines.append(f"- `{key}`: {'PASS' if value else 'FAIL'}")
    lines.extend(["", "## Commands"])
    for item in report.get("commands", []):
        lines.append(f"- `{item.get('label')}`: {item.get('status')}")
    baseline = report.get("baseline") or {}
    if baseline:
        metrics = baseline.get("metrics") or {}
        lines.extend(
            [
                "",
                "## Baseline",
                f"- Samples: `{metrics.get('count')}`",
                f"- Schema OK: `{metrics.get('schema_ok')}`",
                f"- Avg pixel similarity: `{metrics.get('avg_pixel_similarity')}`",
            ]
        )
    diagnostics = report.get("diagnostics") or []
    if diagnostics:
        lines.extend(["", "## Diagnostics"])
        for item in diagnostics:
            lines.append(
                f"- `{item.get('sample')}`: graph edges `{item.get('graph', {}).get('edges')}`, "
                f"render issues `{item.get('render_review', {}).get('issue_count')}`, "
                f"focus tiles `{item.get('focus_tiles', {}).get('count')}`, "
                f"adapter candidates `{item.get('adapter_uied', {}).get('candidate_count')}`"
            )
    scan = report.get("sensitive_scan") or {}
    lines.extend(["", "## Sensitive Scan", f"- Status: `{scan.get('status')}`", f"- Matches: `{scan.get('match_count')}`"])
    return "\n".join(lines) + "\n"


def _redact_provider_report(report: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(json.dumps(report))
    provider = data.get("provider")
    if isinstance(provider, dict):
        provider["base_url"] = "<redacted>" if provider.get("base_url") else None
    for run in data.get("runs", []) or []:
        run.pop("provider", None)
    return data


def _redact_command(command: list[str], options: CloseoutOptions) -> list[str]:
    if not options.base_url:
        return command
    return ["<base-url-redacted>" if value == options.base_url else value for value in command]


def _tail(text: str | None, limit: int = 1800) -> str:
    text = text or ""
    return text[-limit:]


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
