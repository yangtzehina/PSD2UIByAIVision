from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batch import run_batch
from .compare_openai import CompareOptions, IterateOptions, review_run, run_compare_openai, run_iterate_openai
from .evaluate import evaluate_outputs
from .fixtures import download_fixture_set, list_fixture_sets
from .golden import build_golden_from_decisions
from .pipeline import ExtractOptions, run_extract
from .schema import UIIR_JSON_SCHEMA


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="uiir", description="Extract a PSD into UIIR XML/JSON artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract PSD/PSB into UIIR artifacts.")
    extract.add_argument("input", help="Input PSD/PSB path.")
    extract.add_argument("--out", required=True, help="Output directory.")
    extract.add_argument("--no-visual", action="store_true", help="Disable local visual contour candidates.")
    extract.add_argument("--ocr", action="store_true", help="Enable optional local OCR candidates when pytesseract is available.")
    extract.add_argument("--min-area", type=int, default=96, help="Minimum candidate area in pixels.")
    extract.add_argument("--use-openai", action="store_true", help="Use GPT-5.5 semantic refinement.")
    extract.add_argument("--model", default="gpt-5.5", help="OpenAI model for semantic refinement.")
    extract.add_argument("--detail", default="original", choices=("low", "high", "original", "auto"), help="Image detail level.")
    _add_vision_args(extract)
    _add_document_args(extract)
    _add_provider_args(extract)
    extract.add_argument("--corrections", help="Optional corrections.json exported from the inspector.")

    fixtures = subparsers.add_parser("fixtures", help="Manage public PSD fixture sets.")
    fixture_subparsers = fixtures.add_subparsers(dest="fixture_command", required=True)
    fixture_list = fixture_subparsers.add_parser("list", help="List fixture presets.")
    fixture_list.add_argument("--pretty", action="store_true", default=True)
    fixture_download = fixture_subparsers.add_parser("download", help="Download a fixture preset.")
    fixture_download.add_argument("--set", default="parser-smoke", help="Fixture set name.")
    fixture_download.add_argument("--out", required=True, help="Output fixture directory.")
    fixture_download.add_argument("--limit", type=int, help="Maximum number of files across sources.")
    fixture_download.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")

    batch = subparsers.add_parser("batch", help="Run extraction over a directory of PSD/PSB files.")
    batch.add_argument("input", help="Input PSD/PSB file or directory.")
    batch.add_argument("--out", required=True, help="Output batch directory.")
    batch.add_argument("--limit", type=int, help="Maximum number of PSD files to process.")
    batch.add_argument("--no-visual", action="store_true", help="Disable local visual contour candidates.")
    batch.add_argument("--ocr", action="store_true", help="Enable optional local OCR candidates.")
    batch.add_argument("--min-area", type=int, default=96, help="Minimum candidate area in pixels.")
    batch.add_argument("--use-openai", nargs="?", const=True, default=False, type=_parse_bool, help="Use GPT-5.5 semantic refinement.")
    batch.add_argument("--model", default="gpt-5.5", help="OpenAI model for semantic refinement.")
    batch.add_argument("--detail", default="original", choices=("low", "high", "original", "auto"), help="Image detail level.")
    _add_vision_args(batch)
    _add_document_args(batch)
    _add_provider_args(batch)
    batch.add_argument("--corrections", help="Optional corrections.json applied to every item.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate generated UIIR output directories.")
    evaluate.add_argument("output", help="Output directory from uiir extract or uiir batch.")
    evaluate.add_argument("--golden", help="Optional golden UIIR directory.")
    evaluate.add_argument("--report", help="Metrics report path. Defaults to <output>/metrics.json.")

    golden = subparsers.add_parser("golden", help="Build local golden UIIR data from human review decisions.")
    golden_subparsers = golden.add_subparsers(dest="golden_command", required=True)
    golden_build = golden_subparsers.add_parser("build", help="Build a golden UIIR sample from a run and golden_decisions.json.")
    golden_build.add_argument("--psd", required=True, help="Source PSD/PSB used for the run.")
    golden_build.add_argument("--run", required=True, help="OpenAI sample output directory containing candidates.json and vision_quarantined.json.")
    golden_build.add_argument("--decisions", required=True, help="golden_decisions.json exported from the inspector.")
    golden_build.add_argument("--out", required=True, help="Output golden sample directory.")

    compare = subparsers.add_parser("compare-openai", help="Compare local baseline against GPT-5.5 semantic refinement.")
    compare.add_argument("input", help="Input PSD/PSB file or fixture directory.")
    compare.add_argument("--out", required=True, help="Output comparison directory.")
    compare.add_argument("--limit", type=int, default=2, help="Maximum PSD files to process. Defaults to 2.")
    compare.add_argument("--model", default="gpt-5.5", help="OpenAI model for semantic refinement.")
    compare.add_argument("--detail", default="original", choices=("low", "high", "original", "auto"), help="Image detail level.")
    compare.add_argument("--prompt-version", default="semantic_v2", help="Prompt/schema version identifier.")
    _add_vision_args(compare)
    _add_document_args(compare)
    _add_provider_args(compare)
    compare.add_argument("--golden", help="Optional golden UIIR directory for comparison metrics.")
    compare.add_argument("--ocr", action="store_true", help="Enable optional local OCR candidates.")
    compare.add_argument("--min-area", type=int, default=96, help="Minimum candidate area in pixels.")

    iterate = subparsers.add_parser("iterate-openai", help="Run a small OpenAI policy matrix and write a leaderboard.")
    iterate.add_argument("input", help="Input PSD/PSB file or fixture directory.")
    iterate.add_argument("--out", required=True, help="Output iteration directory.")
    iterate.add_argument("--limit", type=int, default=2, help="Maximum PSD files to process. Defaults to 2.")
    iterate.add_argument("--model", default="gpt-5.5", help="OpenAI model for semantic refinement.")
    iterate.add_argument("--detail", default="original", choices=("low", "high", "original", "auto"), help="Image detail level.")
    iterate.add_argument("--prompt-version", default="semantic_v2", help="Prompt/schema version identifier.")
    iterate.add_argument("--prompts", help="Comma-separated prompt versions. Defaults to --prompt-version.")
    iterate.add_argument("--policies", default="audit,strict,balanced", help="Comma-separated vision policies to compare.")
    iterate.add_argument("--golden", help="Optional golden UIIR directory for leaderboard metrics.")
    _add_document_args(iterate)
    _add_provider_args(iterate)
    iterate.add_argument("--ocr", action="store_true", help="Enable optional local OCR candidates.")
    iterate.add_argument("--min-area", type=int, default=96, help="Minimum candidate area in pixels.")

    review = subparsers.add_parser("review-run", help="Review a compare-openai run and write issue summaries.")
    review.add_argument("run", help="Directory containing comparison.json.")

    schema = subparsers.add_parser("schema", help="Print UIIR JSON Schema.")
    schema.add_argument("--pretty", action="store_true", default=True)

    args = parser.parse_args(argv)
    if args.command == "extract":
        return _extract(args)
    if args.command == "fixtures":
        return _fixtures(args)
    if args.command == "batch":
        return _batch(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "golden":
        return _golden(args)
    if args.command == "compare-openai":
        return _compare_openai(args)
    if args.command == "iterate-openai":
        return _iterate_openai(args)
    if args.command == "review-run":
        return _review_run(args)
    if args.command == "schema":
        print(json.dumps(UIIR_JSON_SCHEMA, ensure_ascii=False, indent=2))
        return 0
    parser.print_help()
    return 1


def _extract(args: argparse.Namespace) -> int:
    options = ExtractOptions(
        include_visual=not args.no_visual,
        include_ocr=args.ocr,
        min_area=args.min_area,
        use_openai=args.use_openai,
        model=args.model,
        detail=args.detail,
        provider_name=args.provider_name,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        api_mode=args.api_mode,
        openai_vision_proposals=args.openai_vision_proposals,
        vision_adapter=args.vision_adapter,
        vision_policy=args.vision_policy,
        document_kind=args.document_kind,
        corrections=args.corrections,
    )
    try:
        artifacts = run_extract(args.input, args.out, options)
    except Exception as exc:
        print(f"uiir extract failed: {exc}", file=sys.stderr)
        return 2

    print("UIIR extraction complete")
    for label, path in (
        ("output", artifacts.output_dir),
        ("composite", artifacts.composite),
        ("overlay", artifacts.overlay),
        ("layers", artifacts.layers_json),
        ("candidates", artifacts.candidates_json),
        ("json", artifacts.uiir_json),
        ("xml", artifacts.uiir_xml),
    ):
        print(f"{label}: {Path(path)}")
    return 0


def _compare_openai(args: argparse.Namespace) -> int:
    options = CompareOptions(
        model=args.model,
        detail=args.detail,
        limit=args.limit,
        prompt_version=args.prompt_version,
        include_ocr=args.ocr,
        min_area=args.min_area,
        provider_name=args.provider_name,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        api_mode=args.api_mode,
        openai_vision_proposals=args.openai_vision_proposals,
        vision_adapter=args.vision_adapter,
        vision_policy=args.vision_policy,
        document_kind=args.document_kind,
        golden_root=args.golden,
    )
    try:
        report = run_compare_openai(args.input, args.out, options)
    except Exception as exc:
        print(f"uiir compare-openai failed: {exc}", file=sys.stderr)
        return 2
    print("UIIR OpenAI comparison complete")
    print(f"status: {report['status']}")
    if report["status"] == "skipped":
        print(f"reason: {report.get('reason')}")
    else:
        print(f"baseline_schema_ok: {report['baseline']['schema_ok']}")
        print(f"openai_schema_ok: {report['openai']['schema_ok']}")
        print(f"baseline_avg_pixel_similarity: {report['baseline']['avg_pixel_similarity']}")
        print(f"openai_avg_pixel_similarity: {report['openai']['avg_pixel_similarity']}")
    print(f"comparison: {Path(args.out).expanduser().resolve() / 'comparison.json'}")
    return 0


def _iterate_openai(args: argparse.Namespace) -> int:
    options = IterateOptions(
        model=args.model,
        detail=args.detail,
        limit=args.limit,
        prompt_version=args.prompt_version,
        include_ocr=args.ocr,
        min_area=args.min_area,
        provider_name=args.provider_name,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        api_mode=args.api_mode,
        document_kind=args.document_kind,
        golden_root=args.golden,
        prompt_versions=_parse_csv(args.prompts) or (args.prompt_version,),
        policies=_parse_csv(args.policies) or ("audit", "strict", "balanced"),
    )
    try:
        report = run_iterate_openai(args.input, args.out, options)
    except Exception as exc:
        print(f"uiir iterate-openai failed: {exc}", file=sys.stderr)
        return 2
    print("UIIR OpenAI iteration complete")
    print(f"status: {report['status']}")
    if report["status"] == "skipped":
        print(f"reason: {report.get('reason')}")
    else:
        print(f"runs: {len(report.get('runs', []))}")
    print(f"leaderboard: {Path(args.out).expanduser().resolve() / 'leaderboard.json'}")
    return 0


def _review_run(args: argparse.Namespace) -> int:
    try:
        review = review_run(args.run)
    except Exception as exc:
        print(f"uiir review-run failed: {exc}", file=sys.stderr)
        return 2
    print("UIIR run review complete")
    print(f"findings: {review['finding_count']}")
    print(f"review: {Path(args.run).expanduser().resolve() / 'review.md'}")
    return 0


def _golden(args: argparse.Namespace) -> int:
    if args.golden_command == "build":
        try:
            manifest = build_golden_from_decisions(args.psd, args.run, args.decisions, args.out)
        except Exception as exc:
            print(f"uiir golden build failed: {exc}", file=sys.stderr)
            return 2
        print("UIIR golden build complete")
        print(f"output: {Path(args.out).expanduser().resolve()}")
        print(f"uiir_json: {manifest['uiir_json']}")
        print(f"uiir_xml: {manifest['uiir_xml']}")
        print(f"manifest: {Path(args.out).expanduser().resolve() / 'manifest.json'}")
        return 0
    return 1


def _fixtures(args: argparse.Namespace) -> int:
    if args.fixture_command == "list":
        print(json.dumps(list_fixture_sets(), ensure_ascii=False, indent=2))
        return 0
    if args.fixture_command == "download":
        try:
            manifest = download_fixture_set(args.set, args.out, limit=args.limit, overwrite=args.overwrite)
        except Exception as exc:
            print(f"uiir fixtures download failed: {exc}", file=sys.stderr)
            return 2
        print(f"Downloaded fixture set {manifest['set']} to {manifest['root']}")
        print(f"files: {manifest['count']}")
        for warning in manifest.get("warnings", []):
            print(f"warning: {warning}", file=sys.stderr)
        print(f"manifest: {Path(manifest['root']) / 'fixtures.manifest.json'}")
        return 0
    return 1


def _batch(args: argparse.Namespace) -> int:
    options = ExtractOptions(
        include_visual=not args.no_visual,
        include_ocr=args.ocr,
        min_area=args.min_area,
        use_openai=bool(args.use_openai),
        model=args.model,
        detail=args.detail,
        provider_name=args.provider_name,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        api_mode=args.api_mode,
        openai_vision_proposals=args.openai_vision_proposals,
        vision_adapter=args.vision_adapter,
        vision_policy=args.vision_policy,
        document_kind=args.document_kind,
        corrections=args.corrections,
    )
    try:
        report = run_batch(args.input, args.out, options, limit=args.limit)
    except Exception as exc:
        print(f"uiir batch failed: {exc}", file=sys.stderr)
        return 2
    print("UIIR batch complete")
    print(f"input: {report['input']}")
    print(f"output: {report['output']}")
    print(f"ok: {report['ok']}")
    print(f"skipped: {report.get('skipped', 0)}")
    print(f"failed: {report['failed']}")
    print(f"report: {Path(report['output']) / 'report.json'}")
    return 0 if report["failed"] == 0 else 1


def _evaluate(args: argparse.Namespace) -> int:
    try:
        report = evaluate_outputs(args.output, golden_root=args.golden, report_path=args.report)
    except Exception as exc:
        print(f"uiir evaluate failed: {exc}", file=sys.stderr)
        return 2
    print("UIIR evaluation complete")
    print(f"outputs: {report['count']}")
    print(f"schema_ok: {report['schema_ok']}")
    if report.get("avg_pixel_similarity") is not None:
        print(f"avg_pixel_similarity: {report['avg_pixel_similarity']}")
    if report.get("avg_bbox_iou") is not None:
        print(f"avg_bbox_iou: {report['avg_bbox_iou']}")
    print(f"report: {report['report']}")
    return 0


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value!r}")


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider-name", default="openai", help="Provider label for audit output. Defaults to openai.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable that contains the API key.")
    parser.add_argument(
        "--base-url",
        help=(
            "Optional OpenAI-compatible API base URL. If omitted, UIIR_OPENAI_BASE_URL "
            "or OPENAI_BASE_URL may be used."
        ),
    )
    parser.add_argument(
        "--api-mode",
        default="responses",
        choices=("responses", "chat-completions"),
        help="OpenAI-compatible API surface to use. Defaults to responses.",
    )


def _add_vision_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--openai-vision-proposals",
        action="store_true",
        help="Ask the vision model to propose missing UI candidates before semantic refinement.",
    )
    parser.add_argument(
        "--vision-adapter",
        default="openai",
        choices=("openai", "omniparser"),
        help="Vision proposal adapter. OmniParser is reserved for optional local integration.",
    )
    parser.add_argument(
        "--vision-policy",
        default="strict",
        choices=("audit", "strict", "balanced"),
        help="Vision proposal acceptance policy. Defaults to strict.",
    )


def _add_document_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--document-kind",
        default="auto",
        choices=("auto", "screen", "asset_sheet"),
        help="PSD intent classification. Defaults to auto.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
