from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dynamics_examples import (  # noqa: E402
    DEFAULT_TEST_PATHS,
    build_catalog_report,
    build_markdown_report,
    build_test_dynamics_examples,
    evaluate_examples_against_programs,
    load_manual_transition_examples,
    load_program_sources,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an atomic dynamics example catalog from optional Baba source files and "
            "evaluate which program versions reproduce each transition."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment root directory containing `program_versions/`.",
    )
    parser.add_argument(
        "--program-dir",
        type=Path,
        default=None,
        help="Directory containing `v*.py` program versions.",
    )
    parser.add_argument(
        "--program-file",
        dest="program_files",
        type=Path,
        action="append",
        default=[],
        help="Additional program file to evaluate. Can be repeated.",
    )
    parser.add_argument(
        "--manual-json",
        dest="manual_jsons",
        type=Path,
        action="append",
        default=[],
        help="Manual transition JSON artifact to include as an extra example. Can be repeated.",
    )
    parser.add_argument(
        "--skip-test-catalog",
        action="store_true",
        help="Do not include the optional built-in atomic catalog.",
    )
    parser.add_argument(
        "--test-file",
        dest="test_files",
        type=Path,
        action="append",
        default=[],
        help="Add a specific Python source file to mine dynamics examples from.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write the machine-readable report JSON here.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Write the human-readable Markdown report here.",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Build the atomic dynamics catalog without evaluating any program versions.",
    )
    return parser.parse_args()


def _resolve_output_path(
    *,
    provided: Path | None,
    experiment_dir: Path | None,
    default_name: str,
) -> Path | None:
    if provided is not None:
        return provided
    if experiment_dir is None:
        return None
    return experiment_dir / "results" / default_name


def main() -> int:
    args = _parse_args()

    if (
        not args.catalog_only
        and args.experiment_dir is None
        and args.program_dir is None
        and len(args.program_files) == 0
    ):
        raise ValueError("Provide at least one of --experiment-dir, --program-dir, or --program-file.")

    examples = []
    if not args.skip_test_catalog:
        test_paths: List[Path] = list(DEFAULT_TEST_PATHS)
        if len(args.test_files) > 0:
            test_paths = list(args.test_files)
        examples.extend(build_test_dynamics_examples(test_paths=test_paths))

    if len(args.manual_jsons) > 0:
        examples.extend(load_manual_transition_examples(args.manual_jsons))

    if len(examples) == 0:
        raise ValueError("No dynamics examples were built. Add --manual-json or --test-file.")

    if args.catalog_only:
        report = build_catalog_report(examples=examples)
    else:
        program_sources = load_program_sources(
            experiment_dir=args.experiment_dir,
            program_dir=args.program_dir,
            program_files=args.program_files,
        )
        report = evaluate_examples_against_programs(
            examples=examples,
            program_sources=program_sources,
        )

    output_json = _resolve_output_path(
        provided=args.output_json,
        experiment_dir=args.experiment_dir,
        default_name="dynamics_coverage.json",
    )
    output_markdown = _resolve_output_path(
        provided=args.output_markdown,
        experiment_dir=args.experiment_dir,
        default_name="dynamics_coverage.md",
    )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    markdown = build_markdown_report(report)
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(markdown, encoding="utf-8")

    print(f"Examples: {len(report.get('examples') or [])}")
    print(f"Programs: {len(report.get('versions') or [])}")
    if output_json is not None:
        print(f"JSON report: {output_json}")
    if output_markdown is not None:
        print(f"Markdown report: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
