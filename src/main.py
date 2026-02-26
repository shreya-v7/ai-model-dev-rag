"""CLI entrypoint for running the RAG skeleton."""

from __future__ import annotations

import argparse
import json

from src.pipeline import run_rag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RAG synthesis over up to 10 documents."
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Research question or synthesis topic.",
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        required=True,
        help="Document paths (pdf/txt/md/rst/json/csv), max 10.",
    )
    parser.add_argument(
        "--output",
        default="outputs/synthesis_result.json",
        help="Path for JSON result output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_rag(
        topic=args.topic,
        document_paths=args.docs,
        output_json_path=args.output,
    )
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
