from __future__ import annotations

import argparse
from pathlib import Path

from sda.agents.crew import SmartDocumentCrew
from sda.utils.logging import read_log_tail


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smart Document Analyst for scanned document images"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze",
        help="Analyze a scanned document image end-to-end",
    )
    analyze.add_argument(
        "--document",
        required=True,
        help="Path to a scanned document image: .png, .jpg, .jpeg, .tif, .tiff, .bmp",
    )
    analyze.add_argument("--model-path", default="models/document_cnn.pt")
    analyze.add_argument("--classes-path", default="models/classes.json")
    analyze.add_argument("--output-dir", default="outputs/reports")
    analyze.add_argument("--confidence-threshold", type=float, default=0.70)
    analyze.add_argument(
        "--no-llm",
        action="store_true",
        help="Use reproducible local tool execution instead of external LLM calls",
    )
    analyze.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve final report generation for demos/tests",
    )

    logs = sub.add_parser("logs", help="Show recent JSON action logs")
    logs.add_argument("--tail", type=int, default=20)
    logs.add_argument("--log-path", default="outputs/logs/agent_actions.jsonl")

    return parser


def validate_image_document(document_path: Path) -> None:
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")

    if not document_path.is_file():
        raise ValueError(f"Document path is not a file: {document_path}")

    if document_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Unsupported document format: {document_path.suffix}. "
            f"This project only supports scanned document images: {allowed}"
        )


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "logs":
        print(read_log_tail(Path(args.log_path), args.tail))
        return

    document_path = Path(args.document)
    validate_image_document(document_path)

    crew = SmartDocumentCrew(
        model_path=Path(args.model_path),
        classes_path=Path(args.classes_path),
        confidence_threshold=args.confidence_threshold,
        output_dir=Path(args.output_dir),
        use_llm=not args.no_llm,
        auto_approve=args.auto_approve,
    )

    report_path = crew.analyze(document_path)
    print(f"\nFinal report written to: {report_path}")


if __name__ == "__main__":
    main()