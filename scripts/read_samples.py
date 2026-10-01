"""
Step 5B: read every sample document with the real reader and save inspection copies.

Usage (from the project root):
    python -m scripts.read_samples
    python -m scripts.check_ingestion data/processed/docling
"""
import logging
import time

import config
from rag.reader import UnsupportedFormatError, read_document

OUTPUT_DIR = config.PROCESSED_DIR / "docling"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="       (log) %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in config.SAMPLES_DIR.iterdir() if path.is_file())
    print(f"Reading {len(files)} files from {config.SAMPLES_DIR}")
    print("The first run downloads Docling and OCR models; it can take several minutes.\n")
    print(f"{'File':50s} {'Pages':>5s} {'Chars':>7s} {'Sec':>6s} {'Removed':>7s}  Warnings")

    removed_report: dict[str, list[str]] = {}
    for path in files:
        started = time.perf_counter()
        try:
            document = read_document(path)
        except UnsupportedFormatError as exc:
            print(f"{path.name:50s} {'-':>5s} {'-':>7s} {'-':>6s} {'-':>7s}  skipped: {exc}")
            continue
        except Exception as exc:
            print(f"{path.name:50s} {'-':>5s} {'-':>7s} {'-':>6s} {'-':>7s}  ERROR: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started

        (OUTPUT_DIR / f"{path.name}.md").write_text(document.to_inspection_markdown(), encoding="utf-8")
        page_count = sum(page.number is not None for page in document.pages) or "-"
        print(f"{path.name:50s} {page_count!s:>5s} {len(document.text):>7d} {elapsed:>6.1f} "
              f"{len(document.removed_lines):>7d}  {'; '.join(document.warnings)}")
        if document.removed_lines:
            removed_report[path.name] = document.removed_lines

    if removed_report:
        print("\nLines removed as repeated page headers/footers:")
        for name, lines in removed_report.items():
            print(f"  {name}")
            for line in lines:
                print(f"     - {line}")

    print(f"\nOutputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()