"""
Step 8: check the document catalog and the metadata attached to every chunk.

Part 1 feeds small, deliberately broken catalogs to load_catalog(): every mistake must stop it.
Part 2 checks the real catalog against the documents folder.
Part 3 reads, chunks and labels every sample document, checks the labels against the answer key
(old version, confidential file, scanned file) and prints one record as an example.

Usage (from the project root):
    python -m scripts.check_metadata
"""
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import config
from rag.chunker import chunk_document
from rag.metadata import CatalogError, attach_metadata, load_catalog, missing_files
from rag.reader import DocumentPage, ParsedDocument, read_document_cached as read_document  # step 9
from scripts.check_env import run_check

EXPECTED_FIELDS = {
    "text", "source", "file_type", "access_level", "status", "effective_date", "replaced_by",
    "heading_path", "page_start", "page_end", "chunk_index", "token_count", "ocr",
}
OLD_POLICY, NEW_POLICY = "IK_Politikasi_2025.pdf", "IK_Politikasi_2026.pdf"
SALARY_SCALE = "Yonetici_Maas_Skalasi_2026.xlsx"
SCANNED_MINUTES = "Yonetim_Kurulu_Tutanagi_2024-09_TARANMIS.pdf"

# Catalogs that load_catalog() must reject: (description, TOML text).
VALID_ENTRY = '[[document]]\nfile = "a.pdf"\naccess_level = "internal"\n'
BROKEN_CATALOGS = [
    ("a typo in a value", '[[document]]\nfile = "a.pdf"\naccess_level = "interal"\n'),
    ("a typo in a field name", '[[document]]\nfile = "a.pdf"\nacess_level = "internal"\n'),
    ("a missing access level", '[[document]]\nfile = "a.pdf"\n'),
    ("a date in quotes", VALID_ENTRY + 'effective_date = "2026-01-01"\n'),
    ("superseded without replaced_by", VALID_ENTRY + 'status = "superseded"\n'),
    ("replaced_by pointing to an unknown file", VALID_ENTRY + 'status = "superseded"\nreplaced_by = "b.pdf"\n'),
    ("the same file twice", VALID_ENTRY + VALID_ENTRY),
    ("single brackets", VALID_ENTRY.replace("[[document]]", "[document]")),
]


# --- Part 1: made-up catalogs -------------------------------------------------------

def expect_rejected(text: str, folder: Path) -> str:
    path = folder / "catalog.toml"
    path.write_text(text, encoding="utf-8")
    try:
        load_catalog(path)
    except CatalogError as exc:
        return f"rejected: {exc}"
    raise AssertionError("the broken catalog was accepted")


def check_uncataloged_file() -> str:
    """A file nobody has labelled yet must get the restrictive default (and a warning in the log)."""
    document = ParsedDocument(source="new_file.pdf", file_type="pdf",
                              pages=[DocumentPage(1, "## 1. Konu\n\nKısa bir metin.", ocr=True)])
    record = attach_metadata(document, chunk_document(document), catalog={})[0]
    if record["access_level"] != "restricted" or not record["ocr"]:     # the policy, not the setting
        raise AssertionError(f"Unexpected record: {record}")
    return f"access_level={record['access_level']!r}, ocr={record['ocr']}"


# --- Part 2: the real catalog -------------------------------------------------------

def check_no_stale_entries(catalog) -> str:
    stale = missing_files(catalog, config.SAMPLES_DIR)
    if stale:
        raise AssertionError(f"in the catalog but not in the folder: {stale}")
    return f"all {len(catalog)} entries have a file"


def check_all_catalogued(catalog) -> str:
    names = sorted(path.name for path in config.SAMPLES_DIR.iterdir() if path.is_file())
    unlabelled = [name for name in names if name not in catalog]
    if unlabelled:
        raise AssertionError(f"not in the catalog, so treated as restricted: {unlabelled}")
    return f"all {len(names)} files are labelled"


# --- Part 3: sample documents -------------------------------------------------------

def require(records: list[dict], condition, what: str) -> str:
    """Every record must meet the condition."""
    if not records:
        raise AssertionError("no records: was the file read?")
    failing = [(record["source"], record["chunk_index"]) for record in records if not condition(record)]
    if failing:
        raise AssertionError(f"not {what}: {failing[:5]}")
    return f"{len(records)} chunks {what}"


def label_samples(catalog) -> dict[str, list[dict]]:
    """Read, chunk and label every sample document; print one summary line per file."""
    files = sorted(path for path in config.SAMPLES_DIR.iterdir() if path.is_file())
    print(f"\nReading {len(files)} files (from the cache when possible)\n")
    print(f"{'File':50s} {'Access':>10s} {'Status':>10s} {'Chunks':>6s} {'OCR':>4s} {'Sec':>6s}")
    records_by_file: dict[str, list[dict]] = {}
    for path in files:
        started = time.perf_counter()
        try:
            document = read_document(path)
        except Exception as exc:
            print(f"{path.name:50s} ERROR: {type(exc).__name__}: {exc}")
            continue
        records = attach_metadata(document, chunk_document(document), catalog)
        records_by_file[path.name] = records
        first = records[0] if records else {"access_level": "-", "status": "-"}
        print(f"{path.name:50s} {first['access_level']:>10s} {first['status']:>10s} {len(records):>6d} "
              f"{sum(record['ocr'] for record in records):>4d} {time.perf_counter() - started:>6.1f}")
    return records_by_file


def show_example(records_by_file: dict[str, list[dict]]) -> None:
    """Print the record of the old leave table: what will be stored next to its vector."""
    for record in records_by_file.get(OLD_POLICY, []):
        if "4.1" in record["heading_path"]:
            shown = dict(record, text=record["text"][:110] + " ...")
            print("\nExample record (the payload stored next to the vector in step 9):")
            print(json.dumps(shown, ensure_ascii=False, indent=2))
            return


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="       (log) %(message)s")
    results: list[bool] = []

    print("Part 1: made-up catalogs")
    with tempfile.TemporaryDirectory() as tmp:
        for description, text in BROKEN_CATALOGS:
            results.append(run_check(f"Catalog with {description}",
                                     lambda text=text: expect_rejected(text, Path(tmp))))
    results.append(run_check("File missing from the catalog", check_uncataloged_file))

    print("\nPart 2: the real catalog")
    try:
        catalog = load_catalog()
    except CatalogError as exc:
        print(f"[FAIL] {config.DOCUMENT_CATALOG.name}: {exc}")
        sys.exit(1)
    results.append(run_check("Catalog loads", lambda: f"{len(catalog)} documents"))
    results.append(run_check("Every catalog entry has a file", lambda: check_no_stale_entries(catalog)))
    results.append(run_check("Every sample file is in the catalog", lambda: check_all_catalogued(catalog)))

    print("\nPart 3: sample documents")
    by_file = label_samples(catalog)
    everything = [record for records in by_file.values() for record in records]

    def others(name: str) -> list[dict]:
        return [record for record in everything if record["source"] != name]

    checks = {
        "Every record has the same fields": lambda: require(
            everything, lambda r: set(r) == EXPECTED_FIELDS, "complete"),
        "T1: old HR policy is superseded": lambda: require(
            by_file.get(OLD_POLICY, []),
            lambda r: r["status"] == "superseded" and r["replaced_by"] == NEW_POLICY, f"replaced by {NEW_POLICY}"),
        "T1: new HR policy is active": lambda: require(
            by_file.get(NEW_POLICY, []), lambda r: r["status"] == "active", "active"),
        "T16: salary scale is restricted": lambda: require(
            by_file.get(SALARY_SCALE, []), lambda r: r["access_level"] == "restricted", "restricted"),
        "T16: nothing else is restricted": lambda: require(
            others(SALARY_SCALE), lambda r: r["access_level"] == "internal", "internal"),
        "T14: scanned minutes are marked as OCR": lambda: require(
            by_file.get(SCANNED_MINUTES, []), lambda r: r["ocr"], "marked as OCR"),
        "T14: no other chunk is marked as OCR": lambda: require(
            others(SCANNED_MINUTES), lambda r: not r["ocr"], "free of OCR"),
    }
    print()
    results += [run_check(name, check) for name, check in checks.items()]

    show_example(by_file)
    print(f"\nTOTAL: {sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()