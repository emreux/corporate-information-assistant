"""
Document metadata: who may see a chunk, whether it is current and how far its text can be trusted.

The rules come from the document catalog (document_catalog.toml), which the document owners
maintain. The facts come from the reader (file type, pages read with OCR). attach_metadata()
joins both with the chunks into records: one dictionary per chunk, stored next to its vector.
"""
import json
import logging
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import config
from rag.chunker import Chunk
from rag.reader import ParsedDocument

logger = logging.getLogger(__name__)

CATALOG_FIELDS = {"file", "access_level", "status", "effective_date", "replaced_by"}


class CatalogError(ValueError):
    """The document catalog is missing, malformed or contains an invalid value."""


@dataclass(frozen=True)
class CatalogEntry:
    access_level: str
    status: str = "active"
    effective_date: date | None = None
    replaced_by: str | None = None


# Used for files that are missing from the catalog: nobody has decided who may see them yet.
UNCATALOGED = CatalogEntry(access_level=config.DEFAULT_ACCESS_LEVEL)


# --- Catalog ------------------------------------------------------------------------

def load_catalog(path: Path = config.DOCUMENT_CATALOG) -> dict[str, CatalogEntry]:
    """Read and validate the catalog. Any mistake stops the program, because labels guard access."""
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError as exc:
        raise CatalogError(f"Document catalog not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise CatalogError(f"{path.name} is not valid TOML: {exc}") from exc

    documents = data.get("document", [])
    if not isinstance(documents, list):
        raise CatalogError(f"{path.name}: write every entry under [[document]] (two brackets).")

    catalog: dict[str, CatalogEntry] = {}
    for number, item in enumerate(documents, start=1):
        name = item.get("file")
        if not name:
            raise CatalogError(f"Document #{number} in {path.name} has no 'file'.")
        if name in catalog:
            raise CatalogError(f"'{name}' appears more than once in {path.name}.")
        catalog[name] = _parse_entry(item, f"'{name}' in {path.name}")

    for name, entry in catalog.items():
        if entry.replaced_by and entry.replaced_by not in catalog:
            raise CatalogError(f"'{name}' is replaced by '{entry.replaced_by}', which is not in the catalog.")
    return catalog


def _parse_entry(item: dict, where: str) -> CatalogEntry:
    unknown = set(item) - CATALOG_FIELDS
    if unknown:                                   # e.g. "acess_level": a typo must not be ignored
        raise CatalogError(f"{where}: unknown field(s) {sorted(unknown)}. Known: {sorted(CATALOG_FIELDS)}")

    access_level = item.get("access_level")
    if access_level not in config.ACCESS_LEVELS:
        raise CatalogError(f"{where}: access_level must be one of {config.ACCESS_LEVELS}, not {access_level!r}.")

    status = item.get("status", "active")
    if status not in config.DOCUMENT_STATUSES:
        raise CatalogError(f"{where}: status must be one of {config.DOCUMENT_STATUSES}, not {status!r}.")

    effective_date = item.get("effective_date")
    if effective_date is not None and type(effective_date) is not date:
        raise CatalogError(f"{where}: effective_date must be a date without quotes, like 2026-01-01.")

    replaced_by = item.get("replaced_by")
    if (status == "superseded") != bool(replaced_by):
        raise CatalogError(f"{where}: a superseded document needs replaced_by, and only a superseded one may have it.")

    return CatalogEntry(access_level, status, effective_date, replaced_by)


def find_entry(catalog: dict[str, CatalogEntry], source: str) -> CatalogEntry:
    """The catalog entry of a file, or the restrictive default when nobody has labelled it yet."""
    entry = catalog.get(source)
    if entry is None:
        logger.warning("'%s' is not in the document catalog; it is treated as '%s' until someone labels it.",
                       source, config.DEFAULT_ACCESS_LEVEL)
        return UNCATALOGED
    return entry


def add_catalog_entry(file_name: str, access_level: str, path: Path = config.DOCUMENT_CATALOG) -> None:
    """
    Append an active entry for a new file to the catalog (used by the upload in the web interface).

    The catalog is loaded again afterwards; if it is no longer valid, the original is restored,
    so a failed upload never leaves a broken catalog behind.
    """
    if access_level not in config.ACCESS_LEVELS:
        raise CatalogError(f"access_level must be one of {config.ACCESS_LEVELS}, not {access_level!r}.")
    if file_name in load_catalog(path):
        raise CatalogError(f"'{file_name}' is already in the catalog.")

    original = path.read_text(encoding="utf-8")
    entry = (f"\n# Added from the web interface on {date.today().isoformat()}\n"
             f"[[document]]\nfile = {json.dumps(file_name, ensure_ascii=False)}\n"   # a JSON string is a valid TOML string
             f'access_level = "{access_level}"\n')
    path.write_text(original.rstrip("\n") + "\n" + entry, encoding="utf-8")
    try:
        load_catalog(path)
    except CatalogError:
        path.write_text(original, encoding="utf-8")
        raise


def missing_files(catalog: dict[str, CatalogEntry], folder: Path) -> list[str]:
    """Catalog entries whose file is not in the folder: the catalog is out of date."""
    return sorted(name for name in catalog if not (folder / name).is_file())


# --- Joining chunks with metadata ---------------------------------------------------

def attach_metadata(document: ParsedDocument, chunks: list[Chunk],
                    catalog: dict[str, CatalogEntry]) -> list[dict]:
    """
    Join every chunk with the catalog entry of its document and the OCR flags of its pages.

    Document fields are the same for all chunks of a file, so they are built once. The result
    has one dictionary per chunk: "text" goes to the embedding model, and the whole dictionary
    is stored as the payload next to the vector (step 9).
    """
    entry = find_entry(catalog, document.source)
    document_fields = {
        "source": document.source,
        "file_type": document.file_type,
        "access_level": entry.access_level,
        "status": entry.status,
        "effective_date": entry.effective_date.isoformat() if entry.effective_date else None,
        "replaced_by": entry.replaced_by,
    }
    ocr_pages = {page.number for page in document.pages if page.ocr}

    records = []
    for chunk in chunks:
        pages = range(chunk.page_start, chunk.page_end + 1) if chunk.page_start is not None else range(0)
        records.append({
            "text": chunk.text,
            **document_fields,
            "heading_path": chunk.heading_path,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "ocr": any(page in ocr_pages for page in pages),
        })
    return records