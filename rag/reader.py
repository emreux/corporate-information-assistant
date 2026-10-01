"""
Document reader: turns any supported file into clean Markdown, page by page.

Docling does the heavy lifting for documents with a layout (layout analysis, tables, headings,
OCR). Spreadsheets and CSV files are read cell by cell in rag/tabular.py. This module adds what
the naive readers of step 5A were missing: page numbers, speaker notes, removal of repeated
page headers/footers and website boilerplate, and loud warnings instead of silent failures.
"""
import hashlib
import html
import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from bs4 import BeautifulSoup
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ContentLayer

import config
from rag.tabular import read_csv, read_excel

logger = logging.getLogger(__name__)

DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md"}
TABULAR_EXTENSIONS = {".xlsx", ".csv"}
PLAIN_TEXT_EXTENSIONS = {".txt"}
SUPPORTED_EXTENSIONS = DOCLING_EXTENSIONS | TABULAR_EXTENSIONS | PLAIN_TEXT_EXTENSIONS
LEGACY_EXTENSIONS = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}

# BODY is the main content; NOTES holds speaker notes of slides. FURNITURE (page headers and
# footers detected by Docling's layout model) is deliberately left out.
EXPORT_LAYERS = {ContentLayer.BODY, ContentLayer.NOTES}

PAGE_NUMBER_LINE = re.compile(r"^\s*(sayfa|page)\s*\d+(\s*(/|of)\s*\d+)?\s*$", re.IGNORECASE)
HTML_NOISE_TAGS = ["nav", "footer", "aside", "script", "style", "noscript", "form"]
HTML_NOISE_ATTRIBUTE = re.compile(r"cookie|consent|banner|breadcrumb|navbar|sidebar", re.IGNORECASE)


class UnsupportedFormatError(ValueError):
    """The file type cannot be read by CIA."""


@dataclass
class DocumentPage:
    number: int | None          # page, slide or sheet number; None for formats without pages
    markdown: str
    ocr: bool = False           # True when the page had no text layer and was read with OCR


@dataclass
class ParsedDocument:
    source: str
    file_type: str
    pages: list[DocumentPage]
    warnings: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The whole document as one Markdown string."""
        return "\n\n".join(page.markdown for page in self.pages if page.markdown)

    def to_inspection_markdown(self) -> str:
        """Markdown with page markers, saved to disk for humans and for the ingestion check."""
        parts = [f"<!-- source: {self.source} | type: {self.file_type} -->"]
        parts += [f"<!-- warning: {warning} -->" for warning in self.warnings]
        for page in self.pages:
            if page.number is not None:
                parts.append(f"<!-- page {page.number} -->")
            parts.append(page.markdown)
        return "\n\n".join(parts) + "\n"


# --- Public API ---------------------------------------------------------------------

def check_supported(path: Path) -> None:
    """Raise UnsupportedFormatError with a helpful message if the file type cannot be read."""
    suffix = path.suffix.lower()
    if suffix in LEGACY_EXTENSIONS:
        raise UnsupportedFormatError(
            f"'{path.name}' uses the legacy {suffix} format. "
            f"Save it as {LEGACY_EXTENSIONS[suffix]} and upload it again."
        )
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"'{path.name}' cannot be read ({suffix or 'no extension'}). "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def read_document(path: Path) -> ParsedDocument:
    """Read a file of any supported format into cleaned, page-aware Markdown."""
    path = Path(path)
    check_supported(path)
    suffix = path.suffix.lower()

    if suffix in TABULAR_EXTENSIONS:
        pages, removed = _read_tabular(path, suffix), set()
    elif suffix in PLAIN_TEXT_EXTENSIONS:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        pages, removed = remove_repeated_lines([DocumentPage(None, text)])
    else:
        pages, removed = remove_repeated_lines(_read_with_docling(path, suffix))

    if suffix == ".pdf":
        ocr_pages = pages_without_text_layer(path)
        pages = [replace(page, ocr=page.number in ocr_pages) for page in pages]

    document = ParsedDocument(
        source=path.name,
        file_type=suffix.lstrip("."),
        pages=pages,
        removed_lines=sorted(removed),
    )
    ocr_count = sum(page.ocr for page in pages)
    if ocr_count:
        document.warnings.append(
            f"{ocr_count} of {len(pages)} pages had no text layer and were read with OCR; "
            "the text may contain recognition errors."
        )
    if len(document.text) < config.MIN_EXPECTED_CHARS_PER_DOCUMENT:
        document.warnings.append(
            "Almost no text was extracted. The file may be scanned, empty or password-protected."
        )
    return document


def read_document_cached(path: Path) -> ParsedDocument:
    """
    Like read_document, but reuse the result of an earlier read of the same file content.

    The cache key is the SHA-256 of the file plus PIPELINE_VERSION, so a changed file or a
    changed reader never gets a stale result. The cache holds the full text of every document:
    it needs the same protection as the documents themselves.
    """
    path = Path(path)
    target = cache_path(path)
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            data["pages"] = [DocumentPage(**page) for page in data["pages"]]
            return replace(ParsedDocument(**data), source=path.name)   # same content may have a new name
        except (ValueError, TypeError, KeyError) as exc:
            logger.warning("Ignoring a damaged cache file %s (%s); reading the document again.", target.name, exc)

    document = read_document(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")               # write, then rename: never a half-written file
    temporary.write_text(json.dumps(asdict(document), ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)
    return document


def cache_path(path: Path) -> Path:
    """Where the cached reading of this file content is (or would be) stored."""
    return config.CACHE_DIR / f"{file_sha256(path)}_v{config.PIPELINE_VERSION}.json"


def file_sha256(path: Path) -> str:
    """Fingerprint of the file content: changes whenever a single byte changes."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pages_without_text_layer(path: Path) -> set[int]:
    """Numbers of the PDF pages that are only images (scanned), so their text came from OCR."""
    pdf = pdfium.PdfDocument(str(path))
    try:
        return {
            index + 1 for index in range(len(pdf))
            if pdf[index].get_textpage().count_chars() < config.OCR_TEXT_LAYER_MIN_CHARS
        }
    finally:
        pdf.close()


# --- Tabular files ------------------------------------------------------------------

def _read_tabular(path: Path, suffix: str) -> list[DocumentPage]:
    """
    Excel sheets become pages; a CSV file is a single page without a number.

    Header/footer cleanup is skipped on purpose: every line here is data, and similar rows on
    different sheets (monthly sheets, for example) must not be mistaken for page footers.
    """
    if suffix == ".csv":
        return [DocumentPage(None, read_csv(path))]
    return [DocumentPage(number, markdown) for number, markdown in enumerate(read_excel(path), start=1)]


# --- Docling ------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_converter() -> DocumentConverter:
    """Build Docling once: loading its layout, table and OCR models takes several seconds."""
    pdf_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=EasyOcrOptions(lang=config.OCR_LANGUAGES),
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


def _read_with_docling(path: Path, suffix: str) -> list[DocumentPage]:
    if suffix in {".html", ".htm"}:
        cleaned = clean_html(path.read_text(encoding="utf-8", errors="replace"))
        source = DocumentStream(name=path.name, stream=BytesIO(cleaned.encode("utf-8")))
    else:
        source = path

    document = get_converter().convert(source).document
    page_numbers = sorted(document.pages)
    if not page_numbers:                        # Word, HTML: no page information
        return [DocumentPage(None, _export_markdown(document))]
    return [DocumentPage(number, _export_markdown(document, number)) for number in page_numbers]


def _export_markdown(document, page_no: int | None = None) -> str:
    markdown = document.export_to_markdown(
        page_no=page_no,
        included_content_layers=EXPORT_LAYERS,
        image_placeholder="",           # no "<!-- image -->" noise in the text
        escape_underscores=False,       # keep snake_case names and codes as written
    )
    markdown = html.unescape(markdown)  # "&gt;" -> ">" so menu paths read naturally
    return re.sub(r"\n{3,}", "\n\n", markdown).strip()


# --- Cleaning -----------------------------------------------------------------------

def clean_html(raw_html: str) -> str:
    """Keep the main content of a web page; drop menus, footers, cookie banners and scripts."""
    soup = BeautifulSoup(raw_html, "html.parser")
    root = soup.find("main") or soup.find("article") or soup.body or soup

    noise = root.find_all(HTML_NOISE_TAGS)
    noise += root.find_all(attrs={"class": HTML_NOISE_ATTRIBUTE})
    noise += root.find_all(attrs={"id": HTML_NOISE_ATTRIBUTE})
    for tag in noise:
        if not tag.decomposed:
            tag.decompose()

    return str(root) if root is soup else f"<html><body>{root}</body></html>"


def remove_repeated_lines(pages: list[DocumentPage]) -> tuple[list[DocumentPage], set[str]]:
    """
    Remove page headers/footers that survived layout analysis.

    A short line that appears on at least REPEATED_LINE_MIN_SHARE of the pages (digits ignored,
    so 'Page 1' and 'Page 2' count as the same line) is treated as a header or footer, as are
    bare page-number lines. Headings and table rows are never removed.
    """
    numbered = [page for page in pages if page.number is not None]
    repeated: set[str] = set()
    if len(numbered) >= 2:
        counts: Counter[str] = Counter()
        for page in numbered:
            counts.update({_line_key(line) for line in page.markdown.splitlines() if _is_candidate(line)})
        min_pages = max(2, math.ceil(len(numbered) * config.REPEATED_LINE_MIN_SHARE))
        repeated = {key for key, count in counts.items() if count >= min_pages}

    cleaned_pages: list[DocumentPage] = []
    removed: set[str] = set()
    for page in pages:
        kept = []
        for line in page.markdown.splitlines():
            is_header_footer = _is_candidate(line) and _line_key(line) in repeated
            if is_header_footer or PAGE_NUMBER_LINE.match(line):
                removed.add(line.strip())
                continue
            kept.append(line)
        markdown = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
        cleaned_pages.append(replace(page, markdown=markdown))
    return cleaned_pages, removed


def _is_candidate(line: str) -> bool:
    """Only short, non-heading, non-table lines can be headers or footers."""
    stripped = line.strip()
    return (
        bool(stripped)
        and len(stripped) <= config.REPEATED_LINE_MAX_LENGTH
        and not stripped.startswith(("#", "|"))
    )


def _line_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())