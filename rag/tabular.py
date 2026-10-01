"""
Reader for tabular files: Excel workbooks (.xlsx) and CSV files.

Spreadsheets have no page layout for Docling to analyse, and converting them through Docling
loses what matters here: sheet names, number formats and merged cells. This module reads the
cells directly and writes every table row as a self-contained record,

    Column name: value | Column name: value | ...

so that a row keeps its meaning even when a long table is split across several chunks.
"""
import csv
import io
import logging
import re
import zipfile
from datetime import date, datetime, time
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

import config

logger = logging.getLogger(__name__)

QUOTED_TEXT = re.compile(r'"([^"]*)"')          # literal text in a number format, e.g. "TL"
BRACKETED = re.compile(r"\[([^\]]*)\]")         # [Red], [>=100], [$₺-41F]
DECIMAL_PLACES = re.compile(r"\.([0#?]+)")
SEPARATORS = str.maketrans({",": config.THOUSANDS_SEPARATOR, ".": config.DECIMAL_SEPARATOR})
RECORD_SEPARATOR = " | "


class TabularReadError(ValueError):
    """The file has a tabular extension but cannot be opened or decoded."""


# --- Public API ---------------------------------------------------------------------

def read_excel(path: Path) -> list[str]:
    """Return one Markdown string per worksheet, in workbook order ('' for an empty sheet)."""
    try:
        # data_only=True: read the value Excel last calculated, not the formula text.
        workbook = load_workbook(path, data_only=True)
    except (zipfile.BadZipFile, InvalidFileException, KeyError) as exc:
        raise TabularReadError(
            f"'{path.name}' could not be opened as an Excel workbook. "
            "It may be password-protected, corrupted or saved in another format."
        ) from exc

    sheets = []
    for sheet in workbook.worksheets:
        if sheet.sheet_state != "visible":
            logger.warning("Sheet '%s' in '%s' is hidden in Excel but will be indexed.", sheet.title, path.name)
        rows = [[format_cell(cell) for cell in row] for row in sheet.iter_rows()]
        sheets.append(rows_to_markdown(sheet.title, rows))
    return sheets


def read_csv(path: Path) -> str:
    """Return a CSV file as Markdown records. No heading: the file name is already the source."""
    text = _decode(path.read_bytes(), path.name)
    sample = "\n".join(text.splitlines()[:config.CSV_SNIFF_LINES])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=config.CSV_DELIMITERS)
    except csv.Error:
        logger.warning("Could not detect the delimiter of '%s'; assuming a comma.", path.name)
        dialect = csv.excel
    rows = [[_clean(value) for value in row] for row in csv.reader(io.StringIO(text), dialect)]
    return rows_to_markdown(None, rows)


def rows_to_markdown(title: str | None, rows: list[list[str]]) -> str:
    """
    Turn a grid of cell texts into Markdown.

    Rows with a single filled cell are titles, notices or notes and become paragraphs. The first
    row with at least two filled cells is the column header; every row after it becomes a record.
    """
    blocks: list[str] = []
    header: list[str] | None = None
    for row in rows:
        filled = [(index, value) for index, value in enumerate(row) if value]
        if not filled:
            continue
        if len(filled) == 1:
            blocks.append(filled[0][1])
        elif header is None:
            header = row
        else:
            blocks.append(_to_record(header, filled))
    if not blocks:
        return ""
    if title:
        blocks.insert(0, f"## {title}")
    return "\n\n".join(blocks)   # blank lines keep every record a separate block for chunking


# --- Cell formatting ----------------------------------------------------------------

def format_cell(cell) -> str:
    """Write a cell the way Excel shows it: 85000 with the format '#,##0 "TL"' becomes '85.000 TL'."""
    value = cell.value
    if value is None:
        return ""
    if isinstance(value, datetime):
        has_time = value.time() != time(0)
        return value.strftime(config.DATETIME_FORMAT if has_time else config.DATE_FORMAT)
    if isinstance(value, date):
        return value.strftime(config.DATE_FORMAT)
    if isinstance(value, time):
        return value.strftime("%H:%M")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_number(value, cell.number_format or "General")
    return _clean(str(value))


def format_number(value: float, number_format: str) -> str:
    """Apply the parts of an Excel number format that change how a number reads."""
    section = number_format.split(";")[0]          # the first section is for positive numbers
    if section.strip().lower() in ("general", "@", ""):
        return _plain_number(value)

    literals = [text.strip() for text in QUOTED_TEXT.findall(section)]
    literals += [code[1:].split("-")[0] for code in BRACKETED.findall(section) if code.startswith("$")]
    pattern = BRACKETED.sub("", QUOTED_TEXT.sub("", section))

    is_percent = "%" in pattern
    if is_percent:
        value *= 100
    match = DECIMAL_PLACES.search(pattern)
    decimals = len(match.group(1)) if match else 0
    grouping = "," if "," in pattern else ""       # "#,##0" shows thousands separators, "0" does not

    text = format(abs(value), f"{grouping}.{decimals}f").translate(SEPARATORS)
    if is_percent:
        text = "%" + text                          # Turkish style: %15
    if value < 0:
        text = "-" + text
    suffix = " ".join(literal for literal in literals if literal)
    return f"{text} {suffix}" if suffix else text


def _plain_number(value: float) -> str:
    """'General' format: no thousands separator, so years and IDs (2026, 1001) stay intact."""
    if float(value).is_integer():
        return str(int(value))
    return format(value, ".10g").translate(SEPARATORS)


# --- Helpers ------------------------------------------------------------------------

def _to_record(header: list[str], filled: list[tuple[int, str]]) -> str:
    fields = []
    for index, value in filled:
        name = header[index] if index < len(header) and header[index] else f"Column {index + 1}"
        fields.append(f"{name}: {value}")
    return RECORD_SEPARATOR.join(fields)


def _clean(text: str) -> str:
    """Collapse line breaks inside a cell, so one row always stays on one line."""
    return " ".join(text.split())


def _decode(data: bytes, name: str) -> str:
    for encoding in config.CSV_ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding != config.CSV_ENCODINGS[0]:
            logger.warning("'%s' is not UTF-8; it was decoded as %s.", name, encoding)
        return text
    raise TabularReadError(f"'{name}' could not be decoded with any of {config.CSV_ENCODINGS}.")