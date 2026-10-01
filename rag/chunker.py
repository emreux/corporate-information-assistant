"""
Chunker: splits a ParsedDocument into retrieval-sized chunks.

It works on the common Markdown format produced by rag/reader.py, so it never needs to know
the original file type:

  1. Split every page into blocks (heading, paragraph, list, table), remembering the page.
  2. Follow the headings with a stack to know the heading path of every block.
  3. Pack the blocks of each section into chunks of about CHUNK_SIZE_TOKENS. A block that is
     too big on its own is split: tables by rows (header repeated), lists by items and
     paragraphs by sentences, with a small overlap.
  4. Start every chunk with "[source > heading path]" so that it can be understood on its own.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, TypeVar

import config
from rag.reader import DocumentPage, ParsedDocument

T = TypeVar("T")

HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SECTION_NUMBER = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+\S")   # "5.", "5.1", "12.3.1"; not "2025"
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
TABLE_SEPARATOR = re.compile(r"^\|[\s:|-]+\|$")
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
PATH_SEPARATOR = " > "


@dataclass
class Chunk:
    text: str                  # context line + content: this is what gets embedded
    source: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    chunk_index: int
    token_count: int


@dataclass
class Block:
    kind: str                  # "heading", "paragraph", "list" or "table"
    text: str
    page: int | None
    level: int = 0             # number of '#' characters; headings only


@dataclass
class Section:
    path: list[str]
    heading_page: int | None
    blocks: list[Block] = field(default_factory=list)


# --- Public API ---------------------------------------------------------------------

def chunk_document(document: ParsedDocument) -> list[Chunk]:
    """Split a document into chunks, each with its heading path and page range."""
    sections = build_sections(split_blocks(document.pages))
    chunks = [chunk for section in sections for chunk in _chunk_section(document.source, section)]
    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
    return chunks


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load the embedding model's tokenizer (about 20 MB, downloaded once, no model weights)."""
    from transformers import AutoTokenizer    # imported here: loading transformers takes a few seconds
    return AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    """Count tokens exactly as the embedding model on the model server will see them."""
    return len(get_tokenizer().encode(text, add_special_tokens=False))


# --- Step 1: blocks -----------------------------------------------------------------

def split_blocks(pages: list[DocumentPage]) -> list[Block]:
    """Cut every page at blank lines and label each piece. A heading always gets its own block."""
    blocks: list[Block] = []
    for page in pages:
        for raw in re.split(r"\n\s*\n", page.markdown):
            text = raw.strip()
            if not text:
                continue
            first_line, _, rest = text.partition("\n")
            heading = HEADING_LINE.match(first_line)
            if heading:
                blocks.append(_heading_block(heading, page.number))
                text = rest.strip()
                if not text:
                    continue
            blocks.append(Block(_block_kind(text), text, page.number))
    return blocks


def _heading_block(match: re.Match, page: int | None) -> Block:
    title = match.group(2).strip()
    if title.endswith(":"):                      # "Response steps:" is a label, not a section
        return Block("paragraph", title, page)
    return Block("heading", title, page, level=len(match.group(1)))


def _block_kind(text: str) -> str:
    if text.startswith("|"):
        return "table"
    if LIST_ITEM.match(text):
        return "list"
    return "paragraph"


# --- Step 2: sections and heading paths ---------------------------------------------

def build_sections(blocks: list[Block]) -> list[Section]:
    """
    Group blocks under their headings and give every section its heading path.

    If the document uses more than one Markdown heading level (Word, HTML), those levels are
    trusted. Otherwise (PDF: every heading is '##') the level comes from the section number:
    "5." is level 1 and "5.1" is level 2. An unnumbered heading goes below the last numbered
    heading if that heading has no text of its own (an umbrella like "5. Playbooks"), and next
    to it otherwise. Sections without any text are dropped: their titles live on in the paths.
    """
    use_markdown_levels = len({block.level for block in blocks if block.kind == "heading"}) > 1
    sections = [Section(path=[], heading_page=None)]         # text before the first heading
    stack: list[tuple[int, str]] = []
    numbered_level: int | None = None                       # level of the last numbered heading
    numbered_has_text = False
    current_is_numbered = False

    for block in blocks:
        if block.kind != "heading":
            sections[-1].blocks.append(block)
            numbered_has_text = numbered_has_text or current_is_numbered
            continue

        number = None if use_markdown_levels else SECTION_NUMBER.match(block.text)
        if use_markdown_levels:
            level = block.level
        elif number:
            level = number.group(1).count(".") + 1
        elif numbered_level is None:
            level = 1                                         # title, slide or sheet name
        elif numbered_has_text:
            level = numbered_level                            # e.g. an appendix after section 8
        else:
            level = numbered_level + 1                        # e.g. "PB-01" under "5. Playbooks"

        current_is_numbered = number is not None
        if current_is_numbered:
            numbered_level, numbered_has_text = level, False

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, block.text))
        sections.append(Section(path=[title for _, title in stack], heading_page=block.page))

    return [section for section in sections if section.blocks]


# --- Step 3 and 4: packing into chunks ----------------------------------------------

def _chunk_section(source: str, section: Section) -> list[Chunk]:
    context = f"[{PATH_SEPARATOR.join([source, *section.path])}]"
    budget = config.CHUNK_SIZE_TOKENS - count_tokens(context)

    pieces: list[Block] = []
    for block in section.blocks:
        pieces.extend(split_block(block, budget) if count_tokens(block.text) > budget else [block])

    chunks = []
    for position, group in enumerate(_greedy_groups(pieces, lambda piece: count_tokens(piece.text), budget)):
        text = context + "\n" + "\n\n".join(piece.text for piece in group)
        pages = [piece.page for piece in group if piece.page is not None]
        if position == 0 and section.heading_page is not None:
            pages.append(section.heading_page)            # the first chunk also covers the heading's page
        chunks.append(Chunk(
            text=text,
            source=source,
            heading_path=PATH_SEPARATOR.join(section.path),
            page_start=min(pages, default=None),
            page_end=max(pages, default=None),
            chunk_index=0,                                 # set by chunk_document
            token_count=count_tokens(text),
        ))
    return chunks


def _greedy_groups(items: list[T], size_of: Callable[[T], int], budget: int) -> list[list[T]]:
    """Fill groups in order. An item that does not fit starts a new group; it is never cut."""
    groups: list[list[T]] = []
    used = 0
    for item in items:
        size = size_of(item) + 1               # +1 for the line break or space that joins it to the others
        if groups and used + size <= budget:
            groups[-1].append(item)
            used += size
        else:
            groups.append([item])
            used = size
    return groups


def split_block(block: Block, budget: int) -> list[Block]:
    """Split a block that does not fit into one chunk, without breaking rows, items or sentences."""
    lines = block.text.splitlines()
    if block.kind == "table":
        header_size = 2 if len(lines) > 1 and TABLE_SEPARATOR.match(lines[1]) else 1
        header, rows = lines[:header_size], lines[header_size:]
        if not rows:
            return [block]
        row_budget = budget - count_tokens("\n".join(header))
        return [Block("table", "\n".join(header + group), block.page)
                for group in _greedy_groups(rows, count_tokens, row_budget)]
    if block.kind == "list":
        return [Block("list", "\n".join(group), block.page)
                for group in _greedy_groups(_list_items(lines), count_tokens, budget)]
    return [Block("paragraph", text, block.page) for text in _split_paragraph(block.text, budget)]


def _list_items(lines: list[str]) -> list[str]:
    """One string per list item; lines that do not start a new item belong to the previous one."""
    items: list[str] = []
    for line in lines:
        if items and not LIST_ITEM.match(line):
            items[-1] += "\n" + line
        else:
            items.append(line)
    return items


def _split_paragraph(text: str, budget: int) -> list[str]:
    """Split at sentence ends. Each piece repeats the last sentence(s) of the previous one."""
    sentences: list[str] = []
    for sentence in SENTENCE_END.split(text):
        if count_tokens(sentence) > budget:                # a "sentence" without any full stop
            words = sentence.split()
            sentences += [" ".join(group) for group in _greedy_groups(words, count_tokens, budget)]
        elif sentence:
            sentences.append(sentence)

    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and count_tokens(" ".join([*current, sentence])) > budget:
            pieces.append(" ".join(current))
            overlap = _tail(current, config.CHUNK_OVERLAP_TOKENS)
            current = overlap if count_tokens(" ".join([*overlap, sentence])) <= budget else []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _tail(sentences: list[str], limit: int) -> list[str]:
    """The last sentences whose total size stays within the limit."""
    tail: list[str] = []
    for sentence in reversed(sentences):
        if count_tokens(" ".join([sentence, *tail])) > limit:
            break
        tail.insert(0, sentence)
    return tail