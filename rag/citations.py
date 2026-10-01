"""
Citations: which numbered sources an answer refers to, whether it says the answer was not found,
and whether the numbers and codes it states can be found in the sources it cites (step 19).

The LLM sees the sources numbered [1] to [n] and cites them in its answer, like [2] or [1][3].
The answer text stays exactly as the LLM wrote it; this module only reads it, so the same
function works for a live answer, for the web interface and for a saved evaluation result.
"""
import re
import textwrap
from dataclasses import dataclass

from rag.sparse import ASCII_FOLD, CODE

NOT_FOUND_MARKER = "[NOT_FOUND]"

# [2]  [1][3]  [1, 3]  [1-3]  (a range may also use an en dash)
CITATION = re.compile(r"\[\s*(\d+(?:\s*[,\-–]\s*\d+)*)\s*\]")
NOT_FOUND = re.compile(r"\s*\[\s*NOT[_ ]FOUND\s*\]", re.IGNORECASE)
MAX_RANGE = 20          # "[1-3]" means 1, 2, 3; a "range" like [1-2000] is not a citation


@dataclass(frozen=True)
class Citations:
    numbers: tuple[int, ...]        # valid source numbers, in the order they are first cited
    invalid: tuple[int, ...]        # numbers in the text that are not in the source list
    not_found: bool                 # the answer says the sources do not contain the answer
    clean_text: str                 # the answer without the NOT_FOUND marker, for display


def parse_citations(text: str, source_count: int) -> Citations:
    """Read the citations of an answer that was given with source_count numbered sources."""
    numbers: list[int] = []
    invalid: list[int] = []
    for match in CITATION.finditer(text):
        for number in _expand(match.group(1)):
            target = numbers if 1 <= number <= source_count else invalid
            if number not in target:
                target.append(number)
    not_found = NOT_FOUND.search(text) is not None
    clean_text = NOT_FOUND.sub("", text).strip()
    return Citations(tuple(numbers), tuple(invalid), not_found, clean_text)


def _expand(group: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"\s*,\s*", group):
        bounds = [int(value) for value in re.split(r"\s*[-–]\s*", part)]
        if len(bounds) == 2 and 0 < bounds[1] - bounds[0] <= MAX_RANGE:
            numbers.extend(range(bounds[0], bounds[1] + 1))
        else:
            numbers.extend(bounds)
    return numbers


def best_excerpt(content: str, answer: str, width: int) -> str:
    """
    The line or sentence of a source that shares the most words with the answer, shortened to width.

    A chunk can be long (a whole table), and its first line is often not the part the answer used.
    This is only a pointer for the reader; verify_citations checks the numbers and codes.
    """
    answer_words = _words(answer)
    pieces = []
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("|"):                       # a table row stays whole: it is one record
            pieces.append(line.strip(" |"))
        else:
            pieces.extend(re.split(r"(?<=[.!?])\s+", line))
    pieces = [piece for piece in pieces if _words(piece)]
    if not pieces:
        return ""
    best = max(pieces, key=lambda piece: len(answer_words & _words(piece)))   # ties: the earliest piece
    return textwrap.shorten(best, width=width, placeholder=" …")


def _words(text: str) -> set[str]:
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return {word for word in re.findall(r"\w+", text) if len(word) > 2 or word.isdigit()}


# --- Verifying citations (step 19) --------------------------------------------------
# A wrong number is the most harmful mistake an answer can make (22 or 20 days of leave, 3.250 or
# 2.750 TL), and the easiest to check without another model: every number and code in a statement
# must appear in the sources that the statement cites. A number the answer calculated (22 + 2 = 24)
# is not in any source either: it is reported too, because a person should check it.

NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
STATEMENT_END = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Unverified:
    fact: str                       # the number or code as the answer wrote it
    statement: str                  # the statement it is in, without citation markers
    cited: tuple[int, ...]          # the sources it was checked against
    found_in: tuple[int, ...]       # other sources shown to the LLM that do contain it (a wrong citation)


def verify_citations(answer: str, sources: list[str], question: str = "") -> list[Unverified]:
    """
    The numbers and codes of the answer that are not in the sources they cite. Each statement is
    checked against its own citations; a statement without any against all sources the answer cites.
    Numbers that are already in the question are not checked. sources: texts of the numbered sources.
    """
    cited_anywhere = parse_citations(answer, len(sources)).numbers
    if not cited_anywhere:
        return []                   # no citation at all: reported as such, nothing to verify against
    source_facts = [_facts(text, expand=True) for text in sources]
    known = _facts(question, expand=True)
    problems: list[Unverified] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    for statement in STATEMENT_END.split(LIST_MARKER.sub("", NOT_FOUND.sub("", answer))):
        cited = parse_citations(statement, len(sources)).numbers or cited_anywhere
        text = CITATION.sub("", statement).strip()
        allowed = set().union(*(source_facts[number - 1] for number in cited))
        for fact, written in _facts_written(text):
            if fact in allowed or fact in known or (fact, cited) in seen:
                continue
            seen.add((fact, cited))
            found_in = tuple(number for number, facts in enumerate(source_facts, start=1)
                             if fact in facts and number not in cited)
            problems.append(Unverified(written, textwrap.shorten(text, 160, placeholder=" …"), cited, found_in))
    return problems


def _fold(text: str) -> str:
    return text.replace("İ", "i").lower().translate(ASCII_FOLD)


def _number_key(number: str) -> str:
    """180.000 -> 180000, 08:00 -> 800, 1,8 -> 18: the digits without separators and leading zeros."""
    digits = re.sub(r"\D", "", number)
    return digits.lstrip("0") or "0"


def _is_code(text: str) -> bool:
    """VPN-ERR-301 or M-1014 is a code; 08:00-20:00 or 180.000-250.000 are numbers."""
    return any(char.isdigit() for char in text) and any(char.isalpha() for char in text)


def _facts_written(text: str) -> list[tuple[str, str]]:
    """(key, as written) for every code and every number outside codes."""
    folded = _fold(text)
    facts = []
    for match in CODE.finditer(folded):
        if _is_code(match.group()):
            facts.append(("code:" + re.sub(r"[-_/]", "", match.group()), text[match.start():match.end()]))
    rest = CODE.sub(lambda match: " " if _is_code(match.group()) else match.group(), folded)
    facts += [(_number_key(match.group()), match.group()) for match in NUMBER.finditer(rest)]
    return facts


def _facts(text: str, expand: bool = False) -> set[str]:
    """
    The keys of the numbers and codes in a text. expand (for sources and the question) also adds the
    parts: the numbers inside a code (VPN-ERR-301 -> 301) and the parts of a number with separators
    (31.08.2028 -> 31, 8, 2028; 85.000 -> 85), so that "31 Ağustos 2028" or "85 bin TL" are found too.
    """
    facts = {key for key, _ in _facts_written(text)}
    if expand:
        for match in NUMBER.finditer(_fold(text)):
            facts.add(_number_key(match.group()))
            facts.update(_number_key(part) for part in re.split(r"[.,:/]", match.group()))
    return facts
