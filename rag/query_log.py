"""
Question log (step 19): one JSON line per question asked in the web interface.

What it is for: which questions find nothing (documents that are missing), which answers have
citation problems, which errors happen and how long answers take. Read it with scripts/show_log.py.

What it holds: who asked, the question, how it was searched, which sources were found and cited,
and what went wrong. Not the answer text and not the document text, so that the log is not a second
copy of the documents. The questions and names are still personal data: the log stays in data/
(outside git), one file per month, and old files should be deleted when no longer needed.

Logging must never break answering: a problem writing the log is only reported as a warning.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import config
from rag.answer import Answer

logger = logging.getLogger(__name__)


def log_path(month: str | None = None) -> Path:
    """The log file of a month, written as 2026-10 (default: this month)."""
    month = month or f"{datetime.now():%Y-%m}"
    return config.QUERY_LOG_DIR / f"queries_{month}.jsonl"


def log_question(username: str, role: str, question: str, answer: Answer | None = None,
                 error: Exception | None = None) -> None:
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "user": username,
        "role": role,
        "question": question,
    }
    if answer is not None:
        citations = answer.citations
        record.update({
            "searched_as": answer.search_question if answer.rewritten else None,
            "sources": [{"source": hit.source, "heading": hit.heading_path, "score": round(hit.score, 3),
                         "access_level": hit.access_level} for hit in answer.sources],
            "cited": list(citations.numbers),
            "invalid_citations": list(citations.invalid),
            "not_found": citations.not_found,
            "llm_asked": answer.llm is not None,
            "model": f"{answer.llm.provider} ({answer.llm.model})" if answer.llm else None,
            "fallback": answer.llm.fallback_reason if answer.llm and answer.llm.used_fallback else None,
            "confidential": answer.confidential,
            "unverified": [item.fact for item in answer.unverified],
            "retrieval_ms": answer.retrieval_ms,
            "llm_ms": answer.llm.elapsed_ms if answer.llm else None,
            "rewrite_ms": answer.rewrite.elapsed_ms if answer.rewrite else None,
        })
    if error is not None:
        record["error"] = type(error).__name__
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("Could not write the question log: %s", exc)


def read_log(month: str | None = None) -> list[dict]:
    """All records of a month; lines that cannot be read are skipped with a warning."""
    path = log_path(month)
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("%s line %d is not valid JSON; skipped.", path.name, number)
    return records
