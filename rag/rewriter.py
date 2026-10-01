"""
Query rewriting (step 17): turn a follow-up question into one that can be searched on its own.

"Peki yöneticiler için?" means nothing to the search: it has no topic. With the earlier questions of
the conversation an LLM can rewrite it as "Yöneticilerin yıllık izni kaç gün?", which the search,
the reranker and the answering LLM all understand.

Only the earlier *questions* are sent, never the answers: answers are written from documents and
may be confidential, and the topic of a follow-up is almost always in the questions.

The rewrite runs before the search, so nobody knows yet whether the conversation is about
confidential documents. For a user who may read confidential documents it therefore runs on a
local model only. If no model answers, the question is searched as it was typed: a worse search,
but never a reason to fail or to send anything elsewhere.
"""
import logging
import re

import config
from rag.llm import LLMError, LLMResult, generate, get_providers

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You rewrite the last question of a conversation so that it can be understood on its own. \
The result is used to search company documents.

Rules:
1. If the last question already makes sense on its own, return it exactly as it is, even when its topic \
differs from the earlier questions.
2. Otherwise add what it refers to from the earlier questions (the topic, a person, a code, a number), \
and nothing else.
3. Do not answer the question. Do not add facts, guesses or explanations.
4. Keep the language and the wording of the last question as far as possible.

Reply with the rewritten question only, on a single line."""

MAX_GROWTH = 200          # characters a rewrite may add; a much longer reply is an answer, not a question
PREFIX = re.compile(r"^(rewritten question|question|soru)\s*:\s*", re.IGNORECASE)


def rewrite_messages(question: str, history: list[str]) -> list[dict]:
    earlier = "\n".join(f"{number}. {text}" for number, text in enumerate(history, start=1))
    return [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"Earlier questions, oldest first:\n{earlier}\n\nLast question: {question}"},
    ]


def rewrite_route(access_levels: tuple[str, ...], route: list[str] | None = None) -> list[str] | None:
    """Local models only for users who may read confidential documents; otherwise the given route."""
    if not set(access_levels) & set(config.LOCAL_ONLY_ACCESS_LEVELS):
        return route
    providers = get_providers()
    return [name for name in config.LLM_TASKS["rewrite"]["providers"] if providers[name].is_local]


def rewrite_question(question: str, history: list[str] | tuple[str, ...], access_levels: tuple[str, ...],
                     route: list[str] | None = None) -> tuple[str, LLMResult | None]:
    """
    The question to search with, and the LLM call that produced it (None when no rewrite was needed
    or possible). history: the earlier questions of the conversation, oldest first.
    """
    history = [text.strip() for text in history if text.strip()][-config.HISTORY_QUESTIONS:]
    if not history:
        return question, None
    names = rewrite_route(access_levels, route)
    if names == []:
        logger.warning("No local model is configured for rewriting; searching the question as typed.")
        return question, None
    try:
        result = generate("rewrite", rewrite_messages(question, history), route=names)
    except LLMError as exc:
        logger.warning("Rewriting failed, searching the question as typed: %s", exc)
        return question, None
    rewritten = clean_rewrite(result.text)
    if not rewritten or len(rewritten) > len(question) + MAX_GROWTH:
        logger.warning("Unusable rewrite %r; searching the question as typed.", result.text[:120])
        return question, result
    return rewritten, result


def clean_rewrite(text: str) -> str:
    """The first non-empty line, without a label like "Question:" and without quotes around it."""
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    line = PREFIX.sub("", line).strip()
    if len(line) >= 2 and line[0] == line[-1] and line[0] in "\"'«“”":
        line = line[1:-1].strip()
    return line.strip("“”").strip()
