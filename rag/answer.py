"""
Answering: find the relevant chunks, show them to the LLM with clear rules, and return the
answer together with the sources it was based on.

Confidential chunks (config.LOCAL_ONLY_ACCESS_LEVELS) are only ever sent to a model that runs on
our own machines. If that model is down, there is no answer rather than a fallback to the cloud.

In a conversation (step 17) a follow-up question is first rewritten into a standalone one, which is
then used for the search and given to the LLM. When the reranker (step 16) finds nothing relevant,
the LLM is not asked at all.
"""
import time
from dataclasses import dataclass, replace

import config
from rag.citations import Citations, Unverified, parse_citations, verify_citations
from rag.llm import LLMResult, LLMUnavailable, generate, get_providers
from rag.retriever import Hit, search
from rag.rewriter import rewrite_question

SYSTEM_PROMPT = """You are CIA (Corporate Information Assistant), the internal assistant that answers \
employees' questions from the company's documents.

Rules:
1. Answer only from the numbered sources in the user message. Do not use outside knowledge, \
even when you are sure of it (for example laws or common practice).
2. If the sources do not answer the question, say so plainly in one sentence, cite nothing, and end \
your reply with [NOT_FOUND]. Do not guess, and do not carry over facts that belong to a similar but \
different case. If the sources answer only part of the question, answer that part and say which part \
is missing, without the marker.
3. After each statement, cite the sources that state it, like [2] or [1][3]. Cite only a source that \
actually contains the statement, and only numbers from the list.
4. If sources disagree, prefer the one with the later effective date and mention the difference.
5. Sources marked as scanned may contain reading errors: say so when your answer relies on one.
6. Answer in the language of the question. Be brief and clear; use a short list only when it helps."""

NO_SOURCES_TEXT = "No relevant documents were found for this question."


class ConfidentialModelUnavailable(LLMUnavailable):
    """The sources are confidential and no local model answered; the cloud is not an option."""


@dataclass
class Answer:
    question: str                  # as the user typed it
    text: str                      # exactly as the LLM wrote it, citation numbers included
    sources: list[Hit]             # what the LLM was shown, numbered from 1 in this order
    llm: LLMResult | None          # None when no source was found and the LLM was not asked
    retrieval_ms: float
    search_question: str = ""      # what was searched and given to the LLM: the question, or its rewrite
    rewrite: LLMResult | None = None   # the LLM call that rewrote a follow-up question (step 17)

    def __post_init__(self) -> None:
        self.search_question = self.search_question or self.question

    @property
    def rewritten(self) -> bool:
        """The question was searched in another wording, filled in from the conversation."""
        return self.search_question.strip() != self.question.strip()

    @property
    def citations(self) -> Citations:
        """Which sources the answer cites, and whether it says the answer was not found."""
        found = parse_citations(self.text, len(self.sources))
        return replace(found, not_found=True) if self.llm is None else found

    def cited_sources(self) -> list[tuple[int, Hit]]:
        """The sources the answer cites, with their numbers, in the order they are first cited."""
        return [(number, self.sources[number - 1]) for number in self.citations.numbers]

    def unused_sources(self) -> list[tuple[int, Hit]]:
        """Sources the LLM was shown but did not cite: useful to see what the search brought in."""
        cited = set(self.citations.numbers)
        return [(number, hit) for number, hit in enumerate(self.sources, start=1) if number not in cited]

    @property
    def confidential(self) -> bool:
        """The LLM was shown at least one chunk that must stay on our own machines."""
        return is_confidential(self.sources)

    @property
    def unverified(self) -> list[Unverified]:
        """Numbers and codes of the answer that are not in the sources it cites (step 19)."""
        if self.llm is None:
            return []
        return verify_citations(self.text, [hit.text for hit in self.sources], self.search_question)


def is_confidential(hits: list[Hit]) -> bool:
    return any(hit.access_level in config.LOCAL_ONLY_ACCESS_LEVELS for hit in hits)


def local_route() -> list[str]:
    """The answer providers that run on our own machines, in the order of the task's route."""
    providers = get_providers()
    return [name for name in config.LLM_TASKS["answer"]["providers"] if providers[name].is_local]


def source_header(number: int, hit: Hit) -> str:
    """One line per source with what the LLM needs to judge it: where it is from and how current it is."""
    details = [hit.pages] if hit.pages else []
    if hit.effective_date:
        details.append(f"effective {hit.effective_date}")
    if hit.status == "superseded":
        details.append(f"SUPERSEDED by {hit.replaced_by}")
    if hit.ocr:
        details.append("scanned document read with OCR, may contain reading errors")
    where = " > ".join(part for part in (hit.source, hit.heading_path) if part)
    return f"[{number}] {where}" + (f" ({', '.join(details)})" if details else "")


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    """The chat messages sent to the LLM: the rules, then the numbered sources and the question."""
    sources = "\n\n".join(f"{source_header(number, hit)}\n{hit.content}"
                          for number, hit in enumerate(hits, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
    ]


def answer_question(question: str, top_k: int = config.RETRIEVAL_TOP_K, include_superseded: bool = False,
                    access_levels: tuple[str, ...] = config.DEFAULT_ACCESS_LEVELS,
                    route: list[str] | None = None, retrieval_mode: str | None = None,
                    history: list[str] | tuple[str, ...] = (), rerank: bool | None = None) -> Answer:
    """
    Rewrite (in a conversation), retrieve, then generate. The LLM is not asked at all when nothing
    relevant was found.

    access_levels: what the person asking may see (step 12 uses a role table, step 14 real users).
    route: LLM providers to use instead of the default order (see rag.llm.generate). Ignored when
           the sources are confidential: then only local providers are used.
    retrieval_mode: hybrid, dense or sparse (default: config.RETRIEVAL_MODE), for comparisons.
    history: the earlier questions of the conversation, oldest first (step 17).
    rerank: use the reranker (default: config.RERANK), for comparisons.
    """
    search_question, rewrite = rewrite_question(question, history, access_levels, route)
    started = time.perf_counter()
    hits = search(search_question, top_k=top_k, include_superseded=include_superseded,
                  access_levels=access_levels, mode=retrieval_mode, rerank=rerank)
    retrieval_ms = round((time.perf_counter() - started) * 1000, 1)
    if not hits:
        return Answer(question, NO_SOURCES_TEXT, [], None, retrieval_ms, search_question, rewrite)

    confidential = is_confidential(hits)
    if confidential:
        route = local_route()
        if not route:
            raise ConfidentialModelUnavailable("The sources are confidential, but no local model is configured.")
    try:
        result = generate("answer", build_messages(search_question, hits), route=route)
    except LLMUnavailable as exc:
        if confidential:                           # fail closed: no answer rather than a cloud model
            raise ConfidentialModelUnavailable(f"The sources are confidential and the local model did not "
                                               f"answer ({exc}).") from exc
        raise
    return Answer(question, result.text, hits, result, retrieval_ms, search_question, rewrite)
