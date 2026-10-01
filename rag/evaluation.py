"""
Evaluation: measure CIA on a fixed question set, so that every change can be compared with the last run.

Every question gets five checks:
  1. retrieval  Was a file with the answer found, at which rank, and did the chunks shown to the LLM
                contain the key facts ("evidence")? Without them the LLM cannot answer. For an
                off-topic question nothing may be found at all, so that the LLM is not asked (step 16).
  2. phrases    Does the answer contain (or avoid) given phrases? Cheap and repeatable, but literal.
  3. judge      An LLM compares the answer with the answer key. Understands wording, but costs a call.
  4. security   Did anything the role may not see reach the LLM or the answer, and did confidential
                content stay on our own machines (answer and judge model)?
  5. citations  Does the answer cite existing sources, at least one of them with the answer, and does
                it mark an answer that is not in the documents as such? Numbers and codes that are not
                in the sources they cite are listed (step 19), but do not fail a question on their own:
                a calculated number (22 + 2 = 24) is in no source either.

When a question does not pass, the diagnosis says where to look: reading (the fact never made it into
the index), retrieval (it is indexed but was not found), generation (it was shown to the LLM, the
answer is still wrong), made up an answer, not stopped (an off-topic question reached the LLM), or LEAK.

Only the judge needs an LLM. The other checks can be repeated on a saved run (rescore_result),
so a change to a check does not cost a new run. The running, waiting and printing live in
scripts/evaluate.py; this module only decides.
"""
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import config
from rag.answer import Answer
from rag.citations import parse_citations
from rag.llm import generate, get_providers
from rag.store import get_qdrant

REQUIRED_FIELDS = {"id", "type", "role", "question", "expected"}
PHRASE_FIELDS = ("sources", "must_include", "any_of", "must_not", "forbidden", "history")
QUESTION_FIELDS = REQUIRED_FIELDS | set(PHRASE_FIELDS) | {"evidence", "answerable", "known_gap", "off_topic"}
VERDICTS = ("correct", "partial", "wrong")
STATUS_BY_VERDICT = {"correct": "pass", "partial": "partial", "wrong": "fail"}

JUDGE_PROMPT = """You grade the answers of a company document assistant against an answer key.

You receive a QUESTION, the REFERENCE answer (the answer key, correct by definition) and the ANSWER to grade. \
Judge only whether the ANSWER agrees with the REFERENCE. Do not use your own knowledge.

- The key facts are the parts of the reference outside parentheses. Parts in parentheses are optional \
context: the answer does not need them, but it must not contradict them.
- correct: the answer contains every key fact (numbers, names, yes or no) and nothing that contradicts \
the reference. Wording, language, extra details that do not contradict the reference and source markers \
like [1] do not matter.
- partial: nothing contradicts the reference, but a key fact is missing (for example one part of a two-part question).
- wrong: a key fact is wrong or contradicts the reference, the question is not answered, or the answer says \
the information is missing although the reference contains it.
- When the reference says the information is not in the documents, the answer is correct only if it says so \
without guessing. Giving a value, or carrying over facts from a similar case, is wrong.

Reply with JSON only, no other text:
{"verdict": "correct" or "partial" or "wrong", "reason": "one short sentence in English"}"""


class QuestionSetError(ValueError):
    """The question file is missing, malformed or does not match the indexed documents."""


@dataclass(frozen=True)
class Question:
    id: int
    type: str
    role: str
    question: str
    expected: str
    answerable: bool = True
    sources: tuple[str, ...] = ()
    evidence: tuple[tuple[str, ...], ...] = ()     # each fact: one or more phrases, any of which proves it
    must_include: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()
    must_not: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    history: tuple[str, ...] = ()
    known_gap: str | None = None
    off_topic: bool = False                         # nothing in the documents is about it: the LLM must not be asked


@dataclass
class Result:
    """Everything measured for one question; saved as JSON so that runs can be compared later."""
    id: int
    type: str
    role: str
    question: str
    answerable: bool
    known_gap: str | None
    off_topic: bool = False
    answer: str = ""
    searched_as: str = ""                      # the rewritten follow-up question; empty when not rewritten
    answered_by: str = ""                      # empty when the LLM was not asked
    sources: list[dict] = field(default_factory=list)
    source_rank: int | None = None             # rank of the first retrieved file that has the answer
    evidence_total: int = 0
    missing_evidence: list[str] = field(default_factory=list)       # not among the chunks shown to the LLM
    evidence_not_indexed: list[str] = field(default_factory=list)   # not in any chunk of the collection
    phrase_problems: list[str] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)              # valid source numbers cited
    invalid_citations: list[int] = field(default_factory=list)      # numbers that are not in the source list
    not_found: bool = False                    # the answer says the documents do not contain it
    citation_problems: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)            # numbers and codes not in the cited sources
    verdict: str | None = None                 # correct, partial or wrong; None when not judged
    judge_reason: str = ""
    judged_by: str = ""                        # the judge model, empty when not judged
    status: str = ""                           # pass, partial or fail
    diagnosis: str = ""                        # why it did not pass: reading, retrieval, generation, ...
    disagreement: bool = False                 # the phrase checks and the judge disagree
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


# --- Matching phrases ---------------------------------------------------------------

def normalize(text: str) -> str:
    """Turkish-aware lowercase without Markdown bold, curly quotes or extra spaces."""
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = text.replace("*", "").replace("`", "").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip()


def contains(text: str, phrase: str) -> bool:
    """
    True when the phrase appears in the text at the start of a word (both are normalized first).

    Turkish adds suffixes on the right, so the right edge is left open: "22 gün" matches "22 gündür".
    The left edge must be a word start: "11" does not match inside "M-1011" or "01.11.2024".
    """
    text, phrase = normalize(text), normalize(phrase)
    start = text.find(phrase)
    while start != -1:
        if _starts_word(text, start):
            return True
        start = text.find(phrase, start + 1)
    return False


def _starts_word(text: str, index: int) -> bool:
    if index == 0 or not text[index].isalnum():
        return True
    before = text[index - 1]
    if before.isalnum():
        return False
    inside_number = before in ".," and index >= 2 and text[index - 2].isdigit() and text[index].isdigit()
    return not inside_number


def evidence_label(alternatives: tuple[str, ...]) -> str:
    return " / ".join(alternatives)


# --- Loading the question set -------------------------------------------------------

def load_questions(path: Path = config.EVAL_QUESTIONS) -> list[Question]:
    """Read and validate the question file. A mistake here would make every measurement wrong."""
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except FileNotFoundError as exc:
        raise QuestionSetError(f"Question file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise QuestionSetError(f"{path.name} is not valid TOML: {exc}") from exc

    unknown = set(data) - {"question", "not_found_phrases"}
    if unknown:
        raise QuestionSetError(f"{path.name}: unknown top-level key(s) {sorted(unknown)}.")
    not_found = _phrases(data.get("not_found_phrases", []), "not_found_phrases")
    items = data.get("question", [])
    if not isinstance(items, list) or not items:
        raise QuestionSetError(f"{path.name}: write every question under [[question]] (two brackets).")

    questions: list[Question] = []
    for number, item in enumerate(items, start=1):
        question = _parse_question(item, f"{path.name}, question #{number} (id {item.get('id')})", not_found)
        if any(earlier.id == question.id for earlier in questions):
            raise QuestionSetError(f"{path.name}: id {question.id} is used twice.")
        questions.append(question)
    return questions


def _parse_question(item: dict, where: str, not_found: tuple[str, ...]) -> Question:
    unknown = set(item) - QUESTION_FIELDS
    if unknown:                                   # e.g. "must_includ": a typo must not be ignored
        raise QuestionSetError(f"{where}: unknown field(s) {sorted(unknown)}.")
    missing = REQUIRED_FIELDS - set(item)
    if missing:
        raise QuestionSetError(f"{where}: missing field(s) {sorted(missing)}.")
    if type(item["id"]) is not int:
        raise QuestionSetError(f"{where}: id must be a number.")
    for key in ("type", "role", "question", "expected"):
        if not isinstance(item[key], str) or not item[key].strip():
            raise QuestionSetError(f"{where}: {key} must be a non-empty text.")
    if item["role"] not in config.ROLES:
        raise QuestionSetError(f"{where}: role must be one of {list(config.ROLES)}.")
    answerable = item.get("answerable", True)
    if not isinstance(answerable, bool):
        raise QuestionSetError(f"{where}: answerable must be true or false (without quotes).")
    known_gap = item.get("known_gap")
    if known_gap is not None and not isinstance(known_gap, str):
        raise QuestionSetError(f"{where}: known_gap must be a text.")
    off_topic = item.get("off_topic", False)
    if not isinstance(off_topic, bool):
        raise QuestionSetError(f"{where}: off_topic must be true or false (without quotes).")
    if off_topic and answerable:
        raise QuestionSetError(f"{where}: an off-topic question must also have answerable = false.")

    lists = {key: _phrases(item.get(key, []), f"{where}: {key}") for key in PHRASE_FIELDS}
    evidence = _evidence(item.get("evidence", []), f"{where}: evidence")
    if answerable and not lists["sources"]:
        raise QuestionSetError(f"{where}: an answerable question needs sources.")
    if not answerable and (lists["sources"] or evidence):
        raise QuestionSetError(f"{where}: an unanswerable question cannot have sources or evidence.")
    if not answerable and not lists["any_of"]:
        lists["any_of"] = not_found               # the answer must say that the information is missing

    return Question(id=item["id"], type=item["type"], role=item["role"], question=item["question"],
                    expected=item["expected"], answerable=answerable, known_gap=known_gap,
                    off_topic=off_topic, evidence=evidence, **lists)


def _phrases(value, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(text, str) and text.strip() for text in value):
        raise QuestionSetError(f"{where}: must be a list of non-empty texts, like [\"22 gün\"].")
    return tuple(value)


def _evidence(value, where: str) -> tuple[tuple[str, ...], ...]:
    """Each entry is a phrase, or a list of phrases of which any one proves the same fact."""
    if not isinstance(value, list) or any(entry == [] for entry in value):
        raise QuestionSetError(f"{where}: must be a list, like [\"22 gün\"] or [[\"one wording\", \"another\"]].")
    return tuple(_phrases(entry if isinstance(entry, list) else [entry], where) for entry in value)


def check_against_index(questions: list[Question], client=None,
                        name: str = config.QDRANT_COLLECTION) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """
    Test the question set itself before spending any LLM call: every expected file must be indexed
    and every evidence phrase must exist in one of its chunks. Otherwise the question has a typo,
    or reading lost the fact, and the measurement would blame the wrong thing.

    Returns (problems, not_indexed), both keyed by question id. not_indexed lists the evidence that
    is in no chunk at all: for a known gap that is a finding (diagnosis "reading"), not a typo.
    """
    client = client or get_qdrant()
    chunks: dict[str, list[str]] = {}
    offset = None
    while True:
        points, offset = client.scroll(name, limit=1000, offset=offset,
                                       with_payload=["source", "text"], with_vectors=False)
        for point in points:
            chunks.setdefault(point.payload["source"], []).append(point.payload["text"])
        if offset is None:
            break
    if not chunks:
        raise QuestionSetError(f"Collection '{name}' is empty. Run: python -m scripts.index_documents")

    problems: dict[int, list[str]] = {}
    not_indexed: dict[int, list[str]] = {}
    for question in questions:
        for source in question.sources:
            if source not in chunks:
                problems.setdefault(question.id, []).append(f"'{source}' is not in the collection")
        texts = [text for source in question.sources for text in chunks.get(source, [])]
        for alternatives in question.evidence:
            if not any(contains(text, phrase) for text in texts for phrase in alternatives):
                not_indexed.setdefault(question.id, []).append(evidence_label(alternatives))
                problems.setdefault(question.id, []).append(
                    f"evidence {evidence_label(alternatives)!r} is in none of the chunks of {', '.join(question.sources)}")
    return problems, not_indexed


# --- Checks 1, 2, 4 and 5: without an LLM -------------------------------------------

def check_retrieval(question: Question, hits: list, retrieval_ms: float,
                    not_indexed: tuple[str, ...] | list[str] = ()) -> Result:
    """Checks 1 and 4 on what the search found: enough for a retrieval-only run (no LLM)."""
    result = Result(
        id=question.id, type=question.type, role=question.role, question=question.question,
        answerable=question.answerable, known_gap=question.known_gap, off_topic=question.off_topic,
        sources=[{"source": hit.source, "heading_path": hit.heading_path, "pages": hit.pages,
                  "score": round(hit.score, 3), "access_level": hit.access_level} for hit in hits],
        retrieval_ms=retrieval_ms, evidence_not_indexed=list(not_indexed),
    )

    # 1. Retrieval
    ranks = [rank for rank, hit in enumerate(hits, start=1) if hit.source in question.sources]
    result.source_rank = ranks[0] if ranks else None
    result.evidence_total = len(question.evidence)
    result.missing_evidence = [evidence_label(alternatives) for alternatives in question.evidence
                               if not any(contains(hit.text, phrase) for hit in hits for phrase in alternatives)]

    # 4. Security: the labels of what was retrieved, and forbidden phrases in what the LLM would see
    allowed = config.ROLES[question.role]["access_levels"]
    leaks = [f"{hit.source} ({hit.access_level}) was shown to the LLM"
             for hit in hits if hit.access_level not in allowed]
    for phrase in question.forbidden:
        shown_in = sorted({hit.source for hit in hits if contains(hit.text, phrase)})
        if shown_in:
            leaks.append(f"{phrase!r} was shown to the LLM ({', '.join(shown_in)})")
    result.leaks = sorted(set(leaks))
    return result


def check_answer(question: Question, answer: Answer, not_indexed: tuple[str, ...] | list[str] = ()) -> Result:
    """Run the checks that need no LLM on one answer. not_indexed comes from check_against_index."""
    result = check_retrieval(question, answer.sources, answer.retrieval_ms, not_indexed)
    result.answer = answer.text
    result.searched_as = answer.search_question if answer.rewritten else ""
    result.unverified = [describe_unverified(item) for item in answer.unverified]
    if answer.llm:
        result.answered_by = f"{answer.llm.provider} ({answer.llm.model})"
        result.llm_ms = answer.llm.elapsed_ms
        result.input_tokens, result.output_tokens = answer.llm.input_tokens or 0, answer.llm.output_tokens or 0

    # 2. Phrases in the answer, and 5. citations
    result.phrase_problems = check_phrases(question, answer.text)
    check_citations(question, result)

    # 4. Security, the answer itself
    leaked = [f"the answer contains {phrase!r}" for phrase in question.forbidden if contains(answer.text, phrase)]
    result.leaks = sorted(set(result.leaks) | set(leaked))
    return result


def describe_unverified(item) -> str:
    where = f"in source {', '.join(map(str, item.found_in))}" if item.found_in else "in no source"
    return f"{item.fact} (cites {', '.join(map(str, item.cited))}; {where})"


def not_stopped(result: Result) -> bool:
    """An off-topic question for which the search found chunks, so the LLM was (or would be) asked."""
    return result.off_topic and bool(result.sources)


def finalize_retrieval(result: Result) -> None:
    """Status of a retrieval-only run: was everything needed found, and nothing forbidden?"""
    found = result.source_rank is not None and not result.missing_evidence
    result.status = "fail" if result.leaks or not_stopped(result) else (
        "pass" if found or not result.answerable else "fail")
    if result.leaks:
        result.diagnosis = "LEAK"
    elif not_stopped(result):
        result.diagnosis = "not stopped"
    elif result.status == "pass":
        result.diagnosis = ""
    else:
        result.diagnosis = "reading" if result.evidence_not_indexed else "retrieval"


def check_phrases(question: Question, text: str) -> list[str]:
    problems = [f"missing {phrase!r}" for phrase in question.must_include if not contains(text, phrase)]
    if question.any_of and not any(contains(text, phrase) for phrase in question.any_of):
        problems.append("does not say that the information is missing" if not question.answerable
                        else f"none of {list(question.any_of)}")
    problems += [f"contains {phrase!r}" for phrase in question.must_not if contains(text, phrase)]
    return problems


def check_citations(question: Question, result: Result) -> None:
    """Read the citations of result.answer (its sources must be set) and record what is wrong with them."""
    found = parse_citations(result.answer, len(result.sources))
    result.citations, result.invalid_citations = list(found.numbers), list(found.invalid)
    result.not_found = found.not_found or not result.answered_by      # the LLM was not even asked
    cited_files = {result.sources[number - 1]["source"] for number in found.numbers}

    problems = []
    if found.invalid:
        problems.append(f"cites {list(found.invalid)}, which are not in the source list")
    if not question.answerable:
        if not result.not_found:
            problems.append("not marked as not found ([NOT_FOUND])")
    elif result.not_found:
        problems.append("says the answer is not in the documents")
    elif not found.numbers:
        problems.append("no citation")
    elif not cited_files & set(question.sources):
        problems.append(f"cites none of the expected files (cites {', '.join(sorted(cited_files))})")
    result.citation_problems = problems


def rescore_result(question: Question, saved: dict) -> Result:
    """
    Repeat the checks that need no LLM on a saved result, with the current rules.

    The answer, the judge's verdict and the retrieval and security results are kept as saved:
    the chunk texts they need are not in the saved file.
    """
    known = {name for name in Result.__dataclass_fields__}
    result = Result(**{key: value for key, value in saved.items() if key in known})
    result.type, result.role, result.known_gap = question.type, question.role, question.known_gap
    result.answerable, result.off_topic = question.answerable, question.off_topic
    result.phrase_problems = check_phrases(question, result.answer)
    check_citations(question, result)
    finalize(result)
    return result


# --- Check 3: the LLM judge ---------------------------------------------------------

def judge_messages(question: Question, answer_text: str) -> list[dict]:
    earlier = "".join(f"- {text}\n" for text in question.history)
    context = f"EARLIER QUESTIONS OF THE CONVERSATION:\n{earlier}\n" if earlier else ""
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"{context}QUESTION:\n{question.question}\n\nREFERENCE:\n{question.expected}"
                                    f"\n\nANSWER:\n{answer_text}"},
    ]


def parse_verdict(reply: str) -> tuple[str, str]:
    """The verdict and the reason from the judge's reply. ValueError when the reply is not usable."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)        # also accepts JSON wrapped in ```json fences
    if not match:
        raise ValueError(f"no JSON object in the judge's reply: {reply[:80]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"the judge's reply is not valid JSON: {exc}") from exc
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    return verdict, str(data.get("reason", "")).strip()


def judge(question: Question, answer_text: str) -> tuple[str, str, str]:
    """Ask the judge model (task "judge": one fixed model, temperature 0, no fallback).
    Returns the verdict, the reason and which model judged."""
    reply = generate("judge", judge_messages(question, answer_text))
    verdict, reason = parse_verdict(reply.text)
    return verdict, reason, f"{reply.provider} ({reply.model})"


def check_confidential(result: Result) -> None:
    """
    Content labelled local-only must not reach a model outside the company: neither the answer model,
    which sees the chunks, nor the judge, which sees the answer written from them.
    """
    files = sorted({source["source"] for source in result.sources
                    if source["access_level"] in config.LOCAL_ONLY_ACCESS_LEVELS})
    if not files:
        return
    for what, model in (("answer", result.answered_by), ("judge", result.judged_by)):
        provider = model.split(" ")[0]
        if provider and not _is_local(provider):
            result.leaks = sorted(set(result.leaks) | {
                f"confidential content ({', '.join(files)}) went to the external {what} model '{provider}'"})


def _is_local(provider: str) -> bool:
    known = get_providers()
    return provider in known and known[provider].is_local     # an unknown provider counts as external


# --- Verdict for the question and the run -------------------------------------------

def finalize(result: Result) -> None:
    """Combine the checks into one status, and say where the problem is when it did not pass."""
    check_confidential(result)                 # here, because only now is the judge known as well
    phrases_ok = not result.phrase_problems
    if result.leaks or not_stopped(result):
        result.status = "fail"
    elif result.verdict:
        result.status = STATUS_BY_VERDICT[result.verdict]
    else:
        result.status = "pass" if phrases_ok else "fail"
    result.disagreement = result.verdict is not None and phrases_ok != (result.verdict == "correct")

    if result.leaks:
        result.diagnosis = "LEAK"
    elif not_stopped(result):
        result.diagnosis = "not stopped"                  # an off-topic question reached the LLM
    elif result.status == "pass":
        result.diagnosis = ""
    elif not result.answerable:
        result.diagnosis = "made up an answer"            # T20: should have said "not in the documents"
    elif result.evidence_not_indexed:
        result.diagnosis = "reading"                      # the fact never made it into the index
    elif result.source_rank is None or result.missing_evidence:
        result.diagnosis = "retrieval"                    # indexed, but not shown to the LLM
    else:
        result.diagnosis = "generation"                   # the facts were there, the answer is still wrong


def summarize(results: list[Result]) -> dict:
    """The numbers of a run. Flat, so that two runs can be compared key by key."""
    def count(items, condition) -> int:
        return sum(1 for item in items if condition(item))

    answerable = [result for result in results if result.answerable]
    with_evidence = [result for result in answerable if result.evidence_total]
    unanswerable = [result for result in results if not result.answerable]
    off_topic = [result for result in results if result.off_topic]
    with_llm = [result for result in results if result.answered_by]      # the LLM gave an answer
    return {
        "questions": len(results),
        "pass": count(results, lambda r: r.status == "pass"),
        "partial": count(results, lambda r: r.status == "partial"),
        "fail": count(results, lambda r: r.status == "fail"),
        "known_gaps_not_passed": count(results, lambda r: r.known_gap and r.status != "pass"),
        "answerable": len(answerable),
        "top_k": config.RETRIEVAL_TOP_K,
        "hit@1": count(answerable, lambda r: r.source_rank == 1),
        "hit@3": count(answerable, lambda r: r.source_rank is not None and r.source_rank <= 3),
        "hit@k": count(answerable, lambda r: r.source_rank is not None),
        "mrr": round(mean([1 / r.source_rank if r.source_rank else 0 for r in answerable]), 3) if answerable else 0,
        "with_evidence": len(with_evidence),
        "evidence_found": count(with_evidence, lambda r: not r.missing_evidence),
        "phrases_ok": count(results, lambda r: not r.phrase_problems),
        "verdict_correct": count(results, lambda r: r.verdict == "correct"),
        "verdict_partial": count(results, lambda r: r.verdict == "partial"),
        "verdict_wrong": count(results, lambda r: r.verdict == "wrong"),
        "not_judged": count(results, lambda r: r.verdict is None),
        "disagreements": count(results, lambda r: r.disagreement),
        "citations_ok": count(results, lambda r: not r.citation_problems),
        "invalid_citations": count(results, lambda r: r.invalid_citations),
        "avg_cited": round(mean([len(r.citations) for r in with_llm]), 2) if with_llm else 0,
        "unanswerable": len(unanswerable),
        "unanswerable_ok": count(unanswerable, lambda r: r.status == "pass"),
        "off_topic": len(off_topic),
        "off_topic_stopped": count(off_topic, lambda r: not r.sources),
        "no_sources": count(results, lambda r: not r.sources),          # nothing relevant: the LLM is not asked
        "unverified_answers": count(results, lambda r: r.unverified),
        "leaks": count(results, lambda r: r.leaks),
        "avg_retrieval_ms": round(mean([r.retrieval_ms for r in results]), 1) if results else 0,
        "avg_llm_s": round(mean([r.llm_ms for r in with_llm]) / 1000, 2) if with_llm else 0,
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
    }
