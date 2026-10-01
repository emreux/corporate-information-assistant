"""
Step 12: run the evaluation question set, print a report and save the results as JSON.

A full run makes about two LLM calls per question (answer + judge) and takes a few minutes,
because the calls are spaced out to stay under the free tier's limits.

Usage (from the project root):
    python -m scripts.evaluate --label baseline      # all questions
    python -m scripts.evaluate --only 1,5,13         # selected questions
    python -m scripts.evaluate --no-judge            # phrase checks only: half the LLM calls
    python -m scripts.evaluate --provider qwen       # answers from the local model
    python -m scripts.evaluate --rescore evaluation/results/FILE.json
                                                     # repeat the checks without an LLM on a saved run
    python -m scripts.evaluate --retrieval-only      # only the search: no LLM call, a few seconds
    python -m scripts.evaluate --mode dense          # hybrid (default), dense or sparse search
    python -m scripts.evaluate --no-rerank           # without the reranker (step 16), for comparisons
Follow-up questions (with history) are rewritten by an LLM first (step 17), so a retrieval-only run
skips them. Compare two runs:
    python -m scripts.compare_eval --last 2
"""
import argparse
import json
import logging
import re
import sys
import textwrap
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import config
from rag.answer import answer_question
from rag.evaluation import (Question, QuestionSetError, Result, check_against_index, check_answer,
                            check_retrieval, finalize, finalize_retrieval, judge, load_questions,
                            not_stopped, rescore_result, summarize)
from rag.llm import LLMError, LLMUnavailable, get_providers
from rag.model_client import ModelServerError
from rag.retriever import RETRIEVAL_MODES, RetrievalError, search


class RunStopped(RuntimeError):
    """The run cannot continue: no model answers even after waiting, or a service is down."""


class Pacer:
    """Keeps at least `seconds` between two LLM calls, so a run stays under the per-minute limit."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.last_call = 0.0

    def wait(self) -> None:
        remaining = self.last_call + self.seconds - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self.last_call = time.monotonic()


def call_llm(pacer: Pacer, what: str, function):
    """
    Run a function that makes one LLM call. When no model answers (429, outage), wait and retry:
    per-minute limits clear within a minute. If it still fails, the daily quota is probably used up.
    """
    for wait in (*config.EVAL_RETRY_WAITS, None):
        pacer.wait()
        try:
            return function()
        except LLMUnavailable as exc:
            if wait is None:
                raise RunStopped(f"{what}: no model answered, even after waiting ({exc}). The daily "
                                 "quota may be used up: try again tomorrow, or use --provider qwen.") from exc
            print(f"         {what}: no model answered, retrying in {wait} s ({exc})")
            time.sleep(wait)


def evaluate_question(question: Question, provider: str, use_judge: bool, pacer: Pacer,
                      not_indexed: list[str], mode: str, rerank: bool) -> Result:
    access_levels = config.ROLES[question.role]["access_levels"]
    try:
        answer = call_llm(pacer, f"answer {question.id}", lambda: answer_question(
            question.question, access_levels=access_levels, route=[provider], retrieval_mode=mode,
            history=question.history, rerank=rerank))
    except (ModelServerError, RetrievalError) as exc:
        raise RunStopped(f"{type(exc).__name__}: {exc}") from exc
    except LLMError as exc:                       # this request was rejected; the next one may work
        result = Result(question.id, question.type, question.role, question.question,
                        question.answerable, question.known_gap, answer=f"(no answer: {exc})")
        result.phrase_problems = ["no answer"]
        finalize(result)
        return result

    result = check_answer(question, answer, not_indexed)
    if use_judge and answer.confidential:      # the judge runs in the cloud: confidential answers stay here
        result.judge_reason = "not judged: confidential content stays on our own machines"
    elif use_judge and answer.llm is None:     # a fixed text: the phrase checks are enough
        result.judge_reason = "not judged: nothing relevant was found, the LLM was not asked"
    elif use_judge:
        for _attempt in range(2):                 # a second try when the judge's reply is not valid JSON
            try:
                result.verdict, result.judge_reason, result.judged_by = call_llm(
                    pacer, f"judge {question.id}", lambda: judge(question, answer.text))
                break
            except ValueError as exc:
                result.judge_reason = f"not judged: {exc}"
            except LLMError as exc:
                result.judge_reason = f"not judged: {exc}"
                break
    finalize(result)
    return result


def evaluate_retrieval(question: Question, not_indexed: list[str], mode: str, rerank: bool) -> Result:
    """Only the search: is what the LLM would need among the chunks it would get?"""
    started = time.perf_counter()
    try:
        hits = search(question.question, access_levels=config.ROLES[question.role]["access_levels"], mode=mode,
                      rerank=rerank)
    except (ModelServerError, RetrievalError) as exc:
        raise RunStopped(f"{type(exc).__name__}: {exc}") from exc
    result = check_retrieval(question, hits, round((time.perf_counter() - started) * 1000, 1), not_indexed)
    finalize_retrieval(result)
    return result


# --- Printing -----------------------------------------------------------------------

def result_line(result: Result, seconds: float) -> str:
    if not result.evidence_total:
        evidence = "-"
    elif result.evidence_not_indexed:
        evidence = "UNREAD"                       # not in the index at all: a reading problem
    else:
        evidence = "ok" if not result.missing_evidence else "MISSING"
    status = "LEAK" if result.leaks else result.status.upper()
    return (f"[{status:7s}] {result.id:>3} {result.type:13s} {result.role:8s} "
            f"source {result.source_rank or '-':>2}  evidence {evidence:7s}  "
            f"phrases {'ok' if not result.phrase_problems else 'FAIL':4s}  "
            f"cite {'ok' if not result.citation_problems else 'FAIL':4s}  judge {result.verdict or '-':8s} "
            f"{seconds:5.1f} s" + ("  (known gap)" if result.known_gap and result.status != "pass" else ""))


def retrieval_line(result: Result) -> str:
    if not result.evidence_total:
        evidence = "-"
    elif result.evidence_not_indexed:
        evidence = "UNREAD"
    else:
        evidence = "ok" if not result.missing_evidence else "MISSING"
    status = "LEAK" if result.leaks else result.status.upper()
    return (f"[{status:7s}] {result.id:>3} {result.type:13s} {result.role:8s} "
            f"source {result.source_rank or '-':>2}  evidence {evidence:7s} {result.retrieval_ms:6.0f} ms")


def print_retrieval_report(results: list[Result], summary: dict, settings: dict) -> None:
    s = summary
    print("\n" + "=" * 100)
    print(f"Retrieval only ({settings['retrieval_mode']}, reranker {'on' if settings.get('rerank') else 'off'}, "
          f"top_k {s['top_k']}): no LLM was asked.")
    print(f"Questions {s['questions']}: found {s['pass']}, not found {s['fail']} "
          f"(a question counts as found when a source file and all key facts are in the top {s['top_k']})")
    print(f"hit@1 {s['hit@1']}, hit@3 {s['hit@3']}, hit@{s['top_k']} {s['hit@k']} of {s['answerable']}, "
          f"MRR {s['mrr']:.2f} | key facts found {s['evidence_found']}/{s['with_evidence']} | "
          f"{s['avg_retrieval_ms']:.0f} ms on average")
    print(f"Nothing relevant found (the LLM would not be asked): {s['no_sources']} | "
          f"off-topic questions stopped: {s['off_topic_stopped']}/{s['off_topic']}")
    print(f"Security: {s['leaks']} leak(s)" + ("  <-- STOP AND FIX" if s["leaks"] else ""))
    missed = [result for result in results if result.status != "pass"]
    if missed:
        print("\nNot found:")
    for result in missed:
        print(f"  {result.id:>3} {result.type} -> {result.diagnosis}"
              + (f" ({result.known_gap})" if result.known_gap else ""))
        print(f"      Q: {result.question}")
        if result.missing_evidence:
            print(f"      missing: {result.missing_evidence}")
        if not_stopped(result):
            print(f"      off-topic, but the search returned {len(result.sources)} chunk(s)")
        for problem in result.leaks:
            print(f"      LEAK: {problem}")
        top = " | ".join(f"{hit['source']} > {hit['heading_path']}" for hit in result.sources[:3])
        print(f"      top sources: {top or '-'}")


def print_report(results: list[Result], summary: dict, settings: dict) -> None:
    s = summary
    print("\n" + "=" * 100)
    print(f"Answers: {settings['answer_model']} | judge: {settings['judge_model'] or 'off'} | "
          f"search {settings.get('retrieval_mode', 'dense')}, reranker {'on' if settings.get('rerank') else 'off'} | "
          f"top_k {s['top_k']} | chunk {settings['chunk_size_tokens']} tokens")
    print(f"Questions {s['questions']}: pass {s['pass']}, partial {s['partial']}, fail {s['fail']} "
          f"(known gaps among the partial and failed: {s['known_gaps_not_passed']})")
    print(f"Retrieval ({s['answerable']} answerable): hit@1 {s['hit@1']}, hit@3 {s['hit@3']}, "
          f"hit@{s['top_k']} {s['hit@k']}, MRR {s['mrr']:.2f} | "
          f"key facts shown to the LLM {s['evidence_found']}/{s['with_evidence']}")
    print(f"Answers: phrase checks {s['phrases_ok']}/{s['questions']} | judge: {s['verdict_correct']} correct, "
          f"{s['verdict_partial']} partial, {s['verdict_wrong']} wrong, {s['not_judged']} not judged")
    print(f"Citations: ok {s['citations_ok']}/{s['questions']}, invalid numbers in {s['invalid_citations']} | "
          f"{s['avg_cited']:.1f} of {s['top_k']} sources cited per answer on average")
    print(f"Unanswerable questions handled: {s['unanswerable_ok']}/{s['unanswerable']} | "
          f"off-topic stopped before the LLM: {s['off_topic_stopped']}/{s['off_topic']} | "
          f"LLM not asked (nothing relevant): {s['no_sources']}")
    print(f"Answers with numbers or codes not in their cited sources: {s['unverified_answers']}")
    print(f"Security: {s['leaks']} leak(s)" + ("  <-- STOP AND FIX" if s["leaks"] else ""))
    print(f"Time: retrieval {s['avg_retrieval_ms']:.0f} ms, answer {s['avg_llm_s']:.1f} s on average | "
          f"tokens in {s['input_tokens']:,}, out {s['output_tokens']:,}")

    print(f"\n{'Type':15s} {'n':>3s} {'pass':>5s} {'part.':>5s} {'fail':>5s}")
    for kind in sorted({result.type for result in results}):
        group = [result for result in results if result.type == kind]
        print(f"{kind:15s} {len(group):>3d} " + " ".join(
            f"{sum(r.status == status for r in group):>5d}" for status in ("pass", "partial", "fail")))

    not_passed = [result for result in results if result.status != "pass" and not result.known_gap]
    if not_passed:
        print("\nNot passed:")
        for result in not_passed:
            print_details(result)
    citation_problems = [result for result in results if result.citation_problems and result.status == "pass"]
    if citation_problems:
        print("\nPassed, but with citation problems:")
        for result in citation_problems:
            print(f"  {result.id:>3}  {'; '.join(result.citation_problems)}")
    unverified = [result for result in results if result.unverified]
    if unverified:
        print("\nNumbers or codes not found in the cited sources (a calculation, a wrong citation or a mistake?):")
        for result in unverified:
            print(f"  {result.id:>3}  {result.status:7s} {'; '.join(result.unverified)}")
    rewritten = [result for result in results if result.searched_as]
    if rewritten:
        print("\nFollow-up questions, as they were searched:")
        for result in rewritten:
            print(f"  {result.id:>3}  {result.question!r} -> {result.searched_as!r}")
    disagreements = [result for result in results if result.disagreement]
    if disagreements:
        print("\nPhrase checks and judge disagree (read these to learn how far the judge can be trusted):")
        for result in disagreements:
            phrases = "ok" if not result.phrase_problems else "; ".join(result.phrase_problems)
            print(f"  {result.id:>3}  phrases: {phrases}")
            print(f"       judge: {result.verdict} ({result.judge_reason})")
            print(textwrap.indent(textwrap.shorten(f"A: {result.answer}", 400), "       "))
    gaps = [result for result in results if result.known_gap]
    if gaps:
        print("\nKnown gaps (expected to fail until a later step):")
        for result in gaps:
            diagnosis = f" [{result.diagnosis}]" if result.diagnosis else ""
            print(f"  {result.id:>3}  {result.status}{diagnosis}: {result.known_gap}")


def print_details(result: Result) -> None:
    print(f"  {result.id:>3} {result.type} ({result.role}) -> {result.status}, {result.diagnosis}")
    print(f"      Q: {result.question}")
    if result.searched_as:
        print(f"      searched as: {result.searched_as}")
    print(textwrap.indent(textwrap.shorten(f"A: {result.answer}", 400), "      "))
    for problem in result.leaks:
        print(f"      LEAK: {problem}")
    if not_stopped(result):
        print(f"      off-topic, but the search returned {len(result.sources)} chunk(s) and the LLM was asked")
    if result.evidence_not_indexed:
        print(f"      not in the index at all (reading): {result.evidence_not_indexed}")
    elif result.missing_evidence:
        print(f"      not shown to the LLM: {result.missing_evidence}")
    if result.phrase_problems:
        print(f"      phrases: {'; '.join(result.phrase_problems)}")
    if result.citation_problems:
        print(f"      citations: {'; '.join(result.citation_problems)}")
    if result.verdict:
        print(f"      judge: {result.verdict} ({result.judge_reason})")
    top = " | ".join(f"{hit['source']} > {hit['heading_path']}" for hit in result.sources[:3])
    print(f"      top sources: {top or '-'}")


# --- Saving -------------------------------------------------------------------------

def run_settings(args) -> dict:
    """What was measured: without this, a saved result cannot be interpreted later."""
    providers = get_providers()
    judge_provider = config.LLM_TASKS["judge"]["providers"][0]
    rerank = not args.no_rerank
    return {
        "retrieval_mode": args.mode,
        "rerank": rerank,
        "rerank_candidates": config.RERANK_CANDIDATES if rerank else None,
        "rerank_min_score": config.RERANK_MIN_SCORE if rerank else None,
        "history_questions": config.HISTORY_QUESTIONS,
        "retrieval_only": args.retrieval_only,
        "answer_model": f"{args.provider} ({providers[args.provider].model})" if not args.retrieval_only else None,
        "judge_model": (f"{judge_provider} ({providers[judge_provider].model})"
                        if not (args.no_judge or args.retrieval_only) else None),
        "embedding_model": config.EMBEDDING_MODEL,
        "top_k": config.RETRIEVAL_TOP_K,
        "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        "pipeline_version": config.PIPELINE_VERSION,
        "only": args.only,
    }


def save(label: str, started: datetime, finished: bool, settings: dict, results: list[Result]) -> str:
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^\w-]+", "-", label).strip("-") or "run"
    path = config.EVAL_RESULTS_DIR / f"{started:%Y-%m-%d_%H%M%S}_{safe_label}.json"
    number = 2
    while path.exists():                       # never overwrite an earlier run
        path = path.with_name(f"{started:%Y-%m-%d_%H%M%S}_{safe_label}_{number}.json")
        number += 1
    record = {
        "label": label,
        "started_at": started.isoformat(timespec="seconds"),
        "finished": finished,
        "settings": settings,
        "summary": summarize(results),
        "results": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.relative_to(config.ROOT_DIR))


def rescore(path: Path) -> None:
    """Check the saved answers of an earlier run again: no LLM call, no quota."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        questions = {question.id: question for question in load_questions()}
    except (OSError, json.JSONDecodeError, QuestionSetError) as exc:
        print(f"STOPPED: {exc}")
        sys.exit(2)

    if saved["settings"].get("retrieval_only"):
        print("STOPPED: a retrieval-only run has no answers to check again; run it again instead (it is free).")
        sys.exit(2)
    results = []
    for item in saved["results"]:
        if item["id"] not in questions:
            print(f"Note: question {item['id']} is no longer in the question set; skipped.")
            continue
        if item.get("verdict") and not item.get("judged_by"):     # runs from before step 14
            item["judged_by"] = saved["settings"].get("judge_model") or ""
        results.append(rescore_result(questions[item["id"]], item))
    for result in results:
        print(result_line(result, 0.0))
    settings = {**saved["settings"], "rescored_from": path.name}
    print("\nRescored: answers, judge verdicts, retrieval and security results are those of the saved run.")
    print_report(results, summarize(results), settings)
    new_path = save(f"{saved['label']} rescored", datetime.now(), saved["finished"], settings, results)
    print(f"\nSaved: {new_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CIA evaluation question set.")
    parser.add_argument("--label", default="run", help="A name for this run, e.g. baseline or bm25.")
    parser.add_argument("--only", help="Comma-separated question ids, e.g. 1,5,13.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge (phrase checks only).")
    parser.add_argument("--provider", default=config.LLM_TASKS["answer"]["providers"][0],
                        choices=config.LLM_TASKS["answer"]["providers"],
                        help="The only model that answers during this run (no fallback).")
    parser.add_argument("--rescore", type=Path, metavar="FILE",
                        help="Repeat the checks that need no LLM on a saved run, with the current rules.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Only measure the search (no LLM call, no quota).")
    parser.add_argument("--mode", default=config.RETRIEVAL_MODE, choices=RETRIEVAL_MODES,
                        help="Search with meaning and keywords (hybrid), meaning only or keywords only.")
    parser.add_argument("--no-rerank", action="store_true", help="Keep the order of the search (no reranker).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="         (log) %(message)s")
    if args.rescore:
        rescore(args.rescore)
        return

    try:
        questions = load_questions()
        if args.only:
            if not re.fullmatch(r"\d+(,\d+)*", args.only.replace(" ", "")):
                raise QuestionSetError("--only takes question ids separated by commas, like 1,5,13.")
            wanted = {int(part) for part in args.only.replace(" ", "").split(",")}
            questions = [question for question in questions if question.id in wanted]
            if missing := wanted - {question.id for question in questions}:
                raise QuestionSetError(f"No question with id {sorted(missing)}.")
        if args.retrieval_only and (follow_ups := [q.id for q in questions if q.history]):
            questions = [question for question in questions if not question.history]
            print(f"Skipped follow-up questions {follow_ups}: they are rewritten by an LLM before the search.")
            if not questions:
                raise QuestionSetError("No question left to search.")
        problems, not_indexed = check_against_index(questions)
    except QuestionSetError as exc:
        print(f"STOPPED: {exc}")
        sys.exit(2)
    except Exception as exc:
        print(f"STOPPED: cannot read the collection in Qdrant ({type(exc).__name__}: {exc})")
        sys.exit(2)
    known_gaps = {question.id for question in questions if question.known_gap}
    blocking = {qid: found for qid, found in problems.items() if qid not in known_gaps}
    if blocking:                               # most likely a typo in the question file
        print("STOPPED: the question set does not match the indexed documents:")
        for qid, found in sorted(blocking.items()):
            print("\n".join(f"  - question {qid}: {problem}" for problem in found))
        sys.exit(2)
    for qid, found in sorted(problems.items()):  # known gaps: expected, so only a warning
        print("\n".join(f"Note: question {qid} (known gap): {problem}" for problem in found))

    settings = run_settings(args)
    rerank = settings["rerank"]
    if args.retrieval_only:
        print(f"{len(questions)} questions, search only ({args.mode}, reranker {'on' if rerank else 'off'}), "
              f"no LLM calls.\n")
    else:
        calls = len(questions) * (1 if args.no_judge else 2) + sum(1 for q in questions if q.history)
        print(f"{len(questions)} questions, about {calls} LLM calls, at least "
              f"{config.EVAL_SECONDS_BETWEEN_LLM_CALLS:.0f} s apart. Answers: {settings['answer_model']}, "
              f"search: {args.mode}, reranker {'on' if rerank else 'off'}\n")

    started = datetime.now()
    pacer = Pacer(config.EVAL_SECONDS_BETWEEN_LLM_CALLS)
    results: list[Result] = []
    finished = False
    try:
        for question in questions:
            if args.retrieval_only:
                result = evaluate_retrieval(question, not_indexed.get(question.id, []), args.mode, rerank)
                results.append(result)
                print(retrieval_line(result))
                continue
            question_started = time.perf_counter()
            result = evaluate_question(question, args.provider, not args.no_judge, pacer,
                                       not_indexed.get(question.id, []), args.mode, rerank)
            results.append(result)
            print(result_line(result, time.perf_counter() - question_started))
        finished = True
    except RunStopped as exc:
        print(f"\nSTOPPED after {len(results)} of {len(questions)} questions: {exc}")
    except KeyboardInterrupt:
        print(f"\nInterrupted after {len(results)} of {len(questions)} questions.")

    if results:
        summary = summarize(results)
        (print_retrieval_report if args.retrieval_only else print_report)(results, summary, settings)
        path = save(args.label, started, finished, settings, results)
        print(f"\nSaved{'' if finished else ' (incomplete run)'}: {path}")
    if not finished:
        sys.exit(2)
    sys.exit(1 if summarize(results)["leaks"] else 0)


if __name__ == "__main__":
    main()
