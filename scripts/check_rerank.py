"""
Step 16: check the reranker and its threshold (config.RERANK_MIN_SCORE).

Part 1 shows, for some questions, at which rank the chunk with the answer is before the reranker
(the search) and after it.
Part 2 scores every question of the evaluation set with the reranker:
  - answerable questions: the score of the chunks that hold the answer. The threshold must stay well
    below the lowest of them, or answers are dropped (also for questions that are not in the set);
  - unanswerable questions: the best score of any chunk. Off-topic ones should fall below it, so that
    the LLM is not asked. Related ones ("İzmir ofisinde otopark var mı?") may stay above it: then
    the LLM says that the documents do not answer it.
It suggests a threshold and checks the one in config.py.

Needs Qdrant and the model server; no LLM call. Follow-up questions are skipped: they are rewritten
by an LLM before the search (step 17).

Usage (from the project root):
    python -m scripts.check_rerank
    python -m scripts.check_rerank --ids 8,13,41      # other questions for the table of part 1
"""
import argparse
import logging
import math
import sys
import time

import config
from rag.evaluation import Question, contains, load_questions
from rag.retriever import Hit, find_candidates, rerank_hits

DEFAULT_IDS = "8,13,37,38,39,40,41,42,43"
MARGIN = 10              # the suggested threshold is this many times below the weakest chunk with an answer


def holds_answer(question: Question, hit: Hit) -> bool:
    """The chunk has every key fact of the question (or, without key facts, is from an expected file)."""
    if question.evidence:
        return all(any(contains(hit.text, phrase) for phrase in alternatives) for alternatives in question.evidence)
    return hit.source in question.sources


def answer_rank(question: Question, hits: list[Hit]) -> int | None:
    return next((rank for rank, hit in enumerate(hits, start=1) if holds_answer(question, hit)), None)


def needed_score(question: Question, reranked: list[Hit]) -> float | None:
    """
    The lowest score the threshold must keep: for every key fact, the best chunk that has it, and of
    those the weakest (a two-part question needs both of its chunks). None: no chunk has the answer.
    """
    if not question.evidence:
        scores = [hit.score for hit in reranked if hit.source in question.sources]
        return max(scores) if scores else None
    per_fact = []
    for alternatives in question.evidence:
        scores = [hit.score for hit in reranked if any(contains(hit.text, phrase) for phrase in alternatives)]
        if not scores:
            return None
        per_fact.append(max(scores))
    return min(per_fact)


def facts_shown(question: Question, hits: list[Hit]) -> bool:
    """All key facts are among the chunks the LLM would see."""
    return all(any(contains(hit.text, phrase) for hit in hits for phrase in alternatives)
               for alternatives in question.evidence)


def round_down(value: float) -> float:
    """To one significant digit, downwards: 0.0347 -> 0.03."""
    if value <= 0:
        return 0.0
    unit = 10 ** math.floor(math.log10(value))
    return round(math.floor(value / unit) * unit, 10)


def search_both(question: Question) -> tuple[list[Hit], list[Hit], float]:
    """The candidates of the search, the same candidates after the reranker, and the reranker's time."""
    access_levels = config.ROLES[question.role]["access_levels"]
    hits = find_candidates(question.question, config.RERANK_CANDIDATES, access_levels=access_levels)
    started = time.perf_counter()
    reranked = rerank_hits(question.question, hits)
    return hits, reranked, (time.perf_counter() - started) * 1000


def print_rank_table(questions: dict[int, Question], ids: list[int]) -> None:
    print(f"\n{'id':>3}  {'question':50s} {'search':>7s} {'+rerank':>8s} {'score':>7s}")
    first = {"search": 0, "rerank": 0}
    for question_id in ids:
        question = questions[question_id]
        hits, reranked, _ = search_both(question)
        before, after = answer_rank(question, hits), answer_rank(question, reranked)
        first["search"] += before == 1
        first["rerank"] += after == 1
        score = f"{reranked[after - 1].score:.3f}" if after else "-"
        print(f"{question_id:>3}  {question.question[:50]:50s} {before or '-':>7} {after or '-':>8} {score:>7s}")
    print(f"\nAnswer chunk first: search {first['search']}/{len(ids)}, with the reranker {first['rerank']}/{len(ids)} "
          f"(rank among the first {config.RERANK_CANDIDATES} candidates; '-': not among them)")


def check_threshold(questions: list[Question]) -> bool:
    threshold, top_k = config.RERANK_MIN_SCORE, config.RETRIEVAL_TOP_K
    print(f"\n{'id':>3}  {'type':13s} {'kind':10s} {'score':>7s}  at {threshold}")
    needed: dict[int, float] = {}
    off_topic_scores: dict[int, float] = {}
    shown = {"search": 0, "rerank": 0}
    with_evidence = 0
    timings = []
    for question in questions:
        hits, reranked, elapsed = search_both(question)
        timings.append(elapsed)
        best = reranked[0].score if reranked else 0.0
        if question.answerable:
            score = needed_score(question, reranked)
            if score is None:
                print(f"{question.id:>3}  {question.type:13s} {'answer':10s} {'-':>7s}  no candidate has the answer"
                      + (f" (known gap: {question.known_gap})" if question.known_gap else ""))
                continue
            needed[question.id] = score
            print(f"{question.id:>3}  {question.type:13s} {'answer':10s} {score:7.3f}  "
                  f"{'kept' if score >= threshold else 'DROPPED'}")
            if question.evidence:
                with_evidence += 1
                kept = [hit for hit in reranked if hit.score >= threshold][:top_k]
                shown["search"] += facts_shown(question, hits[:top_k])
                shown["rerank"] += facts_shown(question, kept)
        else:
            kind = "off-topic" if question.off_topic else "no answer"
            if question.off_topic:
                off_topic_scores[question.id] = best
            print(f"{question.id:>3}  {question.type:13s} {kind:10s} {best:7.3f}  "
                  f"{'stopped' if best < threshold else 'to the LLM'}")

    print(f"\nKey facts among the {top_k} chunks the LLM sees: search {shown['search']}/{with_evidence}, "
          f"with the reranker {shown['rerank']}/{with_evidence}")
    print(f"Reranker time: {sum(timings) / len(timings):.0f} ms per question on average "
          f"({config.RERANK_CANDIDATES} candidates)")
    if not needed:
        print("No answerable question to check the threshold with.")
        return False
    weakest = min(needed, key=needed.get)
    suggested = round_down(needed[weakest] / MARGIN)
    print(f"Weakest chunk with an answer: {needed[weakest]:.3f} (question {weakest})")
    if off_topic_scores:
        strongest = max(off_topic_scores, key=off_topic_scores.get)
        print(f"Strongest off-topic question: {off_topic_scores[strongest]:.4f} (question {strongest})")
    print(f"Suggested threshold: {suggested} ({MARGIN} times below the weakest chunk with an answer)")

    dropped = [qid for qid, score in needed.items() if score < threshold]
    stopped = [qid for qid, score in off_topic_scores.items() if score < threshold]
    ok = not dropped
    if dropped:
        print(f"[FAIL] RERANK_MIN_SCORE = {threshold} drops the answer of question(s) {dropped}: lower it.")
    elif threshold > suggested:
        print(f"[WARN] RERANK_MIN_SCORE = {threshold} keeps every answer here, but with less than a {MARGIN}x "
              f"margin: answers to other questions may be dropped. Consider {suggested}.")
    else:
        print(f"[OK]   RERANK_MIN_SCORE = {threshold} keeps every answer, with a margin.")
    if off_topic_scores:
        print(f"Off-topic questions stopped before the LLM: {len(stopped)}/{len(off_topic_scores)}"
              + ("" if len(stopped) == len(off_topic_scores) else
                 " (the others reach the LLM, which must say the documents do not answer them)"))
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the reranker and its threshold.")
    parser.add_argument("--ids", default=DEFAULT_IDS, help="Question ids for the rank table of part 1.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="       (log) %(message)s")

    try:
        questions = [question for question in load_questions() if not question.history]
        by_id = {question.id: question for question in questions}
        print("Part 1: where the chunk with the answer is, before and after the reranker")
        print_rank_table(by_id, [int(part) for part in args.ids.split(",") if int(part) in by_id])
        print("\nPart 2: reranker scores of the evaluation set and the threshold")
        ok = check_threshold(questions)
    except Exception as exc:
        print(f"STOPPED: {type(exc).__name__}: {exc}")
        print("Are Qdrant and the model server running (VPN)?")
        sys.exit(2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
