"""
Step 15: check the keyword search (BM25) and show what each kind of search finds.

Part 1 checks the Turkish tokenizer and the BM25 weights (needs nothing else).
Part 2 searches the real collection three ways for some questions of the evaluation set and prints
at which rank the chunk with the answer is found: by meaning (dense), by keywords (sparse) and by
both (hybrid), before the reranker (see scripts/check_rerank.py). Needs Qdrant and the model server;
no LLM call.

Usage (from the project root):
    python -m scripts.check_hybrid
    python -m scripts.check_hybrid --ids 8,13,37,38      # other questions for the table
"""
import argparse
import logging
import sys

import config
from rag.evaluation import contains, load_questions
from rag.retriever import RETRIEVAL_MODES, search
from rag.sparse import encode_document, tokenize
from scripts.check_env import run_check

DEFAULT_IDS = "8,13,37,38,39,40,41,42,43"     # one meaning question (8) and the look-ups of step 15


# --- Part 1 -------------------------------------------------------------------------

def expect(text: str, present: set[str], absent: set[str] = frozenset()) -> str:
    tokens = set(tokenize(text))
    if not present <= tokens or tokens & absent:
        raise AssertionError(f"{text!r} -> {sorted(tokens)}")
    return f"{text!r} -> {sorted(tokens)}"


def check_codes() -> str:
    whole = {"vpnerr204", "vpn", "err", "204"}
    expect("VPN-ERR-204", whole)
    expect("VPN-ERR204", whole)
    return expect("vpn err 204", {"vpn", "err", "204"})


def check_turkish() -> str:
    if not tokenize("Yöneticilere") == tokenize("yöneticiler") == tokenize("YÖNETİCİ"):
        raise AssertionError("suffixes or upper case change the word")
    if tokenize("musteri listesi") != tokenize("Müşteri Listesi"):
        raise AssertionError("typing without Turkish letters changes the words")
    return f"Yöneticilere = yöneticiler = YÖNETİCİ -> {tokenize('Yöneticilere')}, musteri = Müşteri"


def check_weights() -> str:
    once, three = encode_document("izin").values[0], encode_document("izin izin izin").values[0]
    if not once < three < 3 * once:
        raise AssertionError(f"one: {once}, three: {three}")
    return f"one 'izin': {once:.2f}, three: {three:.2f} (more, but not three times as much)"


# --- Part 2 -------------------------------------------------------------------------

def answer_rank(question, mode: str) -> int | None:
    """Rank of the first chunk that holds the answer: all evidence, or the expected file if there is none."""
    hits = search(question.question, top_k=config.RETRIEVAL_PREFETCH,
                  access_levels=config.ROLES[question.role]["access_levels"], mode=mode, rerank=False)
    for rank, hit in enumerate(hits, start=1):
        if question.evidence:
            if all(any(contains(hit.text, phrase) for phrase in alternatives) for alternatives in question.evidence):
                return rank
        elif hit.source in question.sources:
            return rank
    return None


def print_rank_table(ids: list[int]) -> None:
    questions = {question.id: question for question in load_questions()}
    print(f"\n{'id':>3}  {'question':50s} " + " ".join(f"{mode:>7s}" for mode in RETRIEVAL_MODES))
    for question_id in ids:
        question = questions[question_id]
        ranks = [answer_rank(question, mode) for mode in RETRIEVAL_MODES]
        cells = " ".join(f"{(str(rank) if rank else '-'):>7s}" for rank in ranks)
        print(f"{question_id:>3}  {question.question[:50]:50s} {cells}")
    print(f"\nRank of the chunk with the answer among the first {config.RETRIEVAL_PREFETCH} ('-': not found). "
          f"The LLM sees the first {config.RETRIEVAL_TOP_K}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the keyword search and compare search modes.")
    parser.add_argument("--ids", default=DEFAULT_IDS, help="Question ids for the rank table.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="       (log) %(message)s")

    print("Part 1: tokenizer and BM25 weights")
    checks = {
        "Apostrophe suffix dropped": lambda: expect("Murat Kaya'nın", {"murat", "kaya"}, {"nin", "nın"}),
        "Kaya and Kayaalp stay apart": lambda: expect("Murat Kayaalp", {"kayaa"}, {"kaya"}),
        "Code spellings meet": check_codes,
        "Numbers keep their separators": lambda: expect("180.000 TL", {"180000"}, {"000", "180"}),
        "Turkish suffixes and letters": check_turkish,
        "Repeated words count less and less": check_weights,
    }
    results = [run_check(name, check) for name, check in checks.items()]
    print(f"\nTOTAL: {sum(results)}/{len(results)} checks passed")

    print("\nPart 2: where the answer is found (Qdrant and the model server, no LLM)")
    try:
        print_rank_table([int(part) for part in args.ids.split(",")])
    except Exception as exc:
        print(f"STOPPED: {type(exc).__name__}: {exc}")
        print("Was the collection rebuilt for step 15? python -m scripts.index_documents --recreate")
        sys.exit(1)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
