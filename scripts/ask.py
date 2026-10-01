"""
Step 10: ask CIA a question from the terminal (step 13: shows which sources the answer used).

Usage (from the project root):
    python -m scripts.ask "8 yıllık bir çalışanın kaç gün yıllık izni var?"
    python -m scripts.ask "..." --show-context          # also print exactly what the LLM saw
    python -m scripts.ask "..." --include-superseded    # search old versions too
    python -m scripts.ask "..." --top-k 8               # show more chunks to the LLM
    python -m scripts.ask "..." --role manager          # search with the rights of a role (step 14)
    python -m scripts.ask "..." --mode sparse           # hybrid (default), dense or sparse search (step 15)
    python -m scripts.ask "..." --no-rerank             # without the reranker (step 16)
    python -m scripts.ask "Peki yöneticiler için?" --history "Çalışanların yıllık izni kaç gün?"
                                                        # a follow-up question (step 17); repeat --history

There is no login here: whoever can run commands on the server can read the documents anyway.
"""
import argparse
import logging
import sys
import textwrap

import config
from rag.answer import Answer, answer_question, build_messages
from rag.citations import best_excerpt
from rag.llm import LLMError
from rag.model_client import ModelServerError
from rag.retriever import RETRIEVAL_MODES, RetrievalError


def print_answer(answer: Answer, show_context: bool = False) -> None:
    if answer.rewritten:
        print(f"Searched as: {answer.search_question}  ({answer.rewrite.summary() if answer.rewrite else ''})\n")
    if show_context and answer.sources:
        print("=" * 30 + " What the LLM saw " + "=" * 30)
        for message in build_messages(answer.search_question, answer.sources):
            print(f"[{message['role']}]\n{message['content']}\n")
        print("=" * 78 + "\n")

    citations = answer.citations
    print("Answer:")
    print(textwrap.indent(citations.clean_text, "  "))
    if citations.not_found:
        print("\n(The answer says the documents do not contain this.)")
    elif not citations.numbers:
        print("\nWARNING: the answer cites no source.")
    if citations.invalid:
        print(f"WARNING: the answer cites {list(citations.invalid)}, which are not in the source list.")
    for item in answer.unverified:
        where = f"it is in source {list(item.found_in)}" if item.found_in else "it is in no source"
        print(f"CHECK: {item.fact!r} is not in the cited source(s) {list(item.cited)}; {where} "
              f"(a calculation, or a mistake?)")

    if answer.cited_sources():
        print("\nSources used:")
        for number, hit in answer.cited_sources():
            print(f"  [{number}] {hit.score:.3f}  {hit.label()}")
            print(f"        {best_excerpt(hit.content, citations.clean_text, config.SOURCE_EXCERPT_CHARS)}")
    if answer.unused_sources():
        print("\nFound but not used:")
        for number, hit in answer.unused_sources():
            print(f"  [{number}] {hit.score:.3f}  {hit.label()}")
    if answer.confidential:
        print("\n(Confidential sources: answered by a local model only.)")
    llm = answer.llm.summary() if answer.llm else "LLM not asked: nothing relevant was found"
    print(f"\n{llm} | retrieval {answer.retrieval_ms:.0f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask CIA a question.")
    parser.add_argument("question", help="The question, in quotes.")
    parser.add_argument("--show-context", action="store_true", help="Print the messages sent to the LLM.")
    parser.add_argument("--include-superseded", action="store_true", help="Also search superseded documents.")
    parser.add_argument("--top-k", type=int, default=config.RETRIEVAL_TOP_K, help="Chunks shown to the LLM.")
    parser.add_argument("--role", default="employee", choices=list(config.ROLES),
                        help="Whose rights to search with (default: employee).")
    parser.add_argument("--mode", default=config.RETRIEVAL_MODE, choices=RETRIEVAL_MODES,
                        help="Search with meaning and keywords (hybrid), meaning only or keywords only.")
    parser.add_argument("--no-rerank", action="store_true", help="Keep the order of the search (no reranker).")
    parser.add_argument("--history", action="append", default=[], metavar="QUESTION",
                        help="An earlier question of the conversation; repeat it for several, oldest first.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="       (log) %(message)s")

    try:
        answer = answer_question(args.question, top_k=args.top_k, include_superseded=args.include_superseded,
                                 access_levels=config.ROLES[args.role]["access_levels"], retrieval_mode=args.mode,
                                 history=args.history, rerank=not args.no_rerank)
    except (ModelServerError, RetrievalError, LLMError) as exc:
        print(f"STOPPED: {type(exc).__name__}: {exc}")
        sys.exit(1)
    print_answer(answer, show_context=args.show_context)


if __name__ == "__main__":
    main()
