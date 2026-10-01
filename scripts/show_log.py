"""
Step 19: what the question log of the web interface says.

The most useful part is the list of questions the documents could not answer: they show which
documents are missing or hard to find. Errors and answers with unverified numbers come next.

Usage (from the project root):
    python -m scripts.show_log                    # this month
    python -m scripts.show_log --month 2026-10
    python -m scripts.show_log --last 20          # also list the last 20 questions
"""
import argparse
from collections import Counter
from statistics import mean

from rag.query_log import log_path, read_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the CIA question log.")
    parser.add_argument("--month", help="Month to read, like 2026-10 (default: this month).")
    parser.add_argument("--last", type=int, default=0, help="Also list the last N questions.")
    args = parser.parse_args()

    records = read_log(args.month)
    print(f"{log_path(args.month).name}: {len(records)} question(s)")
    if not records:
        return

    answered = [r for r in records if "error" not in r]
    nothing = [r for r in answered if not r.get("llm_asked")]
    not_found = [r for r in answered if r.get("llm_asked") and r.get("not_found")]
    unverified = [r for r in answered if r.get("unverified")]
    errors = Counter(r["error"] for r in records if "error" in r)
    llm_times = [r["llm_ms"] for r in answered if r.get("llm_ms")]
    print(f"  answered {len(answered) - len(nothing) - len(not_found)}, not in the documents {len(not_found)}, "
          f"nothing relevant found {len(nothing)}, errors {sum(errors.values())}")
    print(f"  confidential {sum(1 for r in answered if r.get('confidential'))}, "
          f"rewritten follow-ups {sum(1 for r in answered if r.get('searched_as'))}, "
          f"answered by the fallback model {sum(1 for r in answered if r.get('fallback'))}, "
          f"unverified numbers or codes {len(unverified)}")
    if answered:
        print(f"  average time: search {mean(r['retrieval_ms'] for r in answered):.0f} ms"
              + (f", answer {mean(llm_times) / 1000:.1f} s" if llm_times else ""))
    print(f"  users: {len({r['user'] for r in records})}")

    gaps = Counter(r["question"].strip() for r in nothing + not_found)
    if gaps:
        print("\nNot answered from the documents (missing or hard-to-find documents?):")
        for question, count in gaps.most_common(20):
            print(f"  {count:>3}x  {question}")
    if errors:
        print("\nErrors:")
        for name, count in errors.most_common():
            print(f"  {count:>3}x  {name}")
    if unverified:
        print("\nAnswers with numbers or codes not found in their sources (check them):")
        for record in unverified[-10:]:
            print(f"  {record['time']}  {record['question'][:60]!r}: {', '.join(record['unverified'])}")
    if args.last:
        print(f"\nLast {args.last}:")
        for record in records[-args.last:]:
            outcome = record.get("error") or ("nothing found" if not record.get("llm_asked")
                                               else "not in documents" if record.get("not_found") else "answered")
            print(f"  {record['time']}  {record['user']:10s} {outcome:16s} {record['question'][:70]}")


if __name__ == "__main__":
    main()
