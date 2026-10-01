"""
Step 12: compare two evaluation runs, question by question.

The totals can move by one or two questions from run to run even when nothing changed, because an
LLM does not answer the same way every time. The list of questions that changed, and why, matters more.

Usage (from the project root):
    python -m scripts.compare_eval --last 2                      # the two newest runs
    python -m scripts.compare_eval evaluation/results/OLD.json evaluation/results/NEW.json
"""
import argparse
import json
import sys
from pathlib import Path

import config

STATUS_ORDER = {"fail": 0, "partial": 1, "pass": 2}
SHOWN_METRICS = ["pass", "partial", "fail", "hit@1", "hit@3", "hit@k", "mrr", "evidence_found", "phrases_ok",
                 "verdict_correct", "citations_ok", "invalid_citations", "avg_cited", "unverified_answers",
                 "unanswerable_ok", "off_topic_stopped", "no_sources", "leaks",
                 "avg_retrieval_ms", "avg_llm_s", "input_tokens"]
LOWER_IS_BETTER = {"partial", "fail", "invalid_citations", "unverified_answers", "leaks", "avg_retrieval_ms",
                   "avg_llm_s", "input_tokens"}
NEITHER = {"avg_cited", "no_sources"}   # more or fewer is not better or worse by itself


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"STOPPED: cannot read {path}: {exc}")
        sys.exit(2)


def describe(result: dict) -> str:
    status = "LEAK" if result["leaks"] else result["status"]
    citations = "-" if "citation_problems" not in result else ("ok" if not result["citation_problems"] else "FAIL")
    return f"{status} (source {result['source_rank'] or '-'}, judge {result['verdict'] or '-'}, cite {citations})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two CIA evaluation runs.")
    parser.add_argument("files", nargs="*", type=Path, help="The older and the newer result file.")
    parser.add_argument("--last", type=int, choices=[2], help="Compare the two newest runs.")
    args = parser.parse_args()

    if args.last:
        files = sorted(config.EVAL_RESULTS_DIR.glob("*.json"), key=lambda path: path.stat().st_mtime)[-2:]
    else:
        files = args.files
    if len(files) != 2:
        parser.error("give two result files, or --last 2 (with at least two runs saved)")
    old, new = load(files[0]), load(files[1])
    print(f"OLD: {files[0].name}  ({old['label']}{'' if old['finished'] else ', incomplete'})")
    print(f"NEW: {files[1].name}  ({new['label']}{'' if new['finished'] else ', incomplete'})")

    changed_settings = {key for key in old["settings"].keys() | new["settings"].keys()
                        if old["settings"].get(key) != new["settings"].get(key)}
    print("\nSettings that changed:" if changed_settings else "\nSettings: the same")
    for key in sorted(changed_settings):
        print(f"  {key}: {old['settings'].get(key)} -> {new['settings'].get(key)}")

    print(f"\n{'':18s} {'old':>9s} {'new':>9s}")
    for key in SHOWN_METRICS:
        before, after = old["summary"].get(key), new["summary"].get(key)
        if before is None or after is None:
            continue
        mark = ""
        if after != before and key in NEITHER:
            mark = "  changed"
        elif after != before:
            better = (after < before) if key in LOWER_IS_BETTER else (after > before)
            mark = "  better" if better else "  worse"
        print(f"{key:18s} {before:>9} {after:>9}{mark}")

    old_results = {result["id"]: result for result in old["results"]}
    new_results = {result["id"]: result for result in new["results"]}
    common = sorted(old_results.keys() & new_results.keys())
    groups = {"Fixed": [], "Broken": [], "Other changes": []}
    for question_id in common:
        before, after = old_results[question_id], new_results[question_id]
        rank_before = -1 if before["leaks"] else STATUS_ORDER[before["status"]]
        rank_after = -1 if after["leaks"] else STATUS_ORDER[after["status"]]
        line = f"  {question_id:>3} {after['type']:13s} {describe(before)} -> {describe(after)}"
        if rank_after > rank_before:
            groups["Fixed"].append(line)
        elif rank_after < rank_before:
            groups["Broken"].append(line + (f"  [{after['diagnosis']}]" if after["diagnosis"] else ""))
        elif describe(before) != describe(after):
            groups["Other changes"].append(line)

    for title, lines in groups.items():
        if lines:
            print(f"\n{title} ({len(lines)}):")
            print("\n".join(lines))
    if not any(groups.values()):
        print("\nNo question changed.")
    only_old, only_new = sorted(old_results.keys() - new_results.keys()), sorted(new_results.keys() - old_results.keys())
    if only_old or only_new:
        print(f"\nNot in both runs: only old {only_old or '-'}, only new {only_new or '-'}")


if __name__ == "__main__":
    main()
