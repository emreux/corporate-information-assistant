"""
Step 9: check the indexer and the collection. Needs Qdrant and the A5000 (VPN).

Part 1 plays the sync rules through on small text files in a separate test collection
(cia_documents_test), which is deleted afterwards. The real collection is not touched.
Part 2 checks the real collection built by scripts.index_documents.
Part 3 runs the first searches: no LLM yet, only "does Qdrant find the right chunks?".

Usage (from the project root):
    python -m scripts.index_documents
    python -m scripts.check_index
"""
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

from qdrant_client import models

import config
from rag.indexer import (DENSE_VECTOR, IndexerError, ensure_collection, get_qdrant, sync_folder)
from rag.metadata import CatalogEntry, load_catalog
from rag.model_client import get_model_client
from scripts.check_env import run_check
from scripts.check_metadata import EXPECTED_FIELDS

TEST_COLLECTION = f"{config.QDRANT_COLLECTION}_test"
POINT_FIELDS = EXPECTED_FIELDS | {"fingerprint", "indexed_at"}

# Test documents (Turkish on purpose, like real ones). Version 1 has three sections, version 2 one.
THREE_SECTIONS = "## 1. Giriş\n\nBirinci bölüm.\n\n## 2. Kurallar\n\nİkinci bölüm.\n\n## 3. Ekler\n\nÜçüncü bölüm."
ONE_SECTION = "## 1. Giriş\n\nKısaltılmış tek bölüm."

LEAVE_QUESTION = "8 yıllık bir çalışanın kaç gün yıllık izni var?"
ACTIVE_ONLY = models.Filter(must=[models.FieldCondition(key="status", match=models.MatchValue(value="active"))])
SEARCHES = [
    (LEAVE_QUESTION, None),
    (LEAVE_QUESTION, ACTIVE_ONLY),
    ("vpn err 203 aldım napcam", None),
]


# --- Part 1: sync rules on a test collection ----------------------------------------

class Playground:
    """A throw-away folder and collection for playing the sync rules through."""

    def __init__(self, client, folder: Path) -> None:
        self.client, self.folder = client, folder
        self.catalog = {"a.txt": CatalogEntry("internal"), "b.txt": CatalogEntry("internal")}

    def write(self, name: str, text: str) -> None:
        (self.folder / name).write_text(text, encoding="utf-8")

    def sync(self, dry_run: bool = False) -> dict[str, str]:
        results = sync_folder(self.folder, self.catalog, self.client, TEST_COLLECTION, dry_run=dry_run)
        failed = [f"{r.source}: {r.detail}" for r in results if r.action == "failed"]
        if failed:
            raise AssertionError(f"indexing failed: {failed}")
        return {result.source: result.action for result in results}

    def payloads(self, source: str | None = None) -> list[dict]:
        condition = None if source is None else models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        points, _ = self.client.scroll(TEST_COLLECTION, scroll_filter=condition, limit=1000, with_vectors=False)
        return [point.payload for point in points]


def expect(actual, expected, what: str) -> None:
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected}, got {actual}")


def run_sync_scenarios(client) -> list[bool]:
    ensure_collection(client, TEST_COLLECTION, recreate=True)
    with tempfile.TemporaryDirectory() as tmp:
        play = Playground(client, Path(tmp))
        play.write("a.txt", THREE_SECTIONS)
        play.write("b.txt", ONE_SECTION)
        play.write("~$a.txt", "Office lock file, must be ignored")

        def first_run() -> str:
            expect(play.sync(), {"a.txt": "indexed", "b.txt": "indexed"}, "actions")
            expect(len(play.payloads("a.txt")), 3, "points of a.txt")
            return f"{len(play.payloads())} points, lock file ignored"

        def second_run() -> str:
            expect(play.sync(), {"a.txt": "unchanged", "b.txt": "unchanged"}, "actions")
            expect(len(play.payloads()), 4, "points after a second run")
            return "nothing re-indexed, no duplicates"

        def changed_file() -> str:
            play.write("a.txt", ONE_SECTION + " Yeni sürüm.")          # three chunks become one
            expect(play.sync(), {"a.txt": "indexed", "b.txt": "unchanged"}, "actions")
            payloads = play.payloads("a.txt")
            expect(len(payloads), 1, "points of a.txt after shrinking")
            expect(len({p["fingerprint"] for p in payloads}), 1, "fingerprints of a.txt")
            return "old version replaced, no ghost chunks"

        def changed_label() -> str:
            play.catalog["b.txt"] = CatalogEntry("restricted")
            expect(play.sync(), {"a.txt": "unchanged", "b.txt": "indexed"}, "actions")
            expect({p["access_level"] for p in play.payloads("b.txt")}, {"restricted"}, "labels of b.txt")
            return "file unchanged, new label applied"

        def dry_run() -> str:
            play.write("c.txt", ONE_SECTION)
            try:
                actions = play.sync(dry_run=True)
            finally:
                (play.folder / "c.txt").unlink()
            expect(actions, {"a.txt": "unchanged", "b.txt": "unchanged", "c.txt": "would index"}, "actions")
            expect(len(play.payloads()), 2, "points after a dry run")
            return "reported, nothing written"

        def deleted_file() -> str:
            (play.folder / "b.txt").unlink()
            expect(play.sync(), {"a.txt": "unchanged", "b.txt": "removed"}, "actions")
            expect(len(play.payloads("b.txt")), 0, "points of b.txt")
            return "points of the deleted file removed"

        def empty_folder() -> str:
            (play.folder / "a.txt").unlink()
            try:
                play.sync()
            except IndexerError as exc:
                expect(len(play.payloads("a.txt")), 1, "points of a.txt")
                return f"stopped, nothing deleted: {str(exc)[:60]}..."
            raise AssertionError("an empty folder was synced")

        scenarios = {
            "First run": first_run,
            "Second run": second_run,
            "File changed (3 chunks -> 1)": changed_file,
            "Only the catalog label changed": changed_label,
            "Dry run": dry_run,
            "File deleted from the folder": deleted_file,
            "Folder empty (wrong path?)": empty_folder,
        }
        results = [run_check(name, check) for name, check in scenarios.items()]
    client.delete_collection(TEST_COLLECTION)
    return results


# --- Part 2: the real collection ----------------------------------------------------

def all_points(client, with_vectors: bool = False) -> list:
    points, offset = [], None
    while True:
        page, offset = client.scroll(config.QDRANT_COLLECTION, limit=1000, offset=offset,
                                     with_payload=True, with_vectors=with_vectors)
        points += page
        if offset is None:
            return points


def check_up_to_date(client, catalog) -> str:
    results = sync_folder(config.SAMPLES_DIR, catalog, client, dry_run=True)
    pending = [f"{r.source}: {r.action}" for r in results if r.action not in ("unchanged", "skipped")]
    if pending:
        raise AssertionError(f"run scripts.index_documents first; pending: {pending}")
    return f"all {len(results)} files unchanged"


def check_points(client) -> str:
    points = all_points(client, with_vectors=True)
    pairs = Counter((p.payload["source"], p.payload["chunk_index"]) for p in points)
    duplicates = [pair for pair, count in pairs.items() if count > 1]
    if duplicates:
        raise AssertionError(f"duplicate chunks: {duplicates[:5]}")
    incomplete = [p.id for p in points if set(p.payload) != POINT_FIELDS]
    if incomplete:
        raise AssertionError(f"points with missing or extra fields: {incomplete[:5]}")
    wrong_size = [p.id for p in points if len(p.vector[DENSE_VECTOR]) != config.EMBEDDING_DIM]
    if wrong_size:
        raise AssertionError(f"vectors with the wrong size: {wrong_size[:5]}")
    return f"{len(points)} points from {len({s for s, _ in pairs})} files, no duplicates, all complete"


# --- Part 3: first searches ---------------------------------------------------------

def run_searches(client) -> bool:
    model = get_model_client()
    filter_ok = True
    for question, query_filter in SEARCHES:
        label = "status = active" if query_filter else "no filter"
        hits = client.query_points(config.QDRANT_COLLECTION, query=model.embed_query(question), using=DENSE_VECTOR,
                                   query_filter=query_filter, limit=3, with_payload=True).points
        print(f"\n  {question!r}  ({label})")
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload
            print(f"    {rank}. {hit.score:.3f}  {payload['source']} > {payload['heading_path']}  [{payload['status']}]")
        if query_filter is not None and any(hit.payload["status"] != "active" for hit in hits):
            filter_ok = False
    return filter_ok


def check_filter(filter_ok: bool) -> str:
    if not filter_ok:
        raise AssertionError("a superseded chunk came back despite the status filter")
    return "only active chunks returned"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="       (log) %(message)s")
    client = get_qdrant()
    catalog = load_catalog()

    print(f"Part 1: sync rules on the test collection '{TEST_COLLECTION}'")
    results = run_sync_scenarios(client)

    print(f"\nPart 2: the real collection '{config.QDRANT_COLLECTION}'")
    if not ensure_collection(client, create=False):
        print("[FAIL] Collection does not exist: run python -m scripts.index_documents first")
        sys.exit(1)
    results.append(run_check("Index is up to date", lambda: check_up_to_date(client, catalog)))
    results.append(run_check("Points are complete and unique", lambda: check_points(client)))

    print("\nPart 3: first searches (read the results; only the filter is checked)")
    filter_ok = run_searches(client)
    results.append(run_check("Filter keeps superseded chunks out", lambda: check_filter(filter_ok)))

    print(f"\nTOTAL: {sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()