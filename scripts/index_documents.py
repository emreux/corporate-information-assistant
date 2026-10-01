"""
Step 9: bring the Qdrant collection in line with the documents folder.

New and changed files are indexed, unchanged files are skipped and the points of files that
left the folder are deleted.

Usage (from the project root):
    python -m scripts.index_documents                      # sync data/samples
    python -m scripts.index_documents --dry-run            # only show what would change
    python -m scripts.index_documents --recreate           # delete and rebuild the whole collection
    python -m scripts.index_documents --folder data/raw    # sync another folder
"""
import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import config
from rag.indexer import IndexerError, ensure_collection, get_qdrant, sync_folder
from rag.metadata import CatalogError, load_catalog, missing_files
from rag.model_client import ModelServerUnavailable


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync the documents folder with the Qdrant collection.")
    parser.add_argument("--folder", type=Path, default=config.SAMPLES_DIR, help="Folder with the documents.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change; change nothing.")
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild the whole collection.")
    args = parser.parse_args()
    if args.dry_run and args.recreate:
        parser.error("--dry-run and --recreate cannot be used together.")
    logging.basicConfig(level=logging.WARNING, format="       (log) %(message)s")

    try:
        catalog = load_catalog()
        client = get_qdrant()
        ensure_collection(client, recreate=args.recreate, create=not args.dry_run)
        if args.recreate:
            print(f"Collection '{config.QDRANT_COLLECTION}' was deleted and created again.")
        print(f"Syncing {args.folder} -> collection '{config.QDRANT_COLLECTION}'"
              f"{'  (DRY RUN: nothing will be changed)' if args.dry_run else ''}\n")
        results = sync_folder(args.folder, catalog, client, dry_run=args.dry_run)
    except (CatalogError, IndexerError, ModelServerUnavailable) as exc:
        print(f"STOPPED: {exc}")
        sys.exit(1)

    print(f"{'File':50s} {'Chunks':>6s} {'Read':>8s} {'Sec':>6s}  Result")
    for result in results:
        chunks = str(result.chunks) if result.chunks else "-"
        seconds = f"{result.seconds:.1f}" if result.seconds else "-"
        line = f"{result.source:50s} {chunks:>6s} {result.read_from or '-':>8s} {seconds:>6s}  {result.action}"
        print(line + (f"  ({result.detail})" if result.detail else ""))

    counts = Counter(result.action for result in results)
    print("\n" + ", ".join(f"{count} {action}" for action, count in sorted(counts.items())))
    stale = missing_files(catalog, args.folder)
    if stale:
        print(f"Catalog entries without a file in {args.folder}: {stale}")
    if client.collection_exists(config.QDRANT_COLLECTION):
        total = client.count(config.QDRANT_COLLECTION, exact=True).count
        print(f"Collection '{config.QDRANT_COLLECTION}': {total} points")
    sys.exit(1 if counts["failed"] else 0)


if __name__ == "__main__":
    main()