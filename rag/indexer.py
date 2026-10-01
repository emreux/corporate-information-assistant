"""
Indexer: keeps the Qdrant collection in step with the documents folder.

Rules:
  1. One collection; every point has two named vectors, "dense" (meaning) and "bm25" (keywords,
     step 15), and a payload (the record).
  2. The collection must match config.py; otherwise stop and ask for a rebuild (--recreate).
  3. Point ids come from the content, so writing the same thing twice never creates duplicates.
  4. Prepare everything in memory first, write last, delete the old version only after that.
  5. A file whose fingerprint has not changed is skipped: no reading, no embedding, no writing.
  6. Points of files that are no longer in the folder are deleted.
  7. If the folder has no readable file at all, nothing is deleted.
"""
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient, models

import config
from rag.chunker import chunk_document
from rag.metadata import UNCATALOGED, CatalogEntry, attach_metadata
from rag.model_client import ModelServerUnavailable, get_model_client
from rag.reader import (UnsupportedFormatError, cache_path, check_supported, file_sha256,
                        read_document_cached)
from rag.sparse import encode_document
from rag.sparse import signature as sparse_signature
from rag.store import DENSE_VECTOR, SPARSE_VECTOR, get_qdrant   # get_qdrant is also imported by the scripts

logger = logging.getLogger(__name__)

FILTER_FIELDS = ("source", "access_level", "status")   # fields that get a payload index for fast filtering
IGNORED_PREFIXES = ("~$", ".")                          # Office lock files ("~$report.docx") and hidden files
IGNORED_NAMES = {"desktop.ini", "thumbs.db"}


class IndexerError(RuntimeError):
    """Qdrant cannot be used safely: unreachable, incompatible collection or suspicious folder."""


@dataclass
class FileResult:
    source: str
    action: str             # indexed, unchanged, removed, skipped, failed, would index, would remove
    chunks: int = 0
    read_from: str = ""     # "cache" or "docling"
    seconds: float = 0.0
    detail: str = ""


# --- Rules 1 and 2: the collection --------------------------------------------------

def ensure_collection(client: QdrantClient, name: str = config.QDRANT_COLLECTION,
                      recreate: bool = False, create: bool = True) -> bool:
    """Create the collection if needed and check that it matches config.py. Returns whether it exists."""
    try:
        exists = client.collection_exists(name)
    except Exception as exc:
        raise IndexerError(f"Cannot reach Qdrant at {config.QDRANT_URL}. Is the Docker container running?") from exc

    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        if not create:
            return False
        client.create_collection(
            name,
            vectors_config={DENSE_VECTOR: models.VectorParams(size=config.EMBEDDING_DIM,
                                                              distance=models.Distance.COSINE)},
            # IDF is computed by Qdrant at search time, so it follows the collection as it changes
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field_name in FILTER_FIELDS:
            client.create_payload_index(name, field_name=field_name, field_schema=models.PayloadSchemaType.KEYWORD)
        logger.info("Created collection '%s'.", name)
        return True

    collection_params = client.get_collection(name).config.params
    vectors = collection_params.vectors
    params = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
    sparse = (collection_params.sparse_vectors or {}).get(SPARSE_VECTOR)
    if (params is None or params.size != config.EMBEDDING_DIM or params.distance != models.Distance.COSINE
            or sparse is None or sparse.modifier != models.Modifier.IDF):
        raise IndexerError(
            f"Collection '{name}' does not match config.py (expected a '{DENSE_VECTOR}' vector of size "
            f"{config.EMBEDDING_DIM} with cosine distance and a '{SPARSE_VECTOR}' sparse vector with IDF). "
            "Rebuild it: python -m scripts.index_documents --recreate"
        )
    return True


# --- Rules 3 and 5: fingerprint and point ids ---------------------------------------

def pipeline_signature() -> str:
    """Everything besides the file and its labels that changes what ends up in Qdrant."""
    return (f"{config.EMBEDDING_MODEL}|{config.CHUNK_SIZE_TOKENS}|{config.CHUNK_OVERLAP_TOKENS}"
            f"|v{config.PIPELINE_VERSION}|{sparse_signature()}")


def fingerprint(file_hash: str, entry: CatalogEntry) -> str:
    """Changes when the file, the pipeline settings or the catalog labels of the file change."""
    labels = f"{entry.access_level}|{entry.status}|{entry.effective_date}|{entry.replaced_by}"
    return hashlib.sha256(f"{file_hash}|{pipeline_signature()}|{labels}".encode("utf-8")).hexdigest()


def point_id(file_fingerprint: str, chunk_index: int) -> str:
    """The same fingerprint and chunk always give the same id, so a rewrite overwrites."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"cia/{file_fingerprint}/{chunk_index}"))


def stored_fingerprints(client: QdrantClient, name: str = config.QDRANT_COLLECTION) -> dict[str, set[str]]:
    """The fingerprints in the collection per source file (one pass over the payloads)."""
    result: dict[str, set[str]] = {}
    offset = None
    while True:
        points, offset = client.scroll(name, limit=1000, offset=offset,
                                       with_payload=["source", "fingerprint"], with_vectors=False)
        for point in points:
            result.setdefault(point.payload["source"], set()).add(point.payload["fingerprint"])
        if offset is None:
            return result


def _source_filter(source: str, keep_fingerprint: str | None = None) -> models.Filter:
    must_not = []
    if keep_fingerprint:
        must_not.append(models.FieldCondition(key="fingerprint", match=models.MatchValue(value=keep_fingerprint)))
    return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))],
                         must_not=must_not)


def delete_points(client: QdrantClient, source: str, keep_fingerprint: str | None = None,
                  name: str = config.QDRANT_COLLECTION) -> None:
    """Delete the points of a file, except those with keep_fingerprint (its new version)."""
    client.delete(name, points_selector=models.FilterSelector(filter=_source_filter(source, keep_fingerprint)),
                  wait=True)


# --- Rule 4: indexing one file ------------------------------------------------------

def index_file(path: Path, catalog: dict[str, CatalogEntry], client: QdrantClient,
               name: str = config.QDRANT_COLLECTION) -> FileResult:
    """Read, chunk, label and embed one file in memory, write it, then delete its previous version."""
    started = time.perf_counter()
    path = Path(path)
    new_fingerprint = fingerprint(file_sha256(path), catalog.get(path.name, UNCATALOGED))
    read_from = "cache" if cache_path(path).exists() else "docling"

    document = read_document_cached(path)
    records = attach_metadata(document, chunk_document(document), catalog)
    vectors = get_model_client().embed_documents([record["text"] for record in records])
    indexed_at = datetime.now().isoformat(timespec="seconds")
    points = [
        models.PointStruct(
            id=point_id(new_fingerprint, record["chunk_index"]),
            vector={DENSE_VECTOR: vector, SPARSE_VECTOR: encode_document(record["text"])},
            payload={**record, "fingerprint": new_fingerprint, "indexed_at": indexed_at},
        )
        for record, vector in zip(records, vectors, strict=True)
    ]
    # Nothing has touched Qdrant up to here: a failure above leaves the old version in place.
    for start in range(0, len(points), config.QDRANT_BATCH_SIZE):
        client.upsert(name, points=points[start:start + config.QDRANT_BATCH_SIZE], wait=True)
    delete_points(client, path.name, keep_fingerprint=new_fingerprint, name=name)

    return FileResult(path.name, "indexed", chunks=len(points), read_from=read_from,
                      seconds=time.perf_counter() - started, detail="; ".join(document.warnings))


# --- Rules 5, 6 and 7: keeping the whole folder in sync -----------------------------

def sync_folder(folder: Path, catalog: dict[str, CatalogEntry], client: QdrantClient,
                name: str = config.QDRANT_COLLECTION, dry_run: bool = False) -> list[FileResult]:
    """Make the collection match the folder: index new and changed files, skip unchanged ones,
    remove the points of deleted ones. With dry_run nothing is written or deleted."""
    folder = Path(folder)
    candidates = sorted(path for path in folder.iterdir() if path.is_file()) if folder.is_dir() else []
    candidates = [path for path in candidates
                  if not path.name.startswith(IGNORED_PREFIXES) and path.name.lower() not in IGNORED_NAMES]

    results: list[FileResult] = []
    documents: list[Path] = []
    for path in candidates:
        try:
            check_supported(path)
            documents.append(path)
        except UnsupportedFormatError as exc:
            results.append(FileResult(path.name, "skipped", detail=str(exc)))
    if not documents:                                                            # rule 7
        raise IndexerError(f"No readable documents in '{folder}'. Nothing was indexed or deleted: "
                           "is the folder path right and is the drive connected?")

    stored = stored_fingerprints(client, name) if client.collection_exists(name) else {}
    for path in documents:
        entry = catalog.get(path.name)
        note = "" if entry else f"not in the catalog, treated as '{config.DEFAULT_ACCESS_LEVEL}'"
        if stored.get(path.name) == {fingerprint(file_sha256(path), entry or UNCATALOGED)}:
            results.append(FileResult(path.name, "unchanged", detail=note))              # rule 5
            continue
        if dry_run:
            results.append(FileResult(path.name, "would index", detail=note))
            continue
        try:
            result = index_file(path, catalog, client, name)
        except ModelServerUnavailable:
            raise                               # the A5000 is unreachable: every other file would fail too
        except Exception as exc:
            result = FileResult(path.name, "failed", detail=f"{type(exc).__name__}: {exc}")
        if note:
            result.detail = "; ".join(part for part in (note, result.detail) if part)
        results.append(result)

    for source in sorted(set(stored) - {path.name for path in documents}):       # rule 6
        if not dry_run:
            delete_points(client, source, name=name)
        results.append(FileResult(source, "would remove" if dry_run else "removed"))
    return results
