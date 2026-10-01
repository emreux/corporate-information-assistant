"""
Retriever: finds the chunks that answer a question, among the ones the user may see.

Stage 1, the search (step 15): two searches run inside Qdrant, both with the same access filter:
  - dense: closest in meaning (the embedding), good with other wordings of the same thing;
  - sparse: BM25 on the words, good with codes, numbers, names and abbreviations.
Their rankings are fused with Reciprocal Rank Fusion: each chunk scores 1 / (k + rank) in each list.
Ranks, not scores, are combined, because the two scores are on different scales.

Stage 2, the reranker (step 16): both searches compare the question with a summary of each chunk
made in advance, so they find chunks that look like the question. The reranker (a cross-encoder
on our model server) reads the question and each candidate together and scores whether the chunk
is about what is asked, from 0 to 1. The candidates are sorted by that score, and the ones below
config.RERANK_MIN_SCORE are dropped: when none is left, the documents have nothing on the question.
It only ever sees candidates that passed the access filter.
"""
from dataclasses import dataclass, replace

from qdrant_client import models

import config
from rag.model_client import get_model_client
from rag.sparse import encode_query
from rag.store import DENSE_VECTOR, SPARSE_VECTOR, get_qdrant

RETRIEVAL_MODES = ("hybrid", "dense", "sparse")


class RetrievalError(RuntimeError):
    """The search in Qdrant failed (Qdrant not running, or the collection was never built)."""


@dataclass(frozen=True)
class Hit:
    score: float                   # reranker: relevance 0-1; without it: cosine, BM25 or RRF (only the order matters)
    text: str
    source: str
    heading_path: str
    page_start: int | None
    page_end: int | None
    effective_date: str | None
    status: str
    replaced_by: str | None
    ocr: bool
    access_level: str

    @classmethod
    def from_payload(cls, score: float, payload: dict) -> "Hit":
        """Build a hit from a Qdrant search result (the payload is the record of step 8)."""
        return cls(
            score=score,
            text=payload["text"],
            source=payload["source"],
            heading_path=payload["heading_path"],
            page_start=payload["page_start"],
            page_end=payload["page_end"],
            effective_date=payload["effective_date"],
            status=payload["status"],
            replaced_by=payload["replaced_by"],
            ocr=payload["ocr"],
            access_level=payload["access_level"],
        )

    @property
    def content(self) -> str:
        """The chunk text without its "[source > heading path]" line."""
        return self.text.partition("\n")[2]

    @property
    def pages(self) -> str:
        if self.page_start is None:
            return ""
        if self.page_start == self.page_end:
            return f"page {self.page_start}"
        return f"pages {self.page_start}-{self.page_end}"

    def label(self) -> str:
        """Short description for lists: file > heading path (pages)."""
        where = " > ".join(part for part in (self.source, self.heading_path) if part)
        return f"{where} ({self.pages})" if self.pages else where


def search(question: str, top_k: int = config.RETRIEVAL_TOP_K, include_superseded: bool = False,
           access_levels: tuple[str, ...] = config.DEFAULT_ACCESS_LEVELS, mode: str | None = None,
           rerank: bool | None = None) -> list[Hit]:
    """
    Return at most top_k chunks for the question; fewer, or none, when the reranker finds nothing relevant.
    Old versions are left out unless asked for.
    mode: hybrid, dense or sparse (default: config.RETRIEVAL_MODE); the other two are for comparisons.
    rerank: read the candidates again with the reranker (default: config.RERANK).
    """
    rerank = config.RERANK if rerank is None else rerank
    hits = find_candidates(question, max(top_k, config.RERANK_CANDIDATES) if rerank else top_k,
                           include_superseded, access_levels, mode)
    if not rerank:
        return hits
    return [hit for hit in rerank_hits(question, hits) if hit.score >= config.RERANK_MIN_SCORE][:top_k]


def rerank_hits(question: str, hits: list[Hit]) -> list[Hit]:
    """All hits with the reranker's score instead of the search score, the most relevant first."""
    if not hits:
        return []
    scores = get_model_client().rerank(question, [hit.text for hit in hits])   # text with its heading line
    scored = [replace(hit, score=score) for hit, score in zip(hits, scores, strict=True)]
    return sorted(scored, key=lambda hit: hit.score, reverse=True)


def find_candidates(question: str, limit: int, include_superseded: bool = False,
                    access_levels: tuple[str, ...] = config.DEFAULT_ACCESS_LEVELS,
                    mode: str | None = None) -> list[Hit]:
    """Stage 1: the search in Qdrant, without the reranker."""
    mode = mode or config.RETRIEVAL_MODE
    if mode not in RETRIEVAL_MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}. Known: {RETRIEVAL_MODES}")
    if not access_levels:
        return []
    conditions = [models.FieldCondition(key="access_level", match=models.MatchAny(any=list(access_levels)))]
    if not include_superseded:
        conditions.append(models.FieldCondition(key="status", match=models.MatchValue(value="active")))
    allowed = models.Filter(must=conditions)          # applied to every search below, without exception

    dense = get_model_client().embed_query(question) if mode != "sparse" else None
    sparse = encode_query(question) if mode != "dense" else None
    if sparse is not None and not sparse.indices:     # nothing to search for with keywords
        if mode == "sparse":
            return []
        sparse, mode = None, "dense"
    try:
        if mode == "hybrid":
            response = get_qdrant().query_points(
                config.QDRANT_COLLECTION,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE_VECTOR, filter=allowed, limit=config.RETRIEVAL_PREFETCH),
                    models.Prefetch(query=sparse, using=SPARSE_VECTOR, filter=allowed, limit=config.RETRIEVAL_PREFETCH),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit, with_payload=True,
            )
        else:
            response = get_qdrant().query_points(
                config.QDRANT_COLLECTION, query=dense if mode == "dense" else sparse,
                using=DENSE_VECTOR if mode == "dense" else SPARSE_VECTOR,
                query_filter=allowed, limit=limit, with_payload=True,
            )
    except Exception as exc:
        raise RetrievalError(
            f"Search in Qdrant failed ({type(exc).__name__}: {exc}). "
            "Is Qdrant running, and has 'python -m scripts.index_documents' been run?"
        ) from exc
    return [Hit.from_payload(point.score, point.payload) for point in response.points]
