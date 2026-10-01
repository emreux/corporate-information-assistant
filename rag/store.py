"""
Connection to the vector store (Qdrant), shared by the indexer (writing) and the retriever (reading).

Kept separate so that answering a question does not import the document reader (Docling),
which takes several seconds to load.
"""
from functools import lru_cache

from qdrant_client import QdrantClient

import config

DENSE_VECTOR = "dense"          # meaning: the embedding of the chunk (step 9)
SPARSE_VECTOR = "bm25"          # keywords: BM25 weights of the chunk (step 15)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL, timeout=config.QDRANT_TIMEOUT)
