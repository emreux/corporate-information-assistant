"""
Keyword search (BM25) for Turkish text, as sparse vectors that Qdrant can store and search.

A sparse vector lists only the words a text contains: {word id: weight}. BM25 scores a chunk by the
query words it contains, weighting each word by:
  - how often it appears in the chunk, with diminishing returns (k1),
  - how short the chunk is: words in long chunks count a little less (b, avg length),
  - how rare it is in the whole collection (IDF): Qdrant computes this part itself, at search
    time, because the collection is configured with Modifier.IDF. So new documents update it.

The tokenizer is written for Turkish and for company codes:
  - lowercase without Turkish letters (ç -> c, ı -> i, ş -> s ...): people often type "musteri" for
    "müşteri"; the suffix after an apostrophe is dropped: Kaya'nın -> kaya
  - numbers keep their separators: 180.000 -> 180000 (otherwise "000" would match everywhere)
  - codes are kept whole and in parts: VPN-ERR-204 -> vpnerr204, vpn, err, 204, so that VPN-ERR204
    and "vpn err 204" match as well
  - words are cut to their first letters (config.BM25_PREFIX_LETTERS): yöneticiler, yöneticilere ->
    yonet. A simple stand-in for Turkish stemming that works well in practice; it also keeps
    kaya and kayaalp (-> kayaa) apart.
"""
import hashlib
import re
from collections import Counter

from qdrant_client import models

import config

ENCODER_VERSION = 1          # raise when tokenize() or the weights change: stored vectors must be rebuilt

APOSTROPHE_SUFFIX = re.compile(r"['’]\w+")
ASCII_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")
CODE = re.compile(r"[^\W_]+(?:[-_/][^\W_]+)+")          # parts joined by - _ or /
TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+")         # a number (with separators) or a run of letters


def signature() -> str:
    """Everything that changes the stored sparse vectors; part of the indexer's pipeline signature."""
    return (f"bm25-v{ENCODER_VERSION}-k1={config.BM25_K1}-b={config.BM25_B}"
            f"-avg={config.BM25_AVG_DOC_TOKENS}-prefix={config.BM25_PREFIX_LETTERS}")


def tokenize(text: str) -> list[str]:
    """The search words of a text, with repeats (repeats count in BM25)."""
    text = text.replace("İ", "i").lower().translate(ASCII_FOLD)     # "İ".lower() would add a combining dot
    text = APOSTROPHE_SUFFIX.sub("", text)
    tokens = [re.sub(r"[-_/]", "", code) for code in CODE.findall(text) if any(c.isdigit() for c in code)]
    for token in TOKEN.findall(text):
        if token[0].isdigit():
            tokens.append(re.sub(r"[.,]", "", token))
        elif len(token) > 1:                                 # single letters carry no meaning
            tokens.append(token[:config.BM25_PREFIX_LETTERS])
    return tokens


def encode_document(text: str) -> models.SparseVector:
    """BM25 weights of a chunk, without the IDF part (Qdrant adds it when searching)."""
    counts = Counter(tokenize(text))
    length = sum(counts.values())
    k1, b = config.BM25_K1, config.BM25_B
    norm = k1 * (1 - b + b * length / config.BM25_AVG_DOC_TOKENS)
    weights = {_index(token): count * (k1 + 1) / (count + norm) for token, count in counts.items()}
    return _vector(weights)


def encode_query(text: str) -> models.SparseVector:
    """Each query word counts once; its rarity (IDF) decides how much it matters."""
    return _vector({_index(token): 1.0 for token in set(tokenize(text))})


def _index(token: str) -> int:
    """A stable number for a word (Qdrant stores word ids, not words). Collisions are very rare."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "big")


def _vector(weights: dict[int, float]) -> models.SparseVector:
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[round(weights[index], 4) for index in indices])
