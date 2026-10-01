"""
CIA model server: serves the embedding and reranking models on the GPU.

Run on the A5000 machine, inside the virtual environment:
    uvicorn server:app --host 0.0.0.0 --port 8001
"""
import inspect
import logging
import math
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Callable, Literal, TypeVar

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
API_KEY = os.getenv("MODEL_SERVER_API_KEY", "")
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "1024"))
RERANK_MAX_TOKENS = int(os.getenv("RERANK_MAX_TOKENS", "1024"))
GPU_BATCH_SIZE = int(os.getenv("GPU_BATCH_SIZE", "32"))
MAX_ITEMS_PER_REQUEST = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cia.model_server")

state: dict = {}
gpu_lock = threading.Lock()
T = TypeVar("T")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models once at startup and keep them in GPU memory."""
    if not API_KEY:
        raise RuntimeError("MODEL_SERVER_API_KEY is not set. Add it to the .env file next to server.py.")
    if DEVICE == "cpu":
        logger.warning("CUDA is not available; running on CPU. Expect very slow responses.")

    started = time.perf_counter()
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
    embedder.max_seq_length = EMBED_MAX_TOKENS
    reranker = CrossEncoder(RERANKER_MODEL, device=DEVICE, max_length=RERANK_MAX_TOKENS)
    if DEVICE == "cuda":
        to_half_precision(embedder)
        to_half_precision(reranker)

    state["embedder"] = embedder
    state["reranker"] = reranker
    state["embedding_dim"] = embedder.get_sentence_embedding_dimension()
    state["query_prompt"] = "query" if "query" in (embedder.prompts or {}) else None
    logger.info("Models loaded in %.1f s on %s", time.perf_counter() - started, DEVICE)

    yield

    state.clear()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


app = FastAPI(title="CIA Model Server", lifespan=lifespan)


def to_half_precision(model) -> None:
    """Switch a model to float16 to halve VRAM usage and speed up inference."""
    module = model if isinstance(model, torch.nn.Module) else getattr(model, "model", None)
    if module is not None:
        module.half()


def raw_reranker_logits(reranker: CrossEncoder, pairs: list[tuple[str, str]]) -> list[float]:
    """Return raw logits regardless of the sentence-transformers version's default activation."""
    params = inspect.signature(reranker.predict).parameters
    kwargs = {name: torch.nn.Identity() for name in ("activation_fn", "activation_fct") if name in params}
    logits = reranker.predict(pairs, batch_size=GPU_BATCH_SIZE, convert_to_numpy=True, **kwargs)
    return [float(value) for value in logits]


def sigmoid(value: float) -> float:
    """Numerically stable logistic function: maps a logit to a 0-1 relevance score."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Reject requests that do not carry the shared API key in the X-API-Key header."""
    if not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def run_on_gpu(work: Callable[[], T]) -> T:
    """Serialize GPU work so concurrent requests queue up instead of exhausting VRAM."""
    with gpu_lock:
        try:
            return work()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise HTTPException(status_code=503, detail="GPU out of memory. Retry with a smaller batch.")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_ITEMS_PER_REQUEST)
    input_type: Literal["query", "document"] = "document"


class EmbedResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[list[float]]
    truncated: list[int]
    elapsed_ms: float


class RerankRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[str] = Field(min_length=1, max_length=MAX_ITEMS_PER_REQUEST)


class RerankResponse(BaseModel):
    model: str
    scores: list[float]
    elapsed_ms: float


@app.get("/health")
def health() -> dict:
    info = {
        "status": "ok",
        "device": DEVICE,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": state.get("embedding_dim"),
        "embed_max_tokens": EMBED_MAX_TOKENS,
        "query_prompt": state.get("query_prompt"),
        "reranker_model": RERANKER_MODEL,
    }
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info()
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_free_gb"] = round(free / 1024**3, 1)
        info["gpu_memory_total_gb"] = round(total / 1024**3, 1)
    return info


@app.post("/embed", response_model=EmbedResponse, dependencies=[Depends(require_api_key)])
def embed(request: EmbedRequest) -> EmbedResponse:
    embedder: SentenceTransformer = state["embedder"]
    started = time.perf_counter()

    token_counts = [len(ids) for ids in embedder.tokenizer(request.texts)["input_ids"]]
    truncated = [i for i, count in enumerate(token_counts) if count > EMBED_MAX_TOKENS]
    if truncated:
        logger.warning("%d text(s) exceed %d tokens and were truncated.", len(truncated), EMBED_MAX_TOKENS)

    prompt_name = state["query_prompt"] if request.input_type == "query" else None
    vectors = run_on_gpu(lambda: embedder.encode(
        request.texts,
        batch_size=GPU_BATCH_SIZE,
        normalize_embeddings=True,
        prompt_name=prompt_name,
        convert_to_numpy=True,
    ))
    return EmbedResponse(
        model=EMBEDDING_MODEL,
        dim=int(vectors.shape[1]),
        embeddings=vectors.astype("float32").tolist(),
        truncated=truncated,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )


@app.post("/rerank", response_model=RerankResponse, dependencies=[Depends(require_api_key)])
def rerank(request: RerankRequest) -> RerankResponse:
    reranker: CrossEncoder = state["reranker"]
    started = time.perf_counter()
    pairs = [(request.query, document) for document in request.documents]
    logits = run_on_gpu(lambda: raw_reranker_logits(reranker, pairs))
    return RerankResponse(
        model=RERANKER_MODEL,
        scores=[sigmoid(logit) for logit in logits],
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )