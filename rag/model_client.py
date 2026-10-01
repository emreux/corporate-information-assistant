"""
Client for the CIA model server running on the A5000 machine.

Every embedding and reranking call in the project goes through this module, so connection
handling, batching, retries and model compatibility checks live in exactly one place.
"""
import logging
from functools import lru_cache

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

logger = logging.getLogger(__name__)


class ModelServerError(RuntimeError):
    """The model server returned an error or an unexpected response."""


class ModelServerUnavailable(ModelServerError):
    """The model server cannot be reached (VPN down, server stopped or wrong address)."""


class ModelServerBusy(ModelServerError):
    """The model server could not handle the request right now (e.g. GPU out of memory, slow response)."""


class ModelClient:
    """Thin, typed wrapper around the /health, /embed and /rerank endpoints."""

    def __init__(
        self,
        base_url: str = config.MODEL_SERVER_URL,
        api_key: str = config.MODEL_SERVER_API_KEY,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": api_key},
            timeout=httpx.Timeout(
                config.MODEL_SERVER_READ_TIMEOUT,
                connect=config.MODEL_SERVER_CONNECT_TIMEOUT,
            ),
        )
        self._verified = False

    def close(self) -> None:
        self._http.close()

    # --- Low-level HTTP -------------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        """Send a request and translate transport and HTTP errors into clear exceptions."""
        try:
            response = self._http.request(method, path, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ModelServerUnavailable(
                f"Cannot reach the model server at {self.base_url}. "
                "Is the VPN connected and is the server running on the A5000?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ModelServerBusy(f"The model server at {self.base_url} did not respond in time.") from exc

        if response.status_code == 401:
            raise ModelServerError(
                "The model server rejected the API key. "
                "MODEL_SERVER_API_KEY must be identical in both .env files."
            )
        if response.status_code == 503:
            raise ModelServerBusy(f"The model server is busy: {response.text}")
        if response.is_error:
            raise ModelServerError(f"Model server error {response.status_code} on {path}: {response.text}")
        return response.json()

    @retry(
        retry=retry_if_exception_type(ModelServerBusy),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _post(self, path: str, payload: dict) -> dict:
        """POST with retries for temporary failures only."""
        return self._request("POST", path, payload)

    # --- Health and compatibility ---------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def verify_compatibility(self) -> dict:
        """Fail fast if the server's models differ from the ones this project was indexed with."""
        info = self.health()
        expected = {
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dim": config.EMBEDDING_DIM,
            "reranker_model": config.RERANKER_MODEL,
        }
        mismatches = [
            f"{key}: server={info.get(key)!r}, config={value!r}"
            for key, value in expected.items()
            if info.get(key) != value
        ]
        if mismatches:
            raise ModelServerError("Model server does not match config.py -> " + "; ".join(mismatches))
        self._verified = True
        return info

    def _ensure_verified(self) -> None:
        if not self._verified:
            self.verify_compatibility()

    # --- Embedding and reranking ----------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks in batches. Order of the output matches the input."""
        if not texts:
            return []
        self._ensure_verified()
        vectors: list[list[float]] = []
        batch_size = config.MODEL_SERVER_BATCH_SIZE
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            result = self._post("/embed", {"texts": batch, "input_type": "document"})
            for index in result["truncated"]:
                logger.warning(
                    "Text #%d exceeded the embedding token limit and was truncated: %.80r",
                    start + index, batch[index],
                )
            vectors.extend(result["embeddings"])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question. Queries may be encoded differently from documents."""
        if not text.strip():
            raise ValueError("Query text is empty.")
        self._ensure_verified()
        result = self._post("/embed", {"texts": [text], "input_type": "query"})
        return result["embeddings"][0]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return a 0-1 relevance score for each document, in the same order as the input."""
        if not documents:
            return []
        self._ensure_verified()
        scores: list[float] = []
        batch_size = config.MODEL_SERVER_BATCH_SIZE
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            result = self._post("/rerank", {"query": query, "documents": batch})
            scores.extend(result["scores"])
        return scores


@lru_cache(maxsize=1)
def get_model_client() -> ModelClient:
    """Return one shared client so the whole app reuses a single connection pool."""
    return ModelClient()