"""
Environment check: tries to reach each component and reports the result.

Usage (from the project root):
    python -m scripts.check_env
    python -m scripts.check_env --skip-gemini
    python -m scripts.check_env --skip-gemini --skip-a5000
"""
import argparse
import importlib
import sys
from typing import Callable

import config

MIN_PYTHON = (3, 11)

REQUIRED_PACKAGES = [
    "streamlit",
    "qdrant_client",
    "chromadb",
    "openai",
    "dotenv",
    "docling",
    "pandas",
    "openpyxl",
    "langchain_text_splitters",
    "snowballstemmer",
    "tenacity",
    "httpx",
    "bs4",
    "easyocr",
    "transformers"
]

# Sanity data for the reranker: the negated sentence must win.
RERANK_QUERY = "Deneme süresindeyim, uzaktan çalışabilir miyim?"
RERANK_DOCUMENTS = [
    "Çalışanlar haftada en fazla 2 gün uzaktan çalışabilir.",
    "Deneme süresindeki çalışanlar uzaktan çalışamaz.",
]
RERANK_EXPECTED_WINNER = 1


def run_check(name: str, check: Callable[[], str]) -> bool:
    """Run a single check and print a one-line result."""
    try:
        result = check()
        print(f"[OK]   {name}: {result}")
        return True
    except Exception as exc:
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
        return False


def check_python() -> str:
    version = sys.version_info
    if version[:2] < MIN_PYTHON:
        raise RuntimeError(
            f"Python {version.major}.{version.minor} detected; "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer is required."
        )
    return f"{version.major}.{version.minor}.{version.micro}"


def check_packages() -> str:
    """Import every required package to catch missing or incompatible installs early."""
    broken = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except Exception as exc:
            broken.append(f"{package} ({type(exc).__name__}: {exc})")
    if broken:
        raise RuntimeError("Cannot import: " + "; ".join(broken))
    return f"all {len(REQUIRED_PACKAGES)} packages imported"


def check_qdrant() -> str:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=config.QDRANT_URL)
    names = [collection.name for collection in client.get_collections().collections]
    return f"connected, collections: {names}"


def check_chroma() -> str:
    import chromadb

    client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)
    client.heartbeat()
    return f"connected (client version {chromadb.__version__})"


def check_gemini() -> str:
    """Send a tiny request to Gemini. Consumes one request from the daily quota."""
    from openai import OpenAI

    client = OpenAI(api_key=config.GEMINI_API_KEY, base_url=config.GEMINI_BASE_URL)
    response = client.chat.completions.create(
        model=config.GEMINI_MODEL,
        messages=[{"role": "user", "content": "Reply with only the word 'ready'."}],
    )
    reply = (response.choices[0].message.content or "").strip()
    return f"{config.GEMINI_MODEL} replied: {reply!r}"


def check_model_server() -> str:
    """Reach the model server, verify its models match config.py and run a reranker sanity test."""
    from rag.model_client import ModelClient

    client = ModelClient()
    try:
        info = client.verify_compatibility()
        vector = client.embed_query("test")
        if len(vector) != config.EMBEDDING_DIM:
            raise RuntimeError(f"Embedding has {len(vector)} dimensions, expected {config.EMBEDDING_DIM}.")
        scores = client.rerank(RERANK_QUERY, RERANK_DOCUMENTS)
        winner = max(range(len(scores)), key=scores.__getitem__)
        if winner != RERANK_EXPECTED_WINNER:
            raise RuntimeError(f"Reranker sanity test failed, scores: {scores}")
    finally:
        client.close()
    gpu = f"{info.get('gpu', info.get('device'))}, {info.get('gpu_memory_free_gb', '?')} GB free"
    return f"{gpu}; models match config; reranker scores {[round(s, 3) for s in scores]}"


def check_ollama() -> str:
    """List the models Ollama serves and confirm the configured one is available."""
    from openai import OpenAI

    client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama", timeout=5, max_retries=0)
    names = [model.id for model in client.models.list()]
    if config.OLLAMA_MODEL not in names:
        raise RuntimeError(f"'{config.OLLAMA_MODEL}' is not pulled on the server. Available: {names}")
    return f"reachable, '{config.OLLAMA_MODEL}' available"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that all CIA components are reachable.")
    parser.add_argument("--skip-gemini", action="store_true", help="Do not spend Gemini quota.")
    parser.add_argument("--skip-a5000", action="store_true", help="Skip the model server and Ollama.")
    args = parser.parse_args()

    checks: dict[str, Callable[[], str]] = {
        "Python": check_python,
        "Packages": check_packages,
        "Qdrant": check_qdrant,
        "Chroma": check_chroma,
    }
    if not args.skip_gemini:
        checks["Gemini"] = check_gemini
    if not args.skip_a5000:
        checks["Model server"] = check_model_server
        checks["Ollama"] = check_ollama

    results = [run_check(name, check) for name, check in checks.items()]

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()