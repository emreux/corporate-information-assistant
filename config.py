"""
Central configuration for CIA (Corporate Information Assistant).

Every setting lives here. No other module should hardcode numbers, URLs or model names.
Secret and machine-specific values are read from .env; design decisions are defined here.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _required(name: str) -> str:
    """Return an environment variable or fail fast with a clear message."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required setting '{name}'. Check your .env file (see .env.example)."
        )
    return value


# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "data" / "raw"
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EVAL_DIR = ROOT_DIR / "evaluation"

for directory in (RAW_DIR, SAMPLES_DIR, PROCESSED_DIR, EVAL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# --- LLM: Gemini (OpenAI-compatible endpoint) ---
GEMINI_API_KEY = _required("GEMINI_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# --- A5000 machine: model server (embedding + reranking) ---
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001")
MODEL_SERVER_API_KEY = _required("MODEL_SERVER_API_KEY")
MODEL_SERVER_CONNECT_TIMEOUT = 5.0    # seconds; fail fast when the VPN is down
MODEL_SERVER_READ_TIMEOUT = 120.0     # seconds; large batches and cold starts can be slow
MODEL_SERVER_BATCH_SIZE = 64          # texts per request; the server accepts at most 256

# --- A5000 machine: local LLM via Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")

# --- LLM routing (step 4) ---
# Each task lists providers in order of preference. The next provider is tried only when the
# previous one is unreachable or temporarily failing (see rag/llm.py).
LLM_TASKS = {
    "answer": {"providers": ["gemini", "qwen"], "temperature": 0.2},
    "judge": {"providers": ["gemini"], "temperature": 0.0},     # step 12: one fixed judge, no fallback
    "rewrite": {"providers": ["gemini", "qwen"], "temperature": 0.0},   # step 17: follow-up -> standalone
}
LLM_CONNECT_TIMEOUT = 5.0     # seconds; fail fast when the internet or the VPN is down
LLM_READ_TIMEOUT = 90.0       # seconds; long answers from the local model can be slow

# --- Embedding and reranking models ---
# WARNING: These must match the models loaded on the model server (checked at startup).
# Changing the embedding model requires recreating the vector collection and
# re-embedding every document (see step 9).
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# --- Vector databases ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "cia_documents"
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# --- Document reading (step 5) ---
OCR_LANGUAGES = ["tr", "en"]              # EasyOCR language codes for scanned documents
MIN_EXPECTED_CHARS_PER_DOCUMENT = 50      # below this, warn: probably scanned, empty or protected
REPEATED_LINE_MIN_SHARE = 0.5             # a short line on at least half of the pages is a header/footer
REPEATED_LINE_MAX_LENGTH = 150            # longer lines are content, never treated as headers/footers

# --- Tabular files: Excel and CSV (step 6) ---
# Numbers and dates are written the way a Turkish Excel shows them, so that "85.000" in a
# question matches "85.000" in the text.
THOUSANDS_SEPARATOR = "."
DECIMAL_SEPARATOR = ","
DATE_FORMAT = "%d.%m.%Y"
DATETIME_FORMAT = "%d.%m.%Y %H:%M"
CSV_ENCODINGS = ["utf-8-sig", "cp1254"]   # tried in order; cp1254 is Turkish Windows (old Excel exports)
CSV_DELIMITERS = ";,\t|"                  # candidates for delimiter detection
CSV_SNIFF_LINES = 50                      # lines used to detect the delimiter

# --- Chunking (step 7, tuned in step 20) ---
# Token counts use the embedding model's own tokenizer (EMBEDDING_MODEL).
CHUNK_SIZE_TOKENS = 450       # soft target per chunk, including the "[source > heading path]" line
CHUNK_OVERLAP_TOKENS = 50     # only used when a single paragraph is too long and must be split
CHUNK_MAX_TOKENS = 1000       # hard limit: the model server truncates texts above 1024 tokens

# --- Document metadata (step 8) ---
DOCUMENT_CATALOG = ROOT_DIR / "document_catalog.toml"
ACCESS_LEVELS = ("internal", "restricted")    # internal: every employee; restricted: management and HR
DOCUMENT_STATUSES = ("active", "superseded")
DEFAULT_ACCESS_LEVEL = "restricted"           # for files missing from the catalog: deny by default
OCR_TEXT_LAYER_MIN_CHARS = 20                 # a PDF page with less text than this was read with OCR

# --- Indexing (step 9) ---
PIPELINE_VERSION = 1          # raise by one after changing reader, chunker or metadata code (their output)
CACHE_DIR = PROCESSED_DIR / "cache"   # Docling results with the full text: protect it like the documents
QDRANT_TIMEOUT = 10           # seconds
QDRANT_BATCH_SIZE = 256       # points per upsert request

# --- Answering (step 10) ---
RETRIEVAL_TOP_K = 5                    # chunks shown to the LLM for each question
DEFAULT_ACCESS_LEVELS = ("internal",)  # when the caller does not say who is asking: the least access

# --- Hybrid search (step 15) ---
RETRIEVAL_MODE = "hybrid"     # hybrid (meaning + keywords), dense (meaning only) or sparse (keywords only)
RETRIEVAL_PREFETCH = 20       # candidates taken from each search before they are fused (RRF)
BM25_K1 = 1.2                 # how quickly repeats of a word stop adding to the score
BM25_B = 0.75                 # how much longer chunks are penalised (0: not at all, 1: fully)
BM25_AVG_DOC_TOKENS = 60      # typical chunk length in search words (ours: about 61); an estimate is enough
BM25_PREFIX_LETTERS = 5       # words are cut to this many letters: simple Turkish stemming

# --- Reranking (step 16) ---
# The search above finds candidates quickly; the reranker (a cross-encoder on the model server) then
# reads the question and each candidate together and scores how relevant it is, from 0 to 1.
RERANK = True                 # False: keep the order of the search (for comparisons: --no-rerank)
RERANK_CANDIDATES = 20        # candidates the reranker reads; the best RETRIEVAL_TOP_K of them are kept
RERANK_MIN_SCORE = 0.02       # candidates below this are dropped; if none is left, the LLM is not asked.
                              # Check it after changing documents or models: python -m scripts.check_rerank

# --- Conversations (step 17) ---
HISTORY_QUESTIONS = 3         # earlier questions used to turn a follow-up question into a standalone one

# --- Web interface (step 11) ---
UI_TEXTS = ROOT_DIR / "ui_texts.toml"   # every text shown on screen; the code only uses the keys
SOURCE_EXCERPT_CHARS = 300              # length of the passage shown under each cited source (step 13)

# --- Evaluation (step 12) ---
EVAL_QUESTIONS = EVAL_DIR / "questions.toml"
EVAL_RESULTS_DIR = EVAL_DIR / "results"   # answers from every document: protect them like the documents
EVAL_SECONDS_BETWEEN_LLM_CALLS = 4.0      # stay under the free tier's requests-per-minute limit
EVAL_RETRY_WAITS = (15, 30, 60)           # seconds to wait before each retry when no model answered

# --- Users and roles (step 14) ---
USERS_FILE = ROOT_DIR / "data" / "users.toml"   # password hashes: data/ is kept out of git
MIN_PASSWORD_LENGTH = 10
LOGIN_FAILURE_DELAY = 1.0                       # seconds after a wrong password: slows down guessing
# What each role may see, and whether it may manage documents and use the developer tools.
# Roles are policy: change them here, with a review, not in the user file.
ROLES = {
    "intern":   {"access_levels": ("internal",), "admin": False},
    "employee": {"access_levels": ("internal",), "admin": False},
    "manager":  {"access_levels": ("internal", "restricted"), "admin": False},
    "admin":    {"access_levels": ("internal", "restricted"), "admin": True},
}
LOCAL_ONLY_ACCESS_LEVELS = ("restricted",)   # chunks with these labels may only reach a model we run ourselves

# --- Question log (step 19) ---
# One JSON line per question asked in the web interface: who asked what, which sources were found and
# what went wrong. It holds questions and names, not answers: still personal data, so it stays in data/
# (outside git), and old months should be deleted when they are no longer needed.
QUERY_LOG_DIR = ROOT_DIR / "data" / "logs"
