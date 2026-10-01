"""
Smoke test for the CIA model server.

Usage:
    python smoke_test.py                       # tests http://localhost:8001
    python smoke_test.py http://10.0.0.5:8001  # tests a remote server
"""
import os
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
HEADERS = {"X-API-Key": os.environ.get("MODEL_SERVER_API_KEY", "")}


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def main() -> None:
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=120) as client:
        print("== health")
        print(client.get("/health").json())

        print("\n== embedding: synonym test (password / parola)")
        query = "Şifremi unuttum, ne yapmalıyım?"
        documents = [
            "Parolanızı sıfırlamak için Parolamı Unuttum bağlantısını kullanın.",
            "Yemek kartı her ayın ilk iş günü yüklenir.",
        ]
        query_vec = client.post("/embed", json={"texts": [query], "input_type": "query"}).raise_for_status().json()
        doc_resp = client.post("/embed", json={"texts": documents, "input_type": "document"}).raise_for_status().json()
        print(f"dim={doc_resp['dim']}")
        for text, vec in zip(documents, doc_resp["embeddings"]):
            print(f"  {dot(query_vec['embeddings'][0], vec):.3f}  {text}")

        print("\n== rerank: negation test")
        query = "Deneme süresindeyim, uzaktan çalışabilir miyim?"
        documents = [
            "Çalışanlar haftada en fazla 2 gün uzaktan çalışabilir.",
            "Deneme süresindeki çalışanlar uzaktan çalışamaz.",
            "Ev ofis kurulumu için tek seferlik 5.000 TL destek ödenir.",
        ]
        rerank_resp = client.post("/rerank", json={"query": query, "documents": documents}).raise_for_status().json()
        for score, text in sorted(zip(rerank_resp["scores"], documents), reverse=True):
            print(f"  {score:.3f}  {text}")

        print("\n== throughput: 128 chunk-sized texts")
        chunk = "Yıllık izin talepleri en az 10 iş günü önce Kalkan Portal üzerinden yapılır. " * 12
        started = time.perf_counter()
        batch_resp = client.post("/embed", json={"texts": [chunk] * 128}).raise_for_status().json()
        round_trip = (time.perf_counter() - started) * 1000
        print(f"  server: {batch_resp['elapsed_ms']:.0f} ms, round trip: {round_trip:.0f} ms, "
              f"truncated: {len(batch_resp['truncated'])}")

        print("\n== auth: wrong key must be rejected")
        wrong = httpx.post(f"{BASE_URL}/embed", json={"texts": ["x"]}, headers={"X-API-Key": "wrong"}, timeout=10)
        print(f"  status: {wrong.status_code} (expected 401)")


if __name__ == "__main__":
    main()
