"""
Step 7: check the chunker, then chunk the sample documents and save the chunks for reading.

Part 1 uses small made-up documents for cases the samples do not contain: a paragraph, a table
and a list that are too big for one chunk, text without any full stop and text without headings.
Part 2 reads and chunks every sample document, writes the chunks to data/processed/chunks/
and checks them against the traps in the answer key.

Usage (from the project root):
    python -m scripts.check_chunks
"""
import logging
import re
import statistics
import sys
import time
from collections import Counter

import config
from rag.chunker import Chunk, chunk_document, count_tokens
from rag.reader import DocumentPage, ParsedDocument, read_document_cached as read_document  # step 9
from scripts.check_env import run_check
from scripts.check_ingestion import normalize

OUTPUT_DIR = config.PROCESSED_DIR / "chunks"
SIZE_TOLERANCE = 10     # the context line and the joins between blocks may add a few tokens

# Phrases that must all be in ONE chunk: (document, description, phrases, expected pages or None).
# Phrases ending with "]" match the context line at the top of a chunk.
# The phrases are Turkish because they are copied from the documents themselves.
TOGETHER: list[tuple[str, str, list[str], tuple[int, int] | None]] = [
    ("IK_Politikasi_2026.pdf", "T12: heading and its table on the next page",
     ["> 5.1 İzin Süreleri]", "| 5 yıldan fazla, 15 yıldan az | 22 gün |"], (1, 2)),
    ("IK_Politikasi_2026.pdf", "section number gives the level",
     ["5. Yıllık Ücretli İzin > 5.2 Yöneticiler İçin İlave İzin]", "yılda 2 gün ilave"], None),
    ("SOC_Olay_Mudahale_Playbooklari.pdf", "T13: row after a page break keeps its header",
     ["> 3. Olay Önem Seviyeleri]", "| Seviye | Tanım |", "| P4 - Düşük |"], None),
    ("SOC_Olay_Mudahale_Playbooklari.pdf", "umbrella heading and label rule",
     ["> 5. Playbook'lar > PB-01 Fidye Yazılımı (Ransomware)]", "Müdahale adımları:",
      "kesinlikle kapatmayın"], None),
    ("SOC_Olay_Mudahale_Playbooklari.pdf", "appendix is not under section 8",
     ["[SOC_Olay_Mudahale_Playbooklari.pdf > Ek-A: İletişim Listesi]", "4401"], None),
    ("Yan_Haklar_ve_Uzaktan_Calisma_Yonetmeligi.docx", "exception keeps its heading",
     ["> 7. Uzaktan (Hibrit) Çalışma > 7.2 Kapsam Dışı Kalan Çalışanlar]",
      "Deneme süresindeki çalışanlar uzaktan çalışamaz"], None),
    ("Hizmet_Paketleri_ve_Fiyatlar.xlsx", "sheet name in the context line",
     ["> İndirimler]", "Koşul: Kamu kurumları"], (2, 2)),
    ("Kalkan_Sirket_Tanitimi_2026.pptx", "slide and its speaker notes together",
     ["> Biz Kimiz?]", "200 aktif müşteri"], (2, 2)),
]

# Phrases that must be in DIFFERENT chunks: (document, description, phrase, other phrase).
APART: list[tuple[str, str, str, str]] = [
    ("Sik_Sorulan_Sorular.html", "one question per chunk",
     "Ankara ofisinin kendi otoparkı yoktur", "40 araçlık alan"),
]


# --- Part 1: made-up documents ------------------------------------------------------

def _document(markdown: str) -> ParsedDocument:
    return ParsedDocument(source="test.md", file_type="md", pages=[DocumentPage(1, markdown)])


def _content(chunk: Chunk) -> str:
    """The chunk without its context line."""
    return chunk.text.partition("\n")[2]


def _expect_sizes_and_units(chunks: list[Chunk], units: list[str], min_chunks: int = 2) -> str:
    """Several chunks, none far above the target, every unit whole in at least one chunk."""
    if len(chunks) < min_chunks:
        raise AssertionError(f"Expected at least {min_chunks} chunks, got {len(chunks)}.")
    too_big = [chunk.token_count for chunk in chunks
               if chunk.token_count > config.CHUNK_SIZE_TOKENS + SIZE_TOLERANCE]
    if too_big:
        raise AssertionError(f"Chunks above the target: {too_big}")
    missing = [unit for unit in units if not any(unit in chunk.text for chunk in chunks)]
    if missing:
        raise AssertionError(f"{len(missing)} units lost or broken, e.g. {missing[0]!r}")
    return f"{len(chunks)} chunks, largest {max(chunk.token_count for chunk in chunks)} tokens"


def check_long_paragraph() -> str:
    sentences = [f"Madde {i}: çalışanlar bu kurala uymakla yükümlüdür ve istisna yoktur." for i in range(80)]
    chunks = chunk_document(_document("## 1. Kurallar\n\n" + " ".join(sentences)))
    result = _expect_sizes_and_units(chunks, sentences)
    for previous, current in zip(chunks, chunks[1:]):
        if not any(s in previous.text and s in current.text for s in sentences):
            raise AssertionError(f"No overlap between chunk {previous.chunk_index} and {current.chunk_index}.")
    return result + ", neighbours overlap"


def check_long_table() -> str:
    header = "| Kod | Açıklama | Tutar |\n|---|---|---|"
    rows = [f"| K-{i} | Hizmet kalemi numarası {i} | {i * 100} TL |" for i in range(150)]
    chunks = chunk_document(_document("## Fiyatlar\n\n" + header + "\n" + "\n".join(rows)))
    result = _expect_sizes_and_units(chunks, rows)
    without_header = [chunk.chunk_index for chunk in chunks if not _content(chunk).startswith(header)]
    if without_header:
        raise AssertionError(f"Chunks without the table header: {without_header}")
    return result + ", header repeated"


def check_long_list() -> str:
    items = [f"- Kural {i}: Şirket cihazlarında yalnızca onaylı yazılımlar kullanılır." for i in range(150)]
    return _expect_sizes_and_units(chunk_document(_document("## Kurallar\n\n" + "\n".join(items))), items)


def check_text_without_full_stops() -> str:
    words = [f"kelime{i}" for i in range(3000)]                 # like a badly scanned page
    chunks = chunk_document(_document(" ".join(words)))
    result = _expect_sizes_and_units(chunks, [])
    found = set(re.findall(r"kelime\d+", " ".join(chunk.text for chunk in chunks)))
    if found != set(words):
        raise AssertionError(f"{len(set(words) - found)} words lost")
    return result


def check_text_without_headings() -> str:
    chunks = chunk_document(_document("Başlığı olmayan kısa bir metin."))
    if len(chunks) != 1 or not chunks[0].text.startswith("[test.md]\n") or chunks[0].heading_path:
        raise AssertionError(f"Unexpected result: {chunks}")
    return "one chunk with the file name as context"


# --- Part 2: sample documents -------------------------------------------------------

def write_chunks(name: str, chunks: list[Chunk]) -> None:
    """Save the chunks with a separator line each, for reading them one by one."""
    parts = [f"<!-- {name}: {len(chunks)} chunks -->"]
    for chunk in chunks:
        if chunk.page_start is None:
            pages = "-"
        elif chunk.page_start == chunk.page_end:
            pages = str(chunk.page_start)
        else:
            pages = f"{chunk.page_start}-{chunk.page_end}"
        parts.append(f"===== chunk {chunk.chunk_index} | pages {pages} | {chunk.token_count} tokens =====\n"
                     f"{chunk.text}")
    (OUTPUT_DIR / f"{name}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def general_problems(document: ParsedDocument, chunks: list[Chunk]) -> list[str]:
    """Checks that apply to every document: size limit, empty chunks, lost words, broken lines."""
    problems = []
    too_big = [chunk.chunk_index for chunk in chunks if chunk.token_count > config.CHUNK_MAX_TOKENS]
    if too_big:
        problems.append(f"above the hard limit of {config.CHUNK_MAX_TOKENS} tokens: chunks {too_big}")
    empty = [chunk.chunk_index for chunk in chunks if not _content(chunk).strip()]
    if empty:
        problems.append(f"chunks without content: {empty}")

    chunk_texts = [normalize(chunk.text) for chunk in chunks]
    lost = Counter(re.findall(r"\w+", normalize(document.text)))
    lost.subtract(Counter(re.findall(r"\w+", " ".join(chunk_texts))))
    lost_words = [word for word, count in lost.items() if count > 0]
    if lost_words:
        problems.append(f"{len(lost_words)} words lost, e.g. {lost_words[:8]}")

    broken = []
    for line in document.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or count_tokens(line) > config.CHUNK_SIZE_TOKENS // 2:
            continue
        if not any(normalize(line) in text for text in chunk_texts):
            broken.append(line)
    if broken:
        problems.append(f"{len(broken)} short lines broken across chunks, e.g. {broken[0][:60]!r}")
    return problems


def find_chunks(chunks: list[Chunk], phrases: list[str]) -> list[Chunk]:
    return [chunk for chunk in chunks if all(normalize(p) in normalize(chunk.text) for p in phrases)]


def check_samples() -> tuple[int, int]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in config.SAMPLES_DIR.iterdir() if path.is_file())
    print(f"\nReading (from the cache when possible) and chunking {len(files)} files\n")
    print(f"{'File':50s} {'Chunks':>6s} {'Min':>5s} {'Median':>6s} {'Max':>5s} {'Sec':>6s}")

    results: dict[str, tuple[ParsedDocument, list[Chunk]]] = {}
    for path in files:
        started = time.perf_counter()
        try:
            document = read_document(path)
        except Exception as exc:
            print(f"{path.name:50s} ERROR: {type(exc).__name__}: {exc}")
            continue
        chunks = chunk_document(document)
        results[path.name] = (document, chunks)
        write_chunks(path.name, chunks)
        sizes = [chunk.token_count for chunk in chunks] or [0]
        print(f"{path.name:50s} {len(chunks):>6d} {min(sizes):>5d} {statistics.median(sizes):>6.0f} "
              f"{max(sizes):>5d} {time.perf_counter() - started:>6.1f}")

    all_sizes = [chunk.token_count for _, chunks in results.values() for chunk in chunks]
    if all_sizes:
        print(f"{'ALL':50s} {len(all_sizes):>6d} {min(all_sizes):>5d} {statistics.median(all_sizes):>6.0f} "
              f"{max(all_sizes):>5d}")

    passed = total = 0
    print("\nGeneral checks (every document):")
    for name, (document, chunks) in results.items():
        problems = general_problems(document, chunks)
        passed += not problems
        total += 1
        print(f"   [{'ok' if not problems else 'FAIL':4s}] {name}" + "".join(f"\n          - {p}" for p in problems))

    print("\nTrap checks:")
    for name, description, phrases, pages in TOGETHER:
        chunks = results.get(name, (None, []))[1]
        matches = find_chunks(chunks, phrases)
        ok = bool(matches) and (pages is None or (matches[0].page_start, matches[0].page_end) == pages)
        detail = "" if ok else ("  (no chunk has all phrases)" if not matches else
                                f"  (pages {matches[0].page_start}-{matches[0].page_end}, expected {pages[0]}-{pages[1]})")
        passed += ok
        total += 1
        print(f"   [{'ok' if ok else 'FAIL':4s}] {name}: {description}{detail}")
    for name, description, first, second in APART:
        chunks = results.get(name, (None, []))[1]
        ok = bool(find_chunks(chunks, [first])) and not find_chunks(chunks, [first, second])
        passed += ok
        total += 1
        print(f"   [{'ok' if ok else 'FAIL':4s}] {name}: {description}")

    print(f"\nChunks written to {OUTPUT_DIR}")
    return passed, total


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="       (log) %(message)s")
    print(f"Loading the tokenizer of {config.EMBEDDING_MODEL} (downloaded on the first run)...")
    count_tokens("warm-up")

    print("\nPart 1: made-up documents")
    edge_cases = {
        "Long paragraph": check_long_paragraph,
        "Long table": check_long_table,
        "Long list": check_long_list,
        "Text without full stops": check_text_without_full_stops,
        "Text without headings": check_text_without_headings,
    }
    edge_results = [run_check(name, check) for name, check in edge_cases.items()]

    print("\nPart 2: sample documents")
    passed, total = check_samples()

    passed += sum(edge_results)
    total += len(edge_results)
    print(f"\nTOTAL: {passed}/{total} checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()