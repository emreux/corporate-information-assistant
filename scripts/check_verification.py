"""
Step 19: check the citation verification on made-up answers (needs nothing else, no LLM).

Every number and code of an answer must be in the sources the answer cites for it. The cases below
are the mistakes it must catch and the correct answers it must leave alone (other number formats,
numbers from the question, numbered lists).

Usage (from the project root):
    python -m scripts.check_verification
"""
import sys

from rag.citations import verify_citations
from scripts.check_env import run_check

SOURCES = [
    "[IK_Politikasi_2026.pdf > 5.1 İzin Süreleri]\n| 5 yıldan 15 yıla kadar | 22 gün |\n"
    "Yöneticilere yılda 2 gün ilave izin verilir.",
    "[BT_Destek_Prosedurleri.docx > 3. VPN Bağlantı Sorunları]\n"
    "| VPN-ERR-301 | Eş zamanlı bağlantı lisans limiti aşıldı. |\nTelefon: Dahili 2020, 08:00 - 20:00.",
    "[Musteri_Listesi.csv]\nMusteriNo: M-1014 | Firma: Yeşilvadi Belediyesi | SozlesmeBitis: 31.08.2028 | "
    "Aylık ücret: 85.000 TL",
]


def expect(answer: str, wanted: list[str], question: str = "", found_in: tuple[int, ...] | None = None) -> str:
    items = verify_citations(answer, SOURCES, question)
    facts = [item.fact for item in items]
    if facts != wanted:
        raise AssertionError(f"{answer!r} -> {facts}, expected {wanted}")
    if found_in is not None and items[0].found_in != found_in:
        raise AssertionError(f"{answer!r}: found in {items[0].found_in}, expected {found_in}")
    return f"{answer!r} -> {facts or 'nothing to check'}"


def check_nothing_to_check() -> str:
    expect("Dokümanlarda bu bilgi yok. [NOT_FOUND]", [])
    return expect("İzin 30 gündür.", [])          # no citation: reported by the citation check instead


CHECKS = {
    "Right number": lambda: expect("8 yıllık bir çalışanın izni 22 gündür [1].", [],
                                   question="8 yıllık bir çalışanın kaç gün izni var?"),
    "Wrong number": lambda: expect("Yemek kartına aylık 2.750 TL yüklenir [1].", ["2.750"]),
    "Wrong source cited": lambda: expect("VPN-ERR-301: lisans limiti aşıldı [1].", ["VPN-ERR-301"], found_in=(2,)),
    "Calculated number is reported": lambda: expect("8 yıllık bir müdürün izni 24 gündür [1].", ["24"],
                                                    question="8 yıllık bir müdürün kaç gün izni var?"),
    "Other ways to write it": lambda: expect(
        "VPN-ERR301 için oturumu kapatın [2]. Sözleşme 31 Ağustos 2028'de bitiyor, aylık 85 bin TL [3]. "
        "BT Destek 08:00-20:00 arası açık [2].", []),
    "Numbered list": lambda: expect(
        "1. Başka cihazdaki oturumu kapatın [2].\n2. Sürerse dahili 2020'yi arayın [2].", []),
    "Wrong code": lambda: expect("Donanım için KLK-IT-07 formunu doldurun [2].", ["KLK-IT-07"]),
    "Not found, or no citation at all: nothing to check": check_nothing_to_check,
}


def main() -> None:
    results = [run_check(name, check) for name, check in CHECKS.items()]
    print(f"\nTOTAL: {sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
