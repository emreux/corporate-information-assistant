"""
Steps 5-6: measure how well the sample documents were read, using facts from the answer key.

For every document it checks phrases that MUST appear (facts, headings, table rows) and
phrases that must NOT appear (page headers and footers, website menus, broken table output).

Usage (from the project root):
    python -m scripts.check_ingestion data/processed/naive
    python -m scripts.check_ingestion data/processed/docling
"""
import argparse
import re
import sys
from pathlib import Path

# Each check is (description, phrase). Phrases are matched case-insensitively, with all
# whitespace (including line breaks) collapsed to a single space. A phrase starting with
# "re:" is a regular expression matched against the raw text instead.
# The phrases are Turkish because they are copied from the documents themselves.
CHECKS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "IK_Politikasi_2025.pdf": {
        "must": [
            ("fact: old leave duration", "20 gün"),
        ],
        "must_not": [
            ("page footer removed", "Gizlilik Sınıfı"),
            ("page number removed", "Sayfa 1"),
        ],
    },
    "IK_Politikasi_2026.pdf": {
        "must": [
            ("fact: leave for 5-15 years", "22 gün"),
            ("fact: extra leave for managers", "yılda 2 gün ilave"),
            ("heading kept as heading", "# 5.1 İzin Süreleri"),
            ("table row kept together", "| 5 yıldan fazla, 15 yıldan az | 22 gün |"),
        ],
        "must_not": [
            ("page footer removed", "Gizlilik Sınıfı"),
            ("page number removed", "Sayfa 1"),
            ("page header removed", "KLK-IK-POL-01 Rev.3"),
        ],
    },
    "Yan_Haklar_ve_Uzaktan_Calisma_Yonetmeligi.docx": {
        "must": [
            ("fact: meal card amount", "3.250 TL"),
            ("fact: probation exception", "Deneme süresindeki çalışanlar uzaktan çalışamaz"),
            ("heading kept as heading", "# 7.2 Kapsam Dışı Kalan Çalışanlar"),
            ("table row kept together", "| Ev ofis desteği | 5.000 TL (tek seferlik) |"),
        ],
        "must_not": [
            ("page footer removed", "Gizlilik Sınıfı"),
        ],
    },
    "BT_Destek_Prosedurleri.docx": {
        "must": [
            ("fact: form code", "KLK-IT-09"),
            ("fact: error code inside a table", "VPN-ERR-203"),
            ("table row kept together", "| VPN-ERR-204 | İstemcinin DNS yapılandırması hatalı. |"),
            ("heading kept as heading", "# 4.1 Parolamı Unuttum"),
        ],
        "must_not": [
            ("page footer removed", "Gizlilik Sınıfı"),
        ],
    },
    "SOC_Olay_Mudahale_Playbooklari.pdf": {
        "must": [
            ("fact: do not shut the device down", "kesinlikle kapatmayın"),
            ("fact: SOC manager extension", "4401"),
            ("table row after a page break", "| P4 - Düşük |"),
            ("heading kept as heading", "# PB-01 Fidye Yazılımı"),
        ],
        "must_not": [
            ("page footer removed", "Gizlilik Sınıfı"),
            ("page number removed", "Sayfa 2"),
            ("page header removed", "KLK-SOC-PB-00 Rev.5"),
        ],
    },
    "Hizmet_Paketleri_ve_Fiyatlar.xlsx": {
        "must": [
            ("fact: package code", "KLK-SOC-110"),
            ("second sheet read", "birleştirilemez"),
            ("price formatted as on screen", "Ücret (TL): 85.000"),
            ("sheet name kept as heading", "## İndirimler"),
            ("row written as a record", "Koşul: Kamu kurumları | İndirim Oranı: %10"),
            ("note under the table kept", "Fiyatlara KDV dahil değildir"),
        ],
        "must_not": [
            ("no empty-cell noise", r"re:\bNaN\b"),
            ("no broken-header noise", r"re:Unnamed: \d"),
        ],
    },
    "Yonetici_Maas_Skalasi_2026.xlsx": {
        "must": [
            ("fact: salary band", "K4 - Müdür"),
            ("price formatted as on screen", "Brüt Aylık Maksimum (TL): 250.000"),
            ("real header row detected", "Kademe: K4 - Müdür | Unvan Örnekleri: SOC Müdürü"),
            ("sheet name kept as heading", "## Ücret Skalası 2026"),
            ("confidentiality notice kept", "Bu doküman GİZLİDİR"),
        ],
        "must_not": [
            ("no empty-cell noise", r"re:\bNaN\b"),
            ("no broken-header noise", r"re:Unnamed: \d"),
            ("merged title not repeated", r"re:(?s)ÜCRET SKALASI \(GİZLİ\).*ÜCRET SKALASI \(GİZLİ\)"),
            ("empty cells skipped", "Maksimum (TL): |"),
        ],
    },
    "Musteri_Listesi.csv": {
        "must": [
            ("fact: customer name", "Mavi Liman Lojistik A.Ş."),
            ("row written as a record", "Firma: Akdeniz Gıda Sanayi A.Ş. | Sektor: Gıda"),
            ("date next to its column name", "SozlesmeBitis: 30.11.2026"),
        ],
        "must_not": [
            ("columns split on ';'", "MusteriNo;Firma"),
            ("byte order mark removed", "\ufeff"),
        ],
    },
    "Kalkan_Sirket_Tanitimi_2026.pptx": {
        "must": [
            ("fact: headcount on a slide", "185 çalışan"),
            ("speaker notes read", "200 aktif müşteri"),
        ],
        "must_not": [],
    },
    "Sik_Sorulan_Sorular.html": {
        "must": [
            ("fact: no parking in Ankara", "Ankara ofisinin kendi otoparkı yoktur"),
            ("question kept as heading", "# Maslak ofisinde otopark var mı?"),
        ],
        "must_not": [
            ("website menu removed", "Ana Sayfa"),
            ("cookie banner removed", "çerez kullanır"),
            ("site footer removed", "Tüm hakları saklıdır"),
        ],
    },
    "Yonetim_Kurulu_Tutanagi_2024-09_TARANMIS.pdf": {
        "must": [
            ("text extracted at all (OCR)", "Yönetim Kurulu"),
            ("fact: SIEM budget", "4.200.000"),
            ("Turkish characters read correctly", "Kozyatağı"),
            ("sentence read without distortion", "kira üst limiti"),
        ],
        "must_not": [],
    },
}


def normalize(text: str) -> str:
    """Turkish-aware lowercase and whitespace collapsing, so line wraps do not break phrases."""
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return re.sub(r"\s+", " ", text)


def contains(raw: str, normalized: str, phrase: str) -> bool:
    if phrase.startswith("re:"):
        return re.search(phrase[3:], raw) is not None
    return normalize(phrase) in normalized


def find_output(folder: Path, document: str) -> Path | None:
    """Outputs are named after the original file plus an extension, e.g. 'a.pdf.txt'."""
    matches = sorted(folder.glob(f"{document}.*"))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check reader outputs against the answer key.")
    parser.add_argument("folder", type=Path, help="Folder with the reader outputs.")
    args = parser.parse_args()

    total_passed = total_checks = 0
    for document, checks in CHECKS.items():
        output = find_output(args.folder, document)
        results: list[tuple[bool, str, str, str]] = []
        if output is None:
            for kind in ("must", "must_not"):
                results += [(False, "MISSING", d, p) for d, p in checks[kind]]
        else:
            raw = output.read_text(encoding="utf-8")
            normalized = normalize(raw)
            for description, phrase in checks["must"]:
                found = contains(raw, normalized, phrase)
                results.append((found, "ok" if found else "MISS", description, phrase))
            for description, phrase in checks["must_not"]:
                found = contains(raw, normalized, phrase)
                results.append((not found, "ok" if not found else "LEAK", description, phrase))

        passed = sum(ok for ok, *_ in results)
        total_passed += passed
        total_checks += len(results)
        print(f"\n{document}  {passed}/{len(results)}")
        for ok, status, description, phrase in results:
            shown = phrase if len(phrase) <= 55 else phrase[:52] + "..."
            print(f"   [{status:4s}] {description:34s} {shown!r}")

    print(f"\nTOTAL: {total_passed}/{total_checks} checks passed ({args.folder})")
    sys.exit(0 if total_passed == total_checks else 1)


if __name__ == "__main__":
    main()