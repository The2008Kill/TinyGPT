"""
scraper.py
----------
Sammelt Text von einer Liste von URLs und speichert alles in einer
einzigen Textdatei (data/corpus.txt), die als Trainingskorpus dient.

Nutzung:
    1. In urls.txt eine URL pro Zeile eintragen
    2. python scraper.py

Hinweis: Bitte nur Seiten scrapen, bei denen du die Erlaubnis dazu hast
bzw. deren robots.txt / Nutzungsbedingungen das erlauben.
"""

import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup

URLS_FILE = Path("urls.txt")
OUTPUT_FILE = Path("data/corpus.txt")
MIN_PARAGRAPH_LEN = 40  # zu kurze Zeilen (Menüs, Buttons) rausfiltern
REQUEST_DELAY_SEC = 1.0  # Höflichkeitspause zwischen Requests


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Skripte, Styles, Navigation etc. entfernen
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p", "h1", "h2", "h3", "li"])]
    paragraphs = [p for p in paragraphs if len(p) >= MIN_PARAGRAPH_LEN]
    return "\n".join(paragraphs)


def main():
    if not URLS_FILE.exists():
        URLS_FILE.write_text(
            "# Eine URL pro Zeile eintragen, z.B.:\n"
            "# https://de.wikipedia.org/wiki/Transformer_(Maschinelles_Lernen)\n"
        )
        print(f"'{URLS_FILE}' wurde angelegt. Bitte URLs eintragen und erneut starten.")
        return

    urls = [
        line.strip()
        for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not urls:
        print("Keine URLs in urls.txt gefunden.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (privates Lernprojekt)"}

    with OUTPUT_FILE.open("a", encoding="utf-8") as out:
        for i, url in enumerate(urls, 1):
            try:
                print(f"[{i}/{len(urls)}] Lade {url} ...")
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                text = extract_text(resp.text)
                out.write(text + "\n\n")
                print(f"    -> {len(text)} Zeichen extrahiert")
            except Exception as e:
                print(f"    -> Fehler bei {url}: {e}")
            time.sleep(REQUEST_DELAY_SEC)

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nFertig. Korpus liegt in {OUTPUT_FILE} ({size_kb:.1f} KB).")


if __name__ == "__main__":
    main()
