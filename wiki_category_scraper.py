"""
wiki_category_scraper.py
-------------------------
Holt automatisch ALLE Artikel aus einer oder mehreren Wikipedia-
Kategorien (über die offizielle MediaWiki-API, sauberer Klartext statt
HTML-Scraping) und hängt sie an data/corpus.txt an.

Deutlich effizienter als einzelne URLs von Hand raussuchen: eine
Kategorie wie "Kategorie:Hardware" kann schon aus hunderten Artikeln
bestehen.

Nutzung:
    In categories.txt Kategorienamen eintragen (eine pro Zeile), dann:
    python wiki_category_scraper.py
"""

import time
from pathlib import Path

import requests

API_URL = "https://de.wikipedia.org/w/api.php"
CATEGORIES_FILE = Path("categories.txt")
OUTPUT_FILE = Path("data/corpus.txt")
MAX_PAGES_PER_CATEGORY = 150  # Sicherheitslimit, damit's nicht ausufert
BATCH_SIZE = 20               # Wikipedia erlaubt bis zu 20 Titel pro Anfrage (ohne API-Key)
REQUEST_DELAY_SEC = 1.0
MAX_RETRIES = 6
BACKOFF_BASE_SEC = 5          # Wartezeit verdoppelt sich mit jedem Fehlversuch (5, 10, 20, 40 ...)

HEADERS = {"User-Agent": "MeinLLM-Lernprojekt/1.0 (privates, nicht-kommerzielles Projekt)"}


def api_get(session, params, timeout):
    """Wie session.get, aber mit automatischem Warten + Wiederholen bei 429 (Too Many Requests)."""
    for attempt in range(1, MAX_RETRIES + 1):
        resp = session.get(API_URL, params=params, headers=HEADERS, timeout=timeout)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            print(f"    -> 429 Too Many Requests, warte {wait:.0f}s "
                  f"(Versuch {attempt}/{MAX_RETRIES}) ...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError(f"Nach {MAX_RETRIES} Versuchen weiterhin 429 - später erneut probieren.")


def get_category_members(session, category: str, limit: int) -> list[str]:
    """Holt alle Artikeltitel einer Kategorie (mit Pagination über cmcontinue)."""
    titles = []
    cmcontinue = None

    while len(titles) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": "50",
            "cmnamespace": "0",  # nur echte Artikel, keine Unterkategorien/Diskussionen
            "format": "json",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        resp = api_get(session, params, timeout=10)
        data = resp.json()

        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)

        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(REQUEST_DELAY_SEC)

    return titles[:limit]


def get_extracts(session, titles: list[str]) -> dict[str, str]:
    """Holt reinen Fließtext (ohne Wiki-Markup, Infoboxen, Navigationsboxen) für Artikeltitel."""
    extracts = {}

    for i in range(0, len(titles), BATCH_SIZE):
        batch = titles[i:i + BATCH_SIZE]
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": "|".join(batch),
            "format": "json",
        }
        resp = api_get(session, params, timeout=15)
        data = resp.json()

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "?")
            text = page.get("extract", "")
            if text:
                extracts[title] = text

        time.sleep(REQUEST_DELAY_SEC)

    return extracts


def main():
    if not CATEGORIES_FILE.exists():
        CATEGORIES_FILE.write_text(
            "# Eine Wikipedia-Kategorie pro Zeile, z.B.:\n"
            "# Kategorie:Hardware\n"
            "# Kategorie:Betriebssystem\n"
        )
        print(f"'{CATEGORIES_FILE}' wurde angelegt. Bitte Kategorien eintragen und erneut starten.")
        return

    categories = [
        line.strip()
        for line in CATEGORIES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not categories:
        print("Keine Kategorien in categories.txt gefunden.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    all_titles: set[str] = set()

    for cat in categories:
        print(f"Hole Artikelliste für '{cat}' ...")
        try:
            titles = get_category_members(session, cat, MAX_PAGES_PER_CATEGORY)
            print(f"    -> {len(titles)} Artikel gefunden")
            all_titles.update(titles)
        except Exception as e:
            print(f"    -> Fehler bei Kategorie '{cat}': {e}")
        time.sleep(REQUEST_DELAY_SEC)

    print(f"\nInsgesamt {len(all_titles)} eindeutige Artikel. Lade Texte ...")
    titles_list = sorted(all_titles)

    written = 0
    with OUTPUT_FILE.open("a", encoding="utf-8") as out:
        for i in range(0, len(titles_list), BATCH_SIZE):
            batch = titles_list[i:i + BATCH_SIZE]
            try:
                extracts = get_extracts(session, batch)
                for title, text in extracts.items():
                    out.write(text + "\n\n")
                    written += 1
                print(f"    [{min(i + BATCH_SIZE, len(titles_list))}/{len(titles_list)}] verarbeitet")
            except Exception as e:
                print(f"    -> Fehler bei Batch {batch}: {e}")

    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"\nFertig. {written} Artikel angehängt. Korpus jetzt {size_kb:.1f} KB groß ({OUTPUT_FILE}).")


if __name__ == "__main__":
    main()
