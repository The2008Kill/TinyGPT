"""
train_tokenizer.py
-------------------
Trainiert einen eigenen Byte-Level-BPE-Tokenizer auf data/corpus.txt.

Das ersetzt den bisherigen Zeichen-Tokenizer: statt jedes einzelne
Zeichen als Token zu behandeln, lernt dieser Tokenizer aus deinem
Korpus die häufigsten Wortbausteine (z.B. "Prozessor", "##oren",
"Schnitt", "##stelle") und fasst sie zu Tokens zusammen. Das Modell
muss Wörter dadurch nicht mehr buchstabenweise neu erfinden.

Nutzung:
    python train_tokenizer.py

Danach: prepare_data.py neu laufen lassen (Daten müssen mit dem neuen
Tokenizer neu kodiert werden), und train.py OHNE --resume neu starten
(altes Modell passt nicht mehr zum neuen Vokabular).
"""

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

CORPUS_FILE = Path("data/corpus.txt")
TOKENIZER_DIR = Path("data/tokenizer")
VOCAB_SIZE = 8000   # Anzahl unterschiedlicher Wortbausteine im Vokabular
MIN_FREQUENCY = 2   # Wortbausteine, die seltener vorkommen, werden nicht aufgenommen


def main():
    if not CORPUS_FILE.exists() or CORPUS_FILE.stat().st_size == 0:
        print(f"'{CORPUS_FILE}' fehlt oder ist leer.")
        print("Erst scraper.py bzw. wiki_category_scraper.py laufen lassen.")
        return

    size_kb = CORPUS_FILE.stat().st_size / 1024
    print(f"Trainiere Tokenizer auf {CORPUS_FILE} ({size_kb:.1f} KB) ...")

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(CORPUS_FILE)],
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer.save_model(str(TOKENIZER_DIR))

    print(f"Fertig. Tokenizer gespeichert in {TOKENIZER_DIR}/ "
          f"(Vokabulargröße: {tokenizer.get_vocab_size()})")
    print("Weiter mit: python prepare_data.py")


if __name__ == "__main__":
    main()
