"""
prepare_data.py
---------------
Liest data/corpus.txt, kodiert den Text mit dem trainierten
Wortstück-Tokenizer (data/tokenizer/) und speichert Trainings-/
Validierungsdaten als Tensoren ab.

Vorher: python train_tokenizer.py ausführen.

Nutzung:
    python prepare_data.py
"""

import pickle
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

DATA_FILE = Path("data/corpus.txt")
TOKENIZER_DIR = Path("data/tokenizer")
OUT_DIR = Path("data")


def main():
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
        print(f"'{DATA_FILE}' fehlt oder ist leer.")
        print("Erst scraper.py / wiki_category_scraper.py laufen lassen.")
        return

    vocab_file = TOKENIZER_DIR / "vocab.json"
    merges_file = TOKENIZER_DIR / "merges.txt"
    if not vocab_file.exists() or not merges_file.exists():
        print(f"Tokenizer fehlt in {TOKENIZER_DIR}/.")
        print("Erst 'python train_tokenizer.py' ausführen.")
        return

    tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer geladen (Vokabulargröße: {vocab_size})")

    text = DATA_FILE.read_text(encoding="utf-8")
    print(f"Korpuslänge: {len(text):,} Zeichen")

    ids = tokenizer.encode(text).ids
    print(f"Kodiert zu {len(ids):,} Tokens "
          f"(~{len(text) / max(len(ids), 1):.1f} Zeichen pro Token)")

    data = torch.tensor(ids, dtype=torch.long)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    torch.save(train_data, OUT_DIR / "train.pt")
    torch.save(val_data, OUT_DIR / "val.pt")

    with open(OUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump({"vocab_size": vocab_size}, f)

    print(f"Gespeichert: {len(train_data):,} Trainings- / {len(val_data):,} Validierungs-Tokens")
    print("Fertig. Weiter mit: python train.py  (OHNE --resume, neues Vokabular!)")


if __name__ == "__main__":
    main()
