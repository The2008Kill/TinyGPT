"""
generate.py
-----------
Lädt den trainierten Checkpoint und generiert damit Text.

Nutzung:
    python generate.py
    python generate.py --prompt "Es war einmal" --length 300
"""

import argparse
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

from model import GPT

CHECKPOINT_PATH = Path("ckpt.pt")
TOKENIZER_DIR = Path("data/tokenizer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Der Computer")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    if not CHECKPOINT_PATH.exists():
        print(f"'{CHECKPOINT_PATH}' fehlt. Erst 'python train.py' ausführen.")
        return

    vocab_file = TOKENIZER_DIR / "vocab.json"
    merges_file = TOKENIZER_DIR / "merges.txt"
    if not vocab_file.exists() or not merges_file.exists():
        print(f"Tokenizer fehlt in {TOKENIZER_DIR}/. Erst 'python train_tokenizer.py' ausführen.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    config = ckpt["config"]
    tokenizer = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))

    if tokenizer.get_vocab_size() != config.vocab_size:
        print(
            f"Vokabular passt nicht zusammen: das gespeicherte Modell erwartet "
            f"{config.vocab_size} Tokens, der aktuelle Tokenizer hat aber "
            f"{tokenizer.get_vocab_size()}.\n"
            f"Vermutlich wurde der Tokenizer neu trainiert, ohne das Modell "
            f"danach neu zu trainieren.\n"
            f"Fix: 'python prepare_data.py' und danach 'python train.py' "
            f"(OHNE --resume) erneut ausführen."
        )
        return

    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([tokenizer.encode(args.prompt).ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=args.length,
                          temperature=args.temperature, top_k=args.top_k)

    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
