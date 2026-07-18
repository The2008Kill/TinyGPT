"""
train.py
--------
Trainiert das GPT-Modell auf dem vorbereiteten Korpus.

Speichert automatisch alle SAVE_INTERVAL Schritte einen Checkpoint
zwischendurch - bei Strg+C geht daher höchstens der Fortschritt seit
dem letzten Zwischenspeichern verloren, nicht das ganze Training.

Nutzung:
    python train.py                 # neues Training von Null
    python train.py --resume         # Training auf ckpt.pt fortsetzen
"""

import argparse
import pickle
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig

# ---- Einstellungen ----------------------------------------------------
BLOCK_SIZE = 128        # Kontextlänge
BATCH_SIZE = 64
N_LAYER = 4
N_HEAD = 4
N_EMBD = 128
DROPOUT = 0.1

MAX_ITERS = 3000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
SAVE_INTERVAL = 250     # alle wie vielen Schritten zwischendurch gespeichert wird
LEARNING_RATE = 3e-4
CHECKPOINT_PATH = Path("ckpt.pt")
# -------------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training läuft auf: {device}")


def load_data():
    meta_path = Path("data/meta.pkl")
    if not meta_path.exists():
        raise FileNotFoundError("data/meta.pkl fehlt. Erst 'python prepare_data.py' ausführen.")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    train_data = torch.load("data/train.pt")
    val_data = torch.load("data/val.pt")
    return train_data, val_data, meta


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(model, config, meta, step):
    torch.save(
        {"model_state": model.state_dict(), "config": config, "meta": meta, "step": step},
        CHECKPOINT_PATH,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Auf vorhandenem ckpt.pt weitertrainieren statt neu zu starten")
    args = parser.parse_args()

    train_data, val_data, meta = load_data()

    config = GPTConfig(
        vocab_size=meta["vocab_size"],
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    )
    model = GPT(config).to(device)

    if args.resume:
        if not CHECKPOINT_PATH.exists():
            print(f"'{CHECKPOINT_PATH}' nicht gefunden - starte stattdessen neues Training.")
        else:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            old_vocab_size = ckpt["config"].vocab_size
            if old_vocab_size != meta["vocab_size"]:
                print(f"Warnung: Vokabulargröße hat sich geändert ({old_vocab_size} -> "
                      f"{meta['vocab_size']}), z.B. durch neue Sonderzeichen im erweiterten "
                      f"Korpus. Alter Checkpoint passt nicht mehr zusammen - starte neu.")
            else:
                model.load_state_dict(ckpt["model_state"])
                print(f"Setze Training von '{CHECKPOINT_PATH}' fort.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    t0 = time.time()
    step = 0
    try:
        for step in range(MAX_ITERS + 1):
            if step % EVAL_INTERVAL == 0:
                losses = estimate_loss(model, train_data, val_data)
                elapsed = time.time() - t0
                print(f"Schritt {step:5d} | train loss {losses['train']:.4f} | "
                      f"val loss {losses['val']:.4f} | {elapsed:.0f}s")

            if step % SAVE_INTERVAL == 0 and step > 0:
                save_checkpoint(model, config, meta, step)
                print(f"    (Zwischenstand bei Schritt {step} gespeichert)")

            x, y = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    except KeyboardInterrupt:
        save_checkpoint(model, config, meta, step)
        print(f"\nAbgebrochen bei Schritt {step}. Zwischenstand wurde gespeichert in {CHECKPOINT_PATH}.")
        print("Weiter mit: python train.py --resume")
        return

    save_checkpoint(model, config, meta, MAX_ITERS)
    print(f"\nTraining fertig. Checkpoint gespeichert in {CHECKPOINT_PATH}")
    print("Weiter mit: python generate.py")


if __name__ == "__main__":
    main()
