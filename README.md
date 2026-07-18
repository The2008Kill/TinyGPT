# Mein eigenes LLM

Ein kleines, selbst geschriebenes Sprachmodell (Transformer, GPT-Architektur)
zum Selbstlernen. Zeichen-basiert, komplett von Grund auf implementiert –
kein vortrainiertes Modell, keine Blackbox.

**Wichtig:** Das ist ein Lernprojekt, kein ChatGPT-Ersatz. Mit ein paar
hunderttausend Zeichen Trainingsdaten und einer CPU/GPU zu Hause bekommst
du ein Modell, das Textmuster und Stil lernt – nicht "Wissen" oder
Schlussfolgerungsfähigkeit im großen Stil.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

In IntelliJ IDEA: Settings → Plugins → "Python" installieren, dann das
Projekt als Python-Projekt öffnen und `venv` als Interpreter auswählen.

## Ablauf

### 1. Trainingsdaten sammeln
URLs in `urls.txt` eintragen (eine pro Zeile), dann:
```bash
python scraper.py
```
Ergebnis landet in `data/corpus.txt`. Du kannst dort auch einfach eigenen
Text reinkopieren/ergänzen (eigene Notizen, Bücher als .txt, etc.) —
je mehr thematisch einheitlicher Text, desto konsistenter das Ergebnis.
Für brauchbare Resultate: mindestens ein paar MB Text.

### 2. Daten aufbereiten
```bash
python prepare_data.py
```
Baut den Zeichen-Tokenizer und speichert Trainings-/Validierungsdaten.

### 3. Trainieren
```bash
python train.py
```
Läuft standardmäßig 3000 Schritte. Auf CPU dauert das je nach Korpusgröße
einige Minuten bis Stunden, mit GPU deutlich schneller. Verlustwerte
(`train loss` / `val loss`) sollten über die Zeit sinken.

Parameter zum Anpassen stehen oben in `train.py`:
- `N_LAYER`, `N_HEAD`, `N_EMBD` → Modellgröße
- `BLOCK_SIZE` → wie viel Kontext das Modell auf einmal sieht
- `MAX_ITERS` → Trainingsdauer

### 4. Text generieren
```bash
python generate.py --prompt "Es war einmal" --length 300
```

## Nächste Schritte, wenn du weiterkommen willst
- **Größerer Korpus**: mehr/bessere Scraper-Quellen, HTML-Filterung verfeinern
- **Besserer Tokenizer**: von Zeichen- auf Wortstück-Ebene wechseln (z.B. BPE
  via `tiktoken`) → das Modell lernt schneller und kohärenter
- **GPU-Training**: z.B. über eine Cloud-GPU (Colab, Lambda, RunPod), falls
  keine eigene Grafikkarte vorhanden ist
- **Websuche anbinden**: separates Such-Tool schreiben (z.B. über eine
  Such-API), das Ergebnisse als Kontext ins Modell einspeist (Retrieval) –
  das ist unabhängig vom Training selbst
