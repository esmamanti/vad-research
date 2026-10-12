"""Download MORE noise-only clips from the train split, for the improvement experiment (notebook 04).

Run once from the project folder:
    python scripts/download_more_noise.py

Result: data/train/noise_extra/0.wav, 1.wav, ...

The first 30 noise clips of the train split are already in data/train/noise_only,
so they are skipped here. The test split is not touched: the test clips stay exactly the same.
"""
import io
import time
from pathlib import Path

import requests
import soundfile as sf

DATASET = "Aynursusuz/noisy-speech-dataset"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "train" / "noise_extra"
PAGE = 100

ALREADY_HAVE = 30      # noise clips already downloaded by download_data.py
EXTRA_WANTED = 60      # new noise clips to download


def get(url, **params):
    """Download a web address. If the server is busy, wait a little and try again (up to 5 times)."""
    for attempt in range(5):
        response = requests.get(url, params=params or None, timeout=60)
        if response.status_code == 200:
            return response
        print("  server answered", response.status_code, "- waiting 10 s and trying again", flush=True)
        time.sleep(10)
    response.raise_for_status()


OUT_DIR.mkdir(parents=True, exist_ok=True)
noise_rows_seen = 0
saved = 0
offset = 0

while saved < EXTRA_WANTED:
    rows = get(ROWS_URL, dataset=DATASET, config="default", split="train", offset=offset, length=PAGE).json()["rows"]
    if not rows:
        break
    for item in rows:
        row = item["row"]
        if row["filename"].replace("\\", "/").split("/")[0] != "noise_only":
            continue
        noise_rows_seen += 1
        if noise_rows_seen <= ALREADY_HAVE:      # we already have these ones
            continue

        y, sr = sf.read(io.BytesIO(get(row["audio"][0]["src"]).content))
        if y.ndim > 1:
            y = y.mean(axis=1)
        sf.write(OUT_DIR / f"{saved}.wav", y, sr)
        saved += 1
        print("extra noise", saved, flush=True)
        if saved >= EXTRA_WANTED:
            break
    offset += PAGE

print("Saved", saved, "extra noise clips to", OUT_DIR)