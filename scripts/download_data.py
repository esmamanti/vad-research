"""Download a small part of the Hugging Face dataset and save it as ordinary wav files.

Run once from the project folder:
    python scripts/download_data.py

Result (you can open and play these files with any audio player):
    data/train/clean_speech/0.wav ...      data/test/clean_speech/0.wav ...
    data/train/noise_only/0.wav ...        data/test/noise_only/0.wav ...
    data/train/noisy_speech/0.wav ...      data/test/noisy_speech/0.wav ...   (only for listening)
"""
import io
from collections import Counter
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

DATASET = "Aynursusuz/noisy-speech-dataset"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# How many clips to save from each folder of the dataset, per split.
# The dataset has about 3400 clips of 15-20 s; a small part is enough for this project.
CLIPS_WANTED = {
    "train": {"clean_speech": 30, "noise_only": 30, "noisy_speech": 5},
    "test": {"clean_speech": 20, "noise_only": 20, "noisy_speech": 5},
}

for split, wanted in CLIPS_WANTED.items():
    print("Connecting to the", split, "split...", flush=True)

    # streaming=True reads the clips one by one instead of downloading the whole dataset first.
    # decode=False gives us the raw wav bytes, which we read ourselves with soundfile.
    dataset = load_dataset(DATASET, split=split, streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    saved = Counter()
    for row in dataset:
        # The 'filename' column looks like "clean_speech\82.wav": folder name, then file name
        folder = row["filename"].replace("\\", "/").split("/")[0]
        if folder not in wanted or saved[folder] >= wanted[folder]:
            continue

        audio = row["audio"]
        source = io.BytesIO(audio["bytes"]) if audio["bytes"] is not None else audio["path"]
        y, sr = sf.read(source)
        if y.ndim > 1:                 # stereo -> mono
            y = y.mean(axis=1)

        out_dir = DATA_DIR / split / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / f"{saved[folder]}.wav", y, sr)
        saved[folder] += 1
        print(split, folder, saved[folder], flush=True)     # progress: one line per saved clip

        if all(saved[name] >= count for name, count in wanted.items()):
            break

    print(split, dict(saved))

print("Saved to", DATA_DIR)