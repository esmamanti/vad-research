"""Loading the audio, labelling every frame, and mixing speech with noise.

Folder layout expected (created by scripts/download_data.py):
    data/train/clean_speech/*.wav    speech recorded without background noise
    data/train/noise_only/*.wav      background noise without the target speech
    data/test/...                    same, kept apart for testing

Every 25 ms frame gets one of three classes:
    silence : nothing is playing          (pauses inside the clean speech clips)
    noise   : sound, but no speech        (noise clips, and pauses after noise was mixed in)
    speech  : somebody is speaking        (clean, or with noise mixed in)

From these three classes we build the two questions of the project:
    audio present?   silence -> 0, noise and speech -> 1
    speech present?  silence and noise -> 0, speech -> 1
"""
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from features import SR, split_into_frames, energy_db, extract_features

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SNR_LEVELS = [20, 10, 0]            # speech is 20, 10 and 0 dB louder than the noise
SPEECH_DROP_DB = 40                 # speech must be within 40 dB of the loudest frame of the clip
FLOOR_MARGIN_DB = 8                 # and at least 8 dB above the background level of the clip
LABEL_SMOOTHING = 11                # labels are smoothed over 11 frames (about 0.1 s)


def load_wav(path):
    """Read a wav file as a mono float signal at 16 kHz."""
    y, sr = sf.read(path)
    if y.ndim > 1:                  # stereo -> mono
        y = y.mean(axis=1)
    if sr != SR:
        # The file has another sample rate (for example 44100 Hz), so we convert it to 16000 Hz.
        # resample_poly first removes the frequencies above 8000 Hz, then keeps fewer samples per second.
        divisor = gcd(sr, SR)
        y = resample_poly(y, SR // divisor, sr // divisor)
    return y


def list_clips(split, kind):
    """Sorted list of wav files, e.g. list_clips('train', 'noise_only')."""
    files = sorted((DATA_DIR / split / kind).glob("*.wav"))
    assert files, f"No wav files in {DATA_DIR / split / kind}. Run scripts/download_data.py first."
    return files


def label_threshold(frame_db):
    """Energy (dB) above which a frame of a CLEAN recording counts as speech.

    Two conditions, and the stricter one wins:
      1) not more than SPEECH_DROP_DB below the loudest frame (speech has a limited loudness range)
      2) at least FLOOR_MARGIN_DB above the background level of this recording.
         The background level is the 10th percentile: we assume the quietest 10% of the frames are pauses.
    """
    background = np.percentile(frame_db, 10)
    return max(frame_db.max() - SPEECH_DROP_DB, background + FLOOR_MARGIN_DB)


def speech_labels(clean):
    """True/False per frame: is somebody speaking in this frame of a CLEAN recording?

    In a clean recording the only sound is the speaker, so loudness is enough to decide.
    We always take the labels from the clean signal, BEFORE noise is added.
    """
    frame_db = energy_db(split_into_frames(clean))
    loud = frame_db > label_threshold(frame_db)
    # Majority vote: a frame is speech if most of the 11 frames around it are loud.
    # This removes one-frame flickers and keeps the very short gaps inside a word as speech.
    votes = pd.Series(loud.astype(float)).rolling(LABEL_SMOOTHING, center=True, min_periods=1).mean()
    return (votes > 0.5).values


def mix(clean, noise, snr_db):
    """Add noise to clean speech so that speech is snr_db decibels louder than the noise.

    SNR = 10 * log10(speech power / noise power). The speech power is measured only on the
    frames where somebody is speaking, so long pauses do not change the result.
    """
    noise = np.resize(noise, len(clean))                 # repeat or cut the noise to the same length
    speech_frames = split_into_frames(clean)[speech_labels(clean)]
    speech_power = np.mean(speech_frames ** 2)
    noise_power = np.mean(noise ** 2)
    wanted_noise_power = speech_power / 10 ** (snr_db / 10)
    return clean + noise * np.sqrt(wanted_noise_power / noise_power)


def build_frame_table(split, max_clips=None):
    """Feature table for one split ('train' or 'test'): one row per frame.

    Extra columns: 'klass' (silence / noise / speech), 'condition' (which kind of audio the
    frame came from) and 'clip' (file name).
    """
    clean_files = list_clips(split, "clean_speech")[:max_clips]
    noise_files = list_clips(split, "noise_only")[:max_clips]
    tables = []

    def add(signal, klass, condition, clip):
        table = extract_features(signal)
        table["klass"], table["condition"], table["clip"] = klass, condition, clip
        tables.append(table)

    # 1) Noise clips: every frame is "sound but no speech"
    for path in noise_files:
        add(load_wav(path), "noise", "noise only", path.name)

    # 2) Clean speech clips, alone and mixed with one noise clip at each SNR
    for i, path in enumerate(clean_files):
        clean = load_wav(path)
        is_speech = speech_labels(clean)
        add(clean, np.where(is_speech, "speech", "silence"), "clean speech", path.name)

        noise = load_wav(noise_files[i % len(noise_files)])
        for snr_db in SNR_LEVELS:
            # After mixing, the pauses are no longer silent: they contain noise
            add(mix(clean, noise, snr_db), np.where(is_speech, "speech", "noise"),
                f"speech + noise {snr_db} dB", path.name)

    table = pd.concat(tables, ignore_index=True)
    table["audio"] = (table["klass"] != "silence").astype(int)     # question 1: is there any sound?
    table["speech"] = (table["klass"] == "speech").astype(int)     # question 2: is there speech?
    return table