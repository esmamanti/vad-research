"""
vad_data.py - loading data, building listening conditions and evaluating models.

A 'record' is a dictionary:
    {"id", "X" (n_frames, 19), "y" (n_frames,), "rms_db", "audio" (optional)}
"""

import glob
import os

import numpy as np

import config as cfg
import vad_core as vc


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_folder(folder, limit=None, seed=cfg.SEED):
    """Read the wav files of a folder."""
    files = sorted(glob.glob(os.path.join(folder, "*.wav")))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    items = []
    for path in files[:limit]:
        stem = os.path.splitext(os.path.basename(path))[0]
        y, _ = vc.read_audio(path)
        items.append({"id": stem, "y": y})
    return items


def load_hf_split(split, n_clean, n_noise, seed=cfg.SEED):
    """Take clean_speech and noise_only files from the dataset.

    The noisy_speech files are NOT used here: we create noisy speech ourselves by
    mixing at a known SNR. This gives exact frame labels and full control of the SNR.
    """
    from datasets import Audio, load_dataset
    ds = load_dataset(cfg.HF_DATASET, split=split)
    ds = ds.cast_column("audio", Audio(decode=False))

    names = ds["filename"]
    rng = np.random.default_rng(seed)
    result = {}
    for category, n in [("clean_speech", n_clean), ("noise_only", n_noise)]:
        idx = [i for i, name in enumerate(names) if vc.parse_filename(name)[0] == category]
        rng.shuffle(idx)
        items = []
        for i in idx[:n]:
            y, _ = vc.read_audio(ds[int(i)]["audio"])
            items.append({"id": f"{split}_{vc.parse_filename(names[i])[1]}", "y": y})
        result[category] = items
    return result["clean_speech"], result["noise_only"]


def load_corpus():
    """Lists of files: {'train_clean', 'train_noise', 'test_clean', 'test_noise'}."""
    if cfg.LOCAL_CORPUS_DIR:
        d = cfg.LOCAL_CORPUS_DIR
        return {
            "train_clean": load_folder(os.path.join(d, "clean_train"), cfg.N_TRAIN_CLEAN),
            "train_noise": load_folder(os.path.join(d, "noise_train"), cfg.N_TRAIN_NOISE),
            "test_clean": load_folder(os.path.join(d, "clean_test"), cfg.N_TEST_CLEAN),
            "test_noise": load_folder(os.path.join(d, "noise_test"), cfg.N_TEST_NOISE),
        }
    train_clean, train_noise = load_hf_split("train", cfg.N_TRAIN_CLEAN, cfg.N_TRAIN_NOISE)
    test_clean, test_noise = load_hf_split("test", cfg.N_TEST_CLEAN, cfg.N_TEST_NOISE)
    return {"train_clean": train_clean, "train_noise": train_noise,
            "test_clean": test_clean, "test_noise": test_noise}


def split_dev(items, fraction=cfg.DEV_FRACTION, seed=cfg.SEED):
    """Split files into training and development sets (frames of one file never end up on both sides)."""
    idx = np.random.default_rng(seed).permutation(len(items))
    n_dev = max(1, int(len(items) * fraction))
    dev = [items[i] for i in idx[:n_dev]]
    train = [items[i] for i in idx[n_dev:]]
    return train, dev


# ---------------------------------------------------------------------------
# Building listening conditions
# ---------------------------------------------------------------------------

def prepare_clean(items, frame_ms=cfg.FRAME_MS, hop_ms=cfg.HOP_MS, drop_db=cfg.LABEL_DROP_DB,
                  pad_seconds=cfg.PAD_SECONDS, seed=cfg.SEED):
    """Turn every clean file into a 'scene': [gap] speech [gap].

    A gap of random length is added before and after the utterance. The gap is not
    digital zero but 'room tone' at the file's own floor level. When noise is added
    later, the gaps are filled with noise as well.

    Adds: item["y"] (scene), item["labels"] (frame labels), item["mask"] (sample mask).
    """
    rng = np.random.default_rng(seed)
    _, hop = vc.frame_params(vc.SR, frame_ms, hop_ms)
    for item in items:
        raw = item.setdefault("raw", item["y"])
        core = vc.energy_labels(raw, frame_ms=frame_ms, hop_ms=hop_ms, drop_db=drop_db)

        lead, trail = (int(rng.uniform(*pad_seconds) * vc.SR) // hop * hop for _ in range(2))
        frame_rms = np.sqrt(np.mean(raw[:len(raw) // hop * hop].reshape(-1, hop) ** 2, axis=1))
        floor = max(np.percentile(frame_rms, 5), 1e-5)

        item["y"] = np.concatenate([rng.normal(size=lead) * floor, raw, rng.normal(size=trail) * floor])
        item["labels"] = np.concatenate([np.zeros(lead // hop, int), core, np.zeros(trail // hop, int)])
        item["mask"] = vc.labels_to_sample_mask(item["labels"], len(item["y"]), frame_ms=frame_ms, hop_ms=hop_ms)
    return items


def render(item, cond, noise_pool, rng, clean_pool=None):
    """Put a clean scene into the given condition and return the audio signal.

    Keys of cond (all optional):
      snr       : dB, or a (low, high) range -> noise is added
      noise     : "pool" (default, real ambient noise) | "white" | "pink" | "babble"
      reverb    : RT60 in seconds (applied to the speech BEFORE the noise)
      telephone : True -> 300-3400 Hz band
      gain      : dB (applied last)
    """
    y = item["y"]
    if cond.get("reverb"):
        y = vc.add_reverb(y, rt60=cond["reverb"], seed=int(rng.integers(1 << 30)))

    snr = cond.get("snr")
    if snr is not None:
        if isinstance(snr, (tuple, list)):
            snr = rng.uniform(snr[0], snr[1])
        kind = cond.get("noise", "pool")
        if kind == "white":
            noise = vc.white_noise(len(y), rng)
        elif kind == "pink":
            noise = vc.pink_noise(len(y), rng)
        elif kind == "babble":
            others = [c["y"] for c in clean_pool if c["id"] != item["id"]]
            noise = vc.babble_noise(others, len(y), rng)
        else:
            noise = noise_pool[rng.integers(len(noise_pool))]["y"]
        y = vc.mix_at_snr(y, noise, snr, item["mask"], rng)

    if cond.get("telephone"):
        y = vc.telephone_band(y)
    if cond.get("gain"):
        y = vc.apply_gain(y, cond["gain"])
    return y


def build_records(items, cond, noise_pool=None, seed=cfg.SEED, keep_audio=False,
                  frame_ms=cfg.FRAME_MS, hop_ms=cfg.HOP_MS):
    """Create a list of records from clean scenes in the given condition."""
    rng = np.random.default_rng(seed)
    records = []
    for item in items:
        y = render(item, cond, noise_pool, rng, clean_pool=items)
        X = vc.extract_features(y, frame_ms=frame_ms, hop_ms=hop_ms)
        record = {"id": item["id"], "X": X, "y": item["labels"], "rms_db": X[:, 0]}
        if keep_audio:
            record["audio"] = y
        records.append(record)
    return records


def noise_records(noise_items, keep_audio=False, frame_ms=cfg.FRAME_MS, hop_ms=cfg.HOP_MS, max_seconds=20):
    """Records that contain only noise: every frame is labelled 0. Used to measure false alarms."""
    records = []
    for item in noise_items:
        y = item["y"][:int(max_seconds * vc.SR)]
        X = vc.extract_features(y, frame_ms=frame_ms, hop_ms=hop_ms)
        record = {"id": item["id"], "X": X, "y": np.zeros(len(X), dtype=int), "rms_db": X[:, 0]}
        if keep_audio:
            record["audio"] = y
        records.append(record)
    return records


def stack(records, groups=None, context=0, max_frames=None, seed=cfg.SEED):
    """Concatenate records into one (X, y) pair. Feature selection and context are applied per record."""
    Xs, ys = [], []
    for r in records:
        X = r["X"] if groups is None else vc.select_features(r["X"], groups)[0]
        Xs.append(vc.add_context(X, context))
        ys.append(r["y"])
    X, y = np.vstack(Xs), np.concatenate(ys)
    if max_frames and len(y) > max_frames:
        idx = np.random.default_rng(seed).choice(len(y), max_frames, replace=False)
        X, y = X[idx], y[idx]
    return X, y


# ---------------------------------------------------------------------------
# Models and evaluation
# ---------------------------------------------------------------------------

def make_model(kind):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=20,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=cfg.SEED)
    raise ValueError(kind)


def fit(kind, records, groups=None, context=0):
    """Train a model and bundle it with its settings so it can be reused."""
    X, y = stack(records, groups, context, max_frames=cfg.MAX_TRAIN_FRAMES)
    model = make_model(kind).fit(X, y)
    return {"kind": kind, "model": model, "groups": groups, "context": context,
            "smooth": 1, "threshold": 0.5, "extend": (0, 0)}


def scores(spec, record):
    """Speech probability per frame for one record."""
    X = record["X"] if spec["groups"] is None else vc.select_features(record["X"], spec["groups"])[0]
    X = vc.add_context(X, spec["context"])
    return vc.smooth_scores(spec["model"].predict_proba(X)[:, 1], spec.get("smooth", 1))


def predict(spec, record):
    """0/1 decision per frame. spec is either a trained model bundle or a baseline setting."""
    if spec["kind"] == "robust_threshold":
        pred, _ = vc.robust_threshold_vad(record["rms_db"], spec.get("k", 1.5), spec.get("cap_db", 6.0))
    elif spec["kind"] == "webrtc":
        pred = vc.webrtc_vad(record["audio"], len(record["y"]), mode=spec["mode"])
    else:
        pred = (scores(spec, record) > spec["threshold"]).astype(int)
    before, after = spec.get("extend", (0, 0))
    return vc.extend(pred, before, after)


def pooled_scores(spec, records):
    y = np.concatenate([r["y"] for r in records])
    s = np.concatenate([scores(spec, r) for r in records])
    return y, s


def calibrate(spec, dev_records, far=cfg.TARGET_FAR):
    """Set the decision threshold on the DEVELOPMENT set for the target false alarm rate."""
    y, s = pooled_scores(spec, dev_records)
    spec["threshold"] = vc.threshold_for_far(y, s, far)
    return spec


def evaluate(spec, records):
    """Metrics computed over the pooled frames of all records."""
    y = np.concatenate([r["y"] for r in records])
    p = np.concatenate([predict(spec, r) for r in records])
    return vc.vad_metrics(y, p)


def auc(spec, records):
    from sklearn.metrics import roc_auc_score
    y, s = pooled_scores(spec, records)
    return roc_auc_score(y, s)


def miss_at_far(spec, records, far=cfg.TARGET_FAR):
    """Miss rate at a false alarm rate of 'far'. The threshold is chosen on the SAME records,
    so use this only to compare features or models on the development set, not as a final result."""
    y, s = pooled_scores(spec, records)
    threshold = vc.threshold_for_far(y, s, far)
    return float((s[y == 1] <= threshold).mean())