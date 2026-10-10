"""
vad_core.py - shared signal-level functions for the VAD experiments.

Usage from a notebook:
    import sys; sys.path.append("../src")
    import vad_core as vc
"""

import io

import librosa
import numpy as np
import soundfile as sf
from scipy.fft import dct

SR = 16000
EPS = 1e-10
MIN_DB = -100.0   # keeps digital silence from producing extreme values such as -200 dB


# ---------------------------------------------------------------------------
# 1. Reading audio and dataset helpers
# ---------------------------------------------------------------------------

def read_audio(audio_field, target_sr=SR):
    """Read a Hugging Face 'audio' field (decode=False) or a file path as mono audio at target_sr."""
    if isinstance(audio_field, dict):
        if audio_field.get("bytes") is not None:
            y, sr = sf.read(io.BytesIO(audio_field["bytes"]))
        else:
            y, sr = sf.read(audio_field["path"])
    else:
        y, sr = sf.read(audio_field)

    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float64)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr


def parse_filename(filename):
    """Split a dataset file name into (category, id): clean_speech/82.wav -> ('clean_speech', '82').

    The dataset's 'label' column does not separate noise_only from noisy_speech,
    so the category is taken from the file name instead.
    """
    parts = filename.replace("\\", "/").split("/")
    category = parts[0]
    file_id = parts[-1].rsplit(".", 1)[0]
    return category, file_id


# ---------------------------------------------------------------------------
# 2. Framing and feature extraction
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "energy_abs": ["rms_db"],
    "energy_rel": ["rms_rel"],
    "zcr": ["zcr"],
    "spectral": ["centroid", "rolloff", "flatness", "flux"],
    "mfcc": [f"mfcc{i}" for i in range(1, 13)],
}
FEATURE_NAMES = [n for group in FEATURE_GROUPS.values() for n in group]


def noise_floor_db(rms_db, range_db=60.0):
    """Noise floor of a recording in dB: the 10th percentile of the frame energy.

    The energy is first limited to at most range_db below the loud level (95th percentile),
    so digital silence cannot distort the floor (finding of EXP-04).
    """
    clipped = np.maximum(rms_db, np.percentile(rms_db, 95) - range_db)
    return np.percentile(clipped, 10), clipped


def frame_params(sr, frame_ms, hop_ms):
    return int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)


def frame_times(n_frames, sr, frame_ms, hop_ms):
    """Centre time of each frame in seconds."""
    frame_length, hop_length = frame_params(sr, frame_ms, hop_ms)
    return (np.arange(n_frames) * hop_length + frame_length / 2) / sr


def extract_features(y, sr=SR, frame_ms=30, hop_ms=10, n_mels=40):
    """Split the signal into frames and return a (n_frames, 19) feature matrix.

    rms_db  : absolute energy. Depends on the recording level (microphone, distance).
    rms_rel : energy relative to the recording's own noise floor. Independent of the level.

    All features are computed from the same frames, so the frame counts always match.
    The column order follows FEATURE_NAMES. MFCC 0 is dropped on purpose because it
    carries almost the same information as the energy.
    """
    frame_length, hop_length = frame_params(sr, frame_ms, hop_ms)
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frames = frames.T  # (n_frames, frame_length)

    # Time domain
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = np.maximum(20 * np.log10(rms + EPS), MIN_DB)
    floor, clipped = noise_floor_db(rms_db)
    rms_rel = clipped - floor
    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)

    # Frequency domain
    n_fft = int(2 ** np.ceil(np.log2(frame_length)))
    window = np.hanning(frame_length)
    mag = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=1))
    power = mag ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)

    mag_sum = mag.sum(axis=1) + EPS
    centroid = (mag * freqs).sum(axis=1) / mag_sum

    cumulative = np.cumsum(mag, axis=1)
    rolloff_idx = np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)
    rolloff = freqs[rolloff_idx]

    flatness = np.exp(np.mean(np.log(power + EPS), axis=1)) / (np.mean(power, axis=1) + EPS)

    norm_mag = mag / mag_sum[:, None]
    flux = np.zeros(len(frames))
    flux[1:] = np.sqrt(np.sum(np.diff(norm_mag, axis=0) ** 2, axis=1))

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    log_mel = np.log(power @ mel_basis.T + EPS)
    mfcc = dct(log_mel, type=2, norm="ortho", axis=1)[:, 1:13]

    features = np.column_stack([rms_db, rms_rel, zcr, centroid, rolloff, flatness, flux, mfcc])
    return features


def select_features(X, groups):
    """Keep only the columns of the given feature groups (used for the ablation experiments)."""
    names = [n for g in groups for n in FEATURE_GROUPS[g]]
    idx = [FEATURE_NAMES.index(n) for n in names]
    return X[:, idx], names


# ---------------------------------------------------------------------------
# 3. Labels
# ---------------------------------------------------------------------------

def energy_labels(y_clean, sr=SR, frame_ms=30, hop_ms=10, drop_db=35,
                  fill_gap_ms=200, min_speech_ms=50):
    """Derive frame labels (1 = speech) from CLEAN speech.

    The threshold is relative, not absolute: drop_db below the loud part of the
    recording (95th percentile), so it does not depend on the recording level.
    Gaps shorter than fill_gap_ms are then filled (short pauses between words count
    as speech) and very short speech segments are removed.
    """
    frame_length, hop_length = frame_params(sr, frame_ms, hop_ms)
    if len(y_clean) < frame_length:
        y_clean = np.pad(y_clean, (0, frame_length - len(y_clean)))
    frames = librosa.util.frame(y_clean, frame_length=frame_length, hop_length=hop_length).T
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + EPS)

    reference = np.percentile(rms_db, 95)
    labels = (rms_db > reference - drop_db).astype(int)

    labels = fill_short(labels, value=0, max_len=int(fill_gap_ms / hop_ms))
    labels = fill_short(labels, value=1, max_len=int(min_speech_ms / hop_ms))
    return labels


# ---------------------------------------------------------------------------
# 4. Post-processing of decisions
# ---------------------------------------------------------------------------

def segments(binary):
    """(start, end) indices of consecutive runs of 1; end is exclusive."""
    binary = np.asarray(binary).astype(int)
    padded = np.concatenate([[0], binary, [0]])
    change = np.diff(padded)
    return list(zip(np.where(change == 1)[0], np.where(change == -1)[0]))


def fill_short(binary, value, max_len):
    """Flip runs of 'value' that are no longer than max_len.

    value=0 -> fills short gaps inside speech.
    value=1 -> removes very short speech segments.
    Runs at the very start or end of the recording are left untouched.
    """
    out = np.asarray(binary).astype(int).copy()
    if max_len <= 0:
        return out
    target = (out == value).astype(int)
    for start, end in segments(target):
        if start == 0 or end == len(out):
            continue
        if end - start <= max_len:
            out[start:end] = 1 - value
    return out


def extend(binary, before=0, after=0):
    """Extend every speech segment by 'before' frames backwards and 'after' frames forwards.

    after  -> hangover (recovers word endings)
    before -> look-back buffer (recovers word onsets; means a delay in real time)
    """
    binary = np.asarray(binary).astype(int)
    out = binary.copy()
    for start, end in segments(binary):
        out[max(0, start - before):min(len(out), end + after)] = 1
    return out


def smooth_scores(score, window=1):
    """Moving average of the scores over 'window' frames. window=1 leaves them unchanged."""
    if window <= 1:
        return np.asarray(score, dtype=float)
    kernel = np.ones(window) / window
    padded = np.pad(score, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def add_context(X, k):
    """Append the features of the k previous and k following frames to every frame.

    Output shape: (n_frames, n_features * (2k + 1)). Edges are padded by repetition.
    Must be applied per recording, before recordings are concatenated.
    """
    if k <= 0:
        return X
    padded = np.pad(X, ((k, k), (0, 0)), mode="edge")
    return np.hstack([padded[i:i + len(X)] for i in range(2 * k + 1)])


# ---------------------------------------------------------------------------
# 5. Threshold-based VAD
# ---------------------------------------------------------------------------

def robust_threshold_vad(rms_db, k=1.5, cap_db=6.0, range_db=60.0):
    """Robust threshold from EXP-04: the margin follows the fluctuation of the noise floor.

    margin = k * (20th percentile - 5th percentile), limited to between 1 dB and cap_db.
    """
    _, clipped = noise_floor_db(rms_db, range_db)
    p5, p10, p20 = np.percentile(clipped, [5, 10, 20])
    threshold = p10 + min(max(k * (p20 - p5), 1.0), cap_db)
    return (rms_db > threshold).astype(int), threshold


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------

def vad_metrics(y_true, y_pred):
    """Missed speech and false alarms are reported separately.

    miss_rate        : share of speech that was cut
    false_alarm_rate : share of silence/noise that was called speech
    balanced_acc     : mean of the accuracies of the two classes
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    miss = fn / (tp + fn) if (tp + fn) else np.nan
    false_alarm = fp / (fp + tn) if (fp + tn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = 1 - miss if not np.isnan(miss) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else np.nan

    return {
        "miss_rate": miss,
        "false_alarm_rate": false_alarm,
        "balanced_acc": (1 - (miss + false_alarm) / 2
                         if not np.isnan(miss) and not np.isnan(false_alarm) else np.nan),
        "precision": precision,
        "f1": f1,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def threshold_for_far(y_true, score, far=0.05):
    """Score threshold that gives a false alarm rate of 'far'. Chosen on the DEVELOPMENT set."""
    negatives = np.asarray(score)[np.asarray(y_true) == 0]
    return float(np.quantile(negatives, 1 - far))


def det_points(y_true, score, n_points=200):
    """Points of the (false alarm rate, miss rate) trade-off curve."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    thresholds = np.quantile(score, np.linspace(0, 1, n_points))
    pos, neg = score[y_true == 1], score[y_true == 0]
    far = np.array([(neg > t).mean() for t in thresholds])
    miss = np.array([(pos <= t).mean() for t in thresholds])
    return far, miss


# ---------------------------------------------------------------------------
# 7. Adding noise and simulating environments
# ---------------------------------------------------------------------------

def labels_to_sample_mask(labels, n_samples, sr=SR, frame_ms=30, hop_ms=10):
    """Turn frame labels into a sample-level mask."""
    frame_length, hop_length = frame_params(sr, frame_ms, hop_ms)
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in segments(labels):
        mask[start * hop_length:(end - 1) * hop_length + frame_length] = True
    return mask


def fit_length(x, n, rng):
    """Bring a signal to n samples: repeat it if too short, take a random part if too long."""
    if len(x) < n:
        x = np.tile(x, int(np.ceil(n / len(x))))
    start = rng.integers(0, len(x) - n + 1)
    return x[start:start + n]


def mix_at_snr(clean, noise, snr_db, sample_mask=None, rng=None):
    """Add noise to clean speech at the requested SNR.

    The speech power is measured only on samples where speech is present (sample_mask).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    noise = fit_length(noise, len(clean), rng)
    speech = clean[sample_mask] if sample_mask is not None and sample_mask.any() else clean
    speech_power = np.mean(speech ** 2) + EPS
    noise_power = np.mean(noise ** 2) + EPS
    scale = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mixed = clean + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * 0.99 / peak
    return mixed


def white_noise(n, rng):
    return rng.normal(size=n)


def pink_noise(n, rng):
    """1/f noise: stronger at low frequencies, closer to real ambient hum."""
    spectrum = np.fft.rfft(rng.normal(size=n))
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    return np.fft.irfft(spectrum / np.sqrt(freqs), n=n)


def babble_noise(signals, n, rng, n_speakers=6):
    """Build 'crowd' noise by overlaying several speech recordings."""
    picks = rng.choice(len(signals), size=min(n_speakers, len(signals)), replace=False)
    total = np.zeros(n)
    for i in picks:
        s = fit_length(signals[i], n, rng)
        total += s / (np.sqrt(np.mean(s ** 2)) + EPS)
    return total


def apply_gain(y, gain_db):
    """Imitate a different recording level (distance to the microphone)."""
    return y * 10 ** (gain_db / 20)


def add_reverb(y, rt60=0.6, sr=SR, seed=0):
    """Artificial room reverberation (RT60 in seconds), using an exponentially decaying noise tail."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    tail = rng.normal(size=n) * np.exp(-6.9 * t / rt60)
    tail[0] = 0
    tail = 0.5 * tail / np.sqrt(np.sum(tail ** 2))
    tail[0] = 1.0
    out = fftconvolve(y, tail)[:len(y)]
    return out * np.sqrt((np.mean(y ** 2) + EPS) / (np.mean(out ** 2) + EPS))


def telephone_band(y, sr=SR):
    """Telephone line / low-quality microphone: 300-3400 Hz band-pass filter."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


# ---------------------------------------------------------------------------
# 8. Comparison with an off-the-shelf VAD
# ---------------------------------------------------------------------------

def webrtc_vad(y, n_frames, mode=2, sr=SR, frame_ms=30, hop_ms=10):
    """Map WebRTC VAD decisions onto our frame grid. mode: 0 (lenient) to 3 (strict).

    Install with: pip install webrtcvad-wheels
    """
    import webrtcvad
    vad = webrtcvad.Vad(mode)
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    step = int(sr * 0.03)
    decisions = [
        vad.is_speech(pcm[i:i + step].tobytes(), sr)
        for i in range(0, len(pcm) - step + 1, step)
    ]
    decisions = np.array(decisions, dtype=int)
    times = frame_times(n_frames, sr, frame_ms, hop_ms)
    idx = np.minimum((times * sr // step).astype(int), len(decisions) - 1)
    return decisions[idx]
