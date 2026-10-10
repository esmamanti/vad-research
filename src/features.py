"""Frame-level audio features, written with plain NumPy so every line can be explained.

The audio is cut into short frames (25 ms). Each function below takes all frames at once
and returns ONE number per frame (MFCC returns 12 numbers per frame).
extract_features() at the bottom puts everything into one table.
"""
import numpy as np
import pandas as pd

SR = 16000      # sample rate: 16000 samples per second
FRAME = 400     # frame length: 400 samples = 25 ms (speech is roughly stable over 20-30 ms)
HOP = 160       # step between frames: 160 samples = 10 ms (so neighbouring frames overlap)
N_FFT = 512     # FFT size: frame is zero-padded from 400 to 512 samples (FRAME must not be larger than this)

FREQS = np.fft.rfftfreq(N_FFT, d=1 / SR)    # centre frequency (Hz) of each of the 257 FFT bins
EPS = 1e-10                                 # tiny number that prevents log(0) and division by 0
assert FRAME <= N_FFT, "FRAME must not be larger than N_FFT"

CONTEXT = 15    # context features look 15 frames (150 ms) to the left and to the right

SCALAR_FEATURES = ["energy_db", "zcr", "centroid", "flatness", "flux", "band_ratio", "harmonicity"]
MFCC_FEATURES = [f"mfcc{i}" for i in range(1, 13)]
CONTEXT_FEATURES = [f"{name}_{stat}" for name in SCALAR_FEATURES for stat in ("mean", "std")]


def split_into_frames(y):
    """Cut a 1-D signal into overlapping frames. Result shape: (number of frames, FRAME)."""
    n_frames = 1 + (len(y) - FRAME) // HOP
    starts = HOP * np.arange(n_frames)                    # first sample of each frame
    index = starts[:, None] + np.arange(FRAME)[None, :]   # sample indices of every frame
    return y[index]


def energy_db(frames):
    """Loudness of the frame: mean of the squared samples, on a decibel scale.
    Silence is very low (about -70 dB or less), loud sound is close to 0 dB."""
    power = np.mean(frames ** 2, axis=1)
    return 10 * np.log10(power + EPS)


def zero_crossing_rate(frames):
    """Share of neighbouring sample pairs where the waveform changes sign (0 to 1).
    Low for voiced speech (slow, low-frequency wave), high for hiss-like sounds ("s", white noise)."""
    negative = frames < 0
    return np.mean(negative[:, 1:] != negative[:, :-1], axis=1)


def magnitude_spectrum(frames):
    """How much of each frequency is in the frame. Result shape: (number of frames, 257).
    The Hann window fades the frame edges to zero so the cut does not create fake frequencies."""
    window = np.hanning(FRAME)
    return np.abs(np.fft.rfft(frames * window, n=N_FFT, axis=1))


def spectral_centroid(mag):
    """'Centre of mass' of the spectrum in Hz: the average frequency, weighted by magnitude.
    Low when the energy sits in low frequencies (vowels), high for bright/hissy sounds."""
    return (mag * FREQS).sum(axis=1) / (mag.sum(axis=1) + EPS)


def spectral_flatness(mag):
    """Geometric mean / arithmetic mean of the power spectrum (0 to 1).
    Close to 1 = flat spectrum = noise-like. Close to 0 = a few strong peaks = tonal (voiced speech)."""
    power = mag ** 2 + EPS
    geometric_mean = np.exp(np.mean(np.log(power), axis=1))
    arithmetic_mean = np.mean(power, axis=1)
    return geometric_mean / arithmetic_mean


def spectral_flux(mag):
    """How much the SHAPE of the spectrum changed since the previous frame.
    Speech changes quickly (sound after sound), steady noise (fan, air conditioner) hardly changes."""
    shape = mag / (mag.sum(axis=1, keepdims=True) + EPS)      # normalise so loudness does not matter
    flux = np.zeros(len(mag))
    flux[1:] = np.sqrt(np.sum(np.diff(shape, axis=0) ** 2, axis=1))
    return flux


def speech_band_ratio(mag):
    """Share of the frame's power that lies between 300 and 3400 Hz (the telephone speech band)."""
    power = mag ** 2
    in_band = (FREQS >= 300) & (FREQS <= 3400)
    return power[:, in_band].sum(axis=1) / (power.sum(axis=1) + EPS)


def harmonicity(frames):
    """Strength of the pitch: how well the frame matches a shifted copy of itself (0 to 1).
    Voiced speech repeats every pitch period (vocal cords vibrate at 80-400 Hz), so the match is strong.
    Noise does not repeat, so the match is weak."""
    centred = frames - frames.mean(axis=1, keepdims=True)
    # Autocorrelation for all shifts at once (computed through the FFT, which is just a fast way to do it)
    autocorr = np.fft.irfft(np.abs(np.fft.rfft(centred, n=2 * N_FFT, axis=1)) ** 2, axis=1)
    shortest, longest = SR // 400, SR // 80      # shifts of 40..200 samples = pitch of 400..80 Hz
    return autocorr[:, shortest:longest + 1].max(axis=1) / (autocorr[:, 0] + EPS)


def mel_filterbank(n_filters=26):
    """Triangular filters spaced on the mel scale (dense at low frequencies, like human hearing)."""
    to_mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    to_hz = lambda mel: 700 * (10 ** (mel / 2595) - 1)
    edges_hz = to_hz(np.linspace(to_mel(0), to_mel(SR / 2), n_filters + 2))
    bank = np.zeros((n_filters, len(FREQS)))
    for i in range(n_filters):
        left, centre, right = edges_hz[i], edges_hz[i + 1], edges_hz[i + 2]
        rising = (FREQS - left) / (centre - left)
        falling = (right - FREQS) / (right - centre)
        bank[i] = np.clip(np.minimum(rising, falling), 0, None)
    return bank


MEL_BANK = mel_filterbank()


def mfcc(mag):
    """12 numbers that describe the overall SHAPE of the spectrum (the spectral envelope).
    Steps: power spectrum -> 26 mel bands -> log -> DCT. Coefficient 0 is dropped because it is just loudness."""
    log_mel = np.log((mag ** 2) @ MEL_BANK.T + EPS)                     # (frames, 26)
    n = log_mel.shape[1]
    k = np.arange(13)[:, None]
    dct_basis = np.cos(np.pi * k * (2 * np.arange(n)[None, :] + 1) / (2 * n))   # DCT-II
    return (log_mel @ dct_basis.T)[:, 1:13]


def extract_features(y):
    """Return a table with one row per frame and one column per feature."""
    frames = split_into_frames(y)
    mag = magnitude_spectrum(frames)
    table = pd.DataFrame({
        "energy_db": energy_db(frames),
        "zcr": zero_crossing_rate(frames),
        "centroid": spectral_centroid(mag),
        "flatness": spectral_flatness(mag),
        "flux": spectral_flux(mag),
        "band_ratio": speech_band_ratio(mag),
        "harmonicity": harmonicity(frames),
    })
    table[MFCC_FEATURES] = mfcc(mag)

    # Context features: a single 25 ms frame is very short, so we also describe its neighbourhood.
    # For every scalar feature we add its average (mean) and how much it varies (std)
    # over the surrounding 31 frames (about 0.3 s).
    for name in SCALAR_FEATURES:
        window = table[name].rolling(2 * CONTEXT + 1, center=True, min_periods=1)
        table[f"{name}_mean"] = window.mean()
        table[f"{name}_std"] = window.std().fillna(0)
    return table