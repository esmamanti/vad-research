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

# Extra time-domain features (notebook 05). Time domain = computed on the waveform samples, no spectrum.
EXTRA_TIME_FEATURES = ["magnitude_db", "teager_db", "energy_entropy", "amdf_periodicity", "pitch_hz"]
TIME_FEATURES = ["energy_db", "magnitude_db", "teager_db", "zcr", "energy_entropy",
                 "harmonicity", "amdf_periodicity", "pitch_hz"]
TIME_CONTEXT_FEATURES = [f"{name}_{stat}" for name in TIME_FEATURES for stat in ("mean", "std")]


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


def magnitude_db(frames):
    """Short-time average magnitude: mean of the ABSOLUTE sample values, in dB.
    Like energy, but without squaring, so a few very large samples count less."""
    return 20 * np.log10(np.mean(np.abs(frames), axis=1) + EPS)


def teager_db(frames):
    """Teager energy: mean of x[n]^2 - x[n-1] * x[n+1], in dB.
    It grows with loudness AND with frequency, and reacts quickly to changes."""
    teager = frames[:, 1:-1] ** 2 - frames[:, :-2] * frames[:, 2:]
    return 10 * np.log10(np.abs(teager.mean(axis=1)) + EPS)


def energy_entropy(frames, n_blocks=10):
    """How evenly the energy is spread over 10 short blocks inside the frame (0 to 1).
    1 = spread evenly (steady sound), lower = energy concentrated in a few blocks (sudden changes)."""
    blocks = frames[:, :FRAME // n_blocks * n_blocks].reshape(len(frames), n_blocks, -1)
    block_energy = (blocks ** 2).sum(axis=2) + EPS
    share = block_energy / block_energy.sum(axis=1, keepdims=True)
    return -(share * np.log2(share)).sum(axis=1) / np.log2(n_blocks)


def amdf_periodicity(frames):
    """Pitch strength from the AMDF (average magnitude difference function), 0 to 1.
    AMDF(k) = mean |x[n] - x[n-k]|. If the wave repeats every k samples, AMDF(k) drops close to 0.
    We return 1 - (deepest dip / average level) over shifts of 40..200 samples (400..80 Hz)."""
    shortest, longest = SR // 400, SR // 80
    amdf = np.stack([np.mean(np.abs(frames[:, k:] - frames[:, :-k]), axis=1)
                     for k in range(shortest, longest + 1)], axis=1)
    return 1 - amdf.min(axis=1) / (amdf.mean(axis=1) + EPS)


def pitch_hz(frames):
    """Pitch estimate in Hz: the shift (80..400 Hz range) where the autocorrelation is highest.
    Only meaningful in voiced speech. In silence and noise the value means nothing:
    it often sticks to the edge of the range (400 Hz), because the autocorrelation simply falls with the shift."""
    centred = frames - frames.mean(axis=1, keepdims=True)
    autocorr = np.fft.irfft(np.abs(np.fft.rfft(centred, n=2 * N_FFT, axis=1)) ** 2, axis=1)
    shortest, longest = SR // 400, SR // 80
    best_shift = shortest + autocorr[:, shortest:longest + 1].argmax(axis=1)
    return SR / best_shift


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

    # Extra time-domain features (notebook 05)
    table["magnitude_db"] = magnitude_db(frames)
    table["teager_db"] = teager_db(frames)
    table["energy_entropy"] = energy_entropy(frames)
    table["amdf_periodicity"] = amdf_periodicity(frames)
    table["pitch_hz"] = pitch_hz(frames)

    # Context features: a single 25 ms frame is very short, so we also describe its neighbourhood.
    # For every scalar and time-domain feature we add its average (mean) and how much it varies (std)
    # over the surrounding 31 frames (about 0.3 s).
    for name in SCALAR_FEATURES + EXTRA_TIME_FEATURES:
        window = table[name].rolling(2 * CONTEXT + 1, center=True, min_periods=1)
        table[f"{name}_mean"] = window.mean()
        table[f"{name}_std"] = window.std().fillna(0)
    return table