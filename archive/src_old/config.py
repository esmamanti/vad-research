"""Shared settings for all notebooks. Change a value here and every notebook uses it."""

SEED = 42

# --- Data source ------------------------------------------------------------
HF_DATASET = "Aynursusuz/noisy-speech-dataset"

# Fill this in to read from a local folder instead of Hugging Face (otherwise leave as None).
# The folder must contain clean_train/ noise_train/ clean_test/ noise_test/ subfolders.
LOCAL_CORPUS_DIR = None

# How many files to use. Small for a first try; set to 200 / 200 / 80 / 80 for the final run.
N_TRAIN_CLEAN = 40
N_TRAIN_NOISE = 40
N_TEST_CLEAN = 20
N_TEST_NOISE = 20
DEV_FRACTION = 0.25          # share of the training files held out for tuning (development set)

# --- Signal processing ------------------------------------------------------
FRAME_MS = 30
HOP_MS = 10
LABEL_DROP_DB = 35           # relative threshold used to derive labels from clean speech
PAD_SECONDS = (1.0, 3.0)     # range of the gap added before and after each utterance (seconds)

# --- Models -----------------------------------------------------------------
MAX_TRAIN_FRAMES = 200_000   # maximum number of frames used for training (random subsample)
TARGET_FAR = 0.05            # operating point: 5% false alarms on the development set

RESULTS_DIR = "../results"