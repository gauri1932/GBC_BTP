# ============================================================
# GBC DATASET CONFIG (single source of truth)
# ============================================================
#
# Every script and model imports paths/constants from here,
# so Rosa change propagates everywhere (new machine, new
# target labels, new split, ...).
#
# ============================================================

from pathlib import Path


# Repo root = folder containing gbc/, models/, scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Raw + derived datasets (large data dirs, git-ignored).
DATASET_DIR = REPO_ROOT / "GBC_dataset"

# Native acquisition resolution: 512x512 grayscale INP,
# 512x512 multiclass GT (values 0..8).
ORIGINAL_SIZE = (512, 512)

# Splits with INP / GT / LABEL triplets.
# Train/Val = patient-level split (scripts/make_split.py),
# Test = untouched official test set.
SPLIT_DIRS = {
    "Train": DATASET_DIR / "split" / "Train",
    "Val": DATASET_DIR / "split" / "Val",
    "Test": DATASET_DIR / "Test",
}

# Classification attributes: LABEL files hold 9 bits,
# bit 0 is always 1, bits 1..8 are the real attributes.
N_LABEL_BITS = 9
CLASS_BITS = list(range(1, 9))

# Segmentation target: gallbladder mass + wall thickening.
TARGET_LABELS = {1, 2}

LABEL_NAMES = {
    0: "Background",
    1: "Gallbladder Mass",
    2: "Gallbladder Wall Thickening",
    3: "Label 3",
    4: "Label 4",
    5: "Label 5",
    6: "Label 6",
    7: "Label 7",
    8: "Label 8",
}
