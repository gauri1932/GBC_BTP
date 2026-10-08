# ============================================================
# GBC PREPROCESSING LIBRARY (generic, model-agnostic)
# ============================================================
#
# Shared helpers for every downstream task
# (segmentation, classification, analysis).
#
# Rules for this file:
#   - NO MedSAM specifics (no 1024, no RGB, no boxes.json).
#   - Pure functions: every behavior comes from arguments,
#     so any pipeline can import and reuse ("recursive use").
#   - This module has NO side effects on import.
#
# ============================================================

import re
from pathlib import Path

import numpy as np
from PIL import Image

from gbc.config import (
    CLASS_BITS,
    DATASET_DIR,
    N_LABEL_BITS,
    ORIGINAL_SIZE,
    SPLIT_DIRS,
)


# Backwards-compatible aliases (old flat-module names).
ROOT = DATASET_DIR


# ============================================================
# FILENAMES / PATIENTS
# ============================================================

def get_patient_id(filename):
    """
    Extract patient ID from a slice filename.

    Example:
        "117_21.png" -> 117
    """
    match = re.match(r"^(\d+)_", Path(filename).name)

    if match:
        return int(match.group(1))

    return None


def list_files(folder, extension):
    """Sorted file names (not paths) with the given extension."""
    folder = Path(folder)

    if not folder.exists():
        return []

    return sorted(
        f.name
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == extension.lower()
    )


def stem(filename):
    """Filename without extension: '117_21.png' -> '117_21'."""
    return Path(filename).stem


def pairing_report(inp_names, gt_names, label_names):
    """
    Generic INP / GT / LABEL pairing check.

    Returns a dict with missing-pair counts (0 means clean).
    """
    inp = {stem(f) for f in inp_names}
    gt = {stem(f) for f in gt_names}
    label = {stem(f) for f in label_names}

    return {
        "n_inp": len(inp),
        "n_gt": len(gt),
        "n_label": len(label),
        "inp_without_gt": len(inp - gt),
        "inp_without_label": len(inp - label),
        "gt_without_inp": len(gt - inp),
        "label_without_inp": len(label - inp),
    }


def patient_level_split(patient_ids, val_ratio=0.20, seed=42):
    """
    Patient-level train/val split (no leakage by construction).

    Returns (train_ids, val_ids) as sorted lists.
    """
    import random

    ids = sorted(set(patient_ids))

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_val = round(len(ids) * val_ratio)

    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])

    assert not (set(train_ids) & set(val_ids)), "patient leakage!"

    return train_ids, val_ids


# ============================================================
# LOADERS (raw dataset -> numpy)
# ============================================================

def load_inp_gray(path):
    """Load INP slice as grayscale (512x512, uint8)."""
    return np.array(Image.open(path).convert("L"))


def load_gt(path):
    """Load GT mask as multiclass uint8 array (values 0..8)."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def load_label_vector(path):
    """
    Load LABEL txt file as int array of shape (9,).

    Example file content: "1 1 0 0 1 0 0 1 0"
    """
    with open(path, "r") as f:
        content = f.read().strip().split()

    return np.array([int(v) for v in content], dtype=np.int64)


def load_class_bits(path):
    """
    Load LABEL file keeping only the 8 informative bits (1..8).

    Returns int array of shape (8,). Bit 0 is dropped because
    it is always 1 in every split.
    """
    return load_label_vector(path)[CLASS_BITS]


# ============================================================
# GENERIC IMAGE / MASK TRANSFORMS
# ============================================================

def resize_image(img_array_or_pil, size):
    """
    Resize a grayscale/RGB image with BILINEAR interpolation.

    `size` is a (width, height) tuple, e.g. (256, 256).
    Accepts a numpy array or PIL image, returns a PIL image.
    """
    if isinstance(img_array_or_pil, np.ndarray):
        img = Image.fromarray(img_array_or_pil)
    else:
        img = img_array_or_pil

    return img.resize(size, Image.Resampling.BILINEAR)


def resize_mask(mask, size):
    """
    Resize a label/binary mask with NEAREST NEIGHBOR.

    Never use bilinear/bicubic for segmentation labels:
    it creates fractional class values.
    """
    pil_mask = Image.fromarray(np.asarray(mask).astype(np.uint8))

    resized = pil_mask.resize(size, Image.Resampling.NEAREST)

    return np.array(resized, dtype=np.uint8)


def to_rgb(img_array_or_pil):
    """Convert grayscale image to 3-channel RGB (PIL)."""
    if isinstance(img_array_or_pil, np.ndarray):
        img = Image.fromarray(img_array_or_pil)
    else:
        img = img_array_or_pil

    return img.convert("RGB")


# ============================================================
# GENERIC MASK LOGIC (any label subset -> binary, bbox)
# ============================================================

def binary_mask_from_labels(gt_array, target_labels):
    """
    Collapse a multiclass GT into a binary mask.

    Pixels whose value is in `target_labels` -> 1, else 0.
    Returns uint8 array with values {0, 1}.
    """
    target = np.isin(np.asarray(gt_array), list(target_labels))

    return target.astype(np.uint8)


def bbox_from_mask(mask):
    """
    Tight bounding box of foreground pixels.

    Returns [x_min, y_min, x_max, y_max] with INCLUSIVE
    max coordinates (pixel indices), or None if empty.

    This is the convention used by the existing boxes.json
    (post stage15) and stage19 training, so every consumer
    must keep it unchanged.
    """
    ys, xs = np.where(np.asarray(mask) > 0)

    if len(xs) == 0:
        return None

    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]


def validate_bbox(bbox, size):
    """
    Range + geometry check for a bbox in a `size` image.

    `size` is (width, height). Returns True if valid.
    None counts as valid (it means "empty target").
    """
    if bbox is None:
        return True

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False

    try:
        x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return False

    w, h = size

    if not (0 <= x_min <= w - 1 and 0 <= y_min <= h - 1):
        return False

    if not (0 <= x_max <= w - 1 and 0 <= y_max <= h - 1):
        return False

    return x_min <= x_max and y_min <= y_max
