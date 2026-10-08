# ============================================================
# GBC EDA LIBRARY (merges old stages 6/9/10/11 + label scripts)
# ============================================================
#
# Pure analysis functions: each takes a folder/split and
# returns a plain dict. Printing/reporting lives in
# scripts/analyze_dataset.py, so a new model can reuse any
# of these without copy-paste.
#
# ============================================================

import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from gbc import data as D
from gbc.config import CLASS_BITS, SPLIT_DIRS, TARGET_LABELS


# ============================================================
# PATIENT INVENTORY (old stage6 + analyze_ids filename check)
# ============================================================

def patient_inventory(inp_dir):
    """
    Per-patient slice counts + malformed filename count.

    Returns dict(patient_id -> n_slices) plus key "_bad_names".
    """
    inp_dir = Path(inp_dir)
    patients = defaultdict(int)
    bad_names = 0

    for name in D.list_files(inp_dir, ".png"):
        pid = D.get_patient_id(name)
        if pid is None:
            bad_names += 1
        else:
            patients[pid] += 1

    counts = sorted(patients.values())

    return {
        "n_patients": len(patients),
        "n_slices": sum(counts),
        "min_slices": min(counts) if counts else 0,
        "max_slices": max(counts) if counts else 0,
        "mean_slices": statistics.mean(counts) if counts else 0,
        "bad_names": bad_names,
        "per_patient": dict(patients),
    }


def split_patient_ids(split_dirs=None):
    """{split: set(patient ids)} for leakage checks."""
    split_dirs = split_dirs or SPLIT_DIRS
    return {
        name: set(
            patient_inventory(Path(d) / "INP")["per_patient"]
        )
        for name, d in split_dirs.items()
    }


# ============================================================
# IMAGE STATISTICS (old stage9, INP branch)
# ============================================================

def image_stats(folder, limit=None):
    """Dims/modes/dtypes + intensity stats over a folder of PNGs."""
    files = [Path(folder) / n for n in D.list_files(folder, ".png")]

    if limit is not None:
        files = files[:limit]

    dims = Counter()
    modes = Counter()
    dtypes = Counter()
    mins, maxs, means, medians = [], [], [], []

    for path in files:
        try:
            with Image.open(path) as img:
                arr = np.array(img)
                dims[arr.shape] += 1
                modes[img.mode] += 1
                dtypes[str(arr.dtype)] += 1
                mins.append(float(arr.min()))
                maxs.append(float(arr.max()))
                means.append(float(arr.mean()))
                medians.append(float(np.median(arr)))
        except Exception:
            pass

    return {
        "n_files": len(files),
        "dims": dict(dims),
        "modes": dict(modes),
        "dtypes": dict(dtypes),
        "global_min": min(mins) if mins else None,
        "global_max": max(maxs) if maxs else None,
        "mean_intensity": statistics.mean(means) if means else None,
        "median_intensity": statistics.median(medians) if medians else None,
    }


# ============================================================
# GT MASK STATISTICS (old stage9 GT branch + analyze_gt_labels)
# ============================================================

def mask_stats(folder, limit=None):
    """
    Dims/modes, full label-value histogram (pixel + image
    counts), foreground size and bbox size stats.
    """
    files = [Path(folder) / n for n in D.list_files(folder, ".png")]

    if limit is not None:
        files = files[:limit]

    dims = Counter()
    pixel_counts = Counter()
    image_counts = Counter()
    fg_pixels = []
    bbox_areas = []

    for path in files:
        try:
            with Image.open(path) as img:
                dims[np.array(img).shape] += 1
            gt = D.load_gt(path)
        except Exception:
            continue

        for value in np.unique(gt).tolist():
            value = int(value)
            pixel_counts[value] += int((gt == value).sum())
            image_counts[value] += 1

        fg = int((gt > 0).sum())
        fg_pixels.append(fg)

        if fg > 0:
            ys, xs = np.where(gt > 0)
            bbox_areas.append(
                (int(xs.max()) - int(xs.min()) + 1)
                * (int(ys.max()) - int(ys.min()) + 1)
            )

    def _summary(values):
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
        }

    return {
        "n_files": len(files),
        "dims": dict(dims),
        "pixel_counts": dict(sorted(pixel_counts.items())),
        "image_counts": dict(sorted(image_counts.items())),
        "foreground_pixels": _summary(fg_pixels),
        "bbox_area": _summary(bbox_areas),
    }


# ============================================================
# CLASSIFICATION-LABEL DISTRIBUTION (old analyze_labels.py)
# ============================================================

def label_distribution(split_dir):
    """
    LABEL vector frequencies + per-bit (1..8) positivity.

    Flags bits with zero positives (unlearnable as-is).
    """
    label_dir = Path(split_dir) / "LABEL"
    names = D.list_files(label_dir, ".txt")

    vectors = Counter()
    bit_pos = Counter()
    n = 0

    for name in names:
        try:
            vec = D.load_label_vector(label_dir / name)
        except Exception:
            continue
        n += 1
        vectors[" ".join(map(str, vec.tolist()))] += 1
        for bit in CLASS_BITS:
            if vec[bit] == 1:
                bit_pos[bit] += 1

    return {
        "n_files": n,
        "n_unique_vectors": len(vectors),
        "most_common": vectors.most_common(10),
        "bit_positives": {
            bit: bit_pos.get(bit, 0) for bit in CLASS_BITS
        },
        "zero_bits": [
            bit for bit in CLASS_BITS if bit_pos.get(bit, 0) == 0
        ],
    }


# ============================================================
# SEGMENTATION-Target ANALYSIS (old stage10 + stage11)
# ============================================================

def _connected_components(binary):
    """(n_components, sizes) via scipy, or (None, []) fallback."""
    if not np.any(binary):
        return 0, []
    try:
        from scipy import ndimage
    except ImportError:
        return None, []
    labeled, num = ndimage.label(binary)
    sizes = np.bincount(labeled.ravel())[1:].tolist()
    return int(num), sizes


def target_analysis(split_dir, target_labels=None):
    """
    For the segmentation target set (default {1, 2}):
    slice presence, per-label presence, bbox w/h/area stats,
    connected components, tiny/huge box counts, patients hit.
    """
    target_labels = set(target_labels or TARGET_LABELS)
    gt_dir = Path(split_dir) / "GT"
    names = D.list_files(gt_dir, ".png")

    label_hits = Counter()
    widths, heights, areas = [], [], []
    comp_counts = []
    tiny = huge = 0
    n_target = n_empty = 0
    patients = Counter()

    for name in names:
        try:
            gt = D.load_gt(gt_dir / name)
        except Exception:
            continue

        present = [lb for lb in target_labels if (gt == lb).any()]
        for lb in present:
            label_hits[lb] += 1

        target = np.isin(gt, list(target_labels))
        if not target.any():
            n_empty += 1
            continue

        n_target += 1
        pid = D.get_patient_id(name)
        if pid is not None:
            patients[pid] += 1

        ys, xs = np.where(target)
        w = int(xs.max()) - int(xs.min()) + 1
        h = int(ys.max()) - int(ys.min()) + 1
        widths.append(w)
        heights.append(h)
        areas.append(w * h)

        n_comp, _ = _connected_components(target)
        if n_comp is not None:
            comp_counts.append(n_comp)

        if w * h < 25:
            tiny += 1
        if w * h > 30000:
            huge += 1

    def _summary(values):
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
        }

    multi = sum(c > 1 for c in comp_counts) if comp_counts else None

    return {
        "n_slices": len(names),
        "target_slices": n_target,
        "empty_slices": n_empty,
        "label_hits": dict(label_hits),
        "bbox_width": _summary(widths),
        "bbox_height": _summary(heights),
        "bbox_area": _summary(areas),
        "components_mean": (
            statistics.mean(comp_counts) if comp_counts else None
        ),
        "multi_component_slices": multi,
        "tiny_boxes": tiny,
        "huge_boxes": huge,
        "patients_with_target": len(patients),
    }
