# ============================================================
# MEDSAM-SPECIFIC PREPARATION (built on gbc_preprocess)
# ============================================================
#
# Everything in this file is specific to MedSAM and must NOT
# go into gbc_preprocess.py:
#   - 1024x1024 RGB inputs  (medsam_preprocessed/)
#   - binary masks of labels {1, 2} + boxes.json (medsam_targets/)
#
# Generic operations (loading, resizing, bbox math, pairing)
# are imported from gbc_preprocess so all pipelines share
# one implementation.
#
# ============================================================

import json
from pathlib import Path

import numpy as np
from PIL import Image

import gbc_preprocess as P


# ============================================================
# MEDSAM CONFIGURATION
# ============================================================

# MedSAM ViT-B encoder input resolution.
MEDSAM_SIZE = (1024, 1024)

# Segmentation target: gallbladder mass + wall thickening.
TARGET_LABELS = {1, 2}

LABEL_NAMES = {
    1: "Gallbladder Mass",
    2: "Gallbladder Wall Thickening",
}

# Outputs (same layout the stage17-19 code already consumes).
IMAGE_ROOT = P.ROOT / "medsam_preprocessed"
TARGET_ROOT = P.ROOT / "medsam_targets"


# ============================================================
# IMAGE BRANCH  (INP 512 gray -> 1024 RGB)
# ============================================================

def convert_inp_to_medsam(src_path, dst_path):
    """
    Convert one raw INP slice to MedSAM input.

    Steps: grayscale -> RGB -> resize 1024x1024 (bilinear).
    """
    img = Image.open(src_path)

    img = P.to_rgb(img)
    img = P.resize_image(img, MEDSAM_SIZE)

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_path, format="PNG")


def build_medsam_images(split_name, split_dir=None):
    """
    Build all MedSAM input images for one split.

    Returns (n_processed, errors : list of (filename, message)).
    Original INP files are never modified.
    """
    split_dir = Path(split_dir) if split_dir else P.SPLIT_DIRS[split_name]

    src_dir = split_dir / "INP"
    dst_dir = IMAGE_ROOT / split_name / "INP"
    dst_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    errors = []

    for filename in P.list_files(src_dir, ".png"):
        try:
            convert_inp_to_medsam(src_dir / filename, dst_dir / filename)
            processed += 1
        except Exception as e:
            errors.append((filename, str(e)))

    return processed, errors


# ============================================================
# TARGET BRANCH  (GT 512 multiclass -> 1024 binary + bbox)
# ============================================================
#
# NOTE (merges old stage13 + stage15):
# The bbox is computed directly on the FINAL 1024x1024 mask,
# so no post-hoc coordinate fix is ever needed.

def build_medsam_target(gt_array):
    """
    GT multiclass array -> (mask_1024, bbox, n_target_512).

    mask_1024 : uint8 {0, 1} at MEDSAM_SIZE.
    bbox      : inclusive [x_min, y_min, x_max, y_max] on the
                1024 mask, or None when the target is absent.
    """
    binary_512 = P.binary_mask_from_labels(gt_array, TARGET_LABELS)

    mask_1024 = P.resize_mask(binary_512, MEDSAM_SIZE)

    bbox = P.bbox_from_mask(mask_1024)

    return mask_1024, bbox, int(binary_512.sum())


def build_medsam_targets(split_name, split_dir=None):
    """
    Build binary masks + boxes.json for one split.

    boxes.json entry per slice:
        bbox=None  -> empty target (no prompt usable)
        bbox=[...] -> inclusive 1024 coordinates, exactly equal
                      to the bounding box of the saved mask.

    Returns a statistics dict. Original GT files are untouched.
    """
    split_dir = Path(split_dir) if split_dir else P.SPLIT_DIRS[split_name]

    gt_dir = split_dir / "GT"

    if not gt_dir.exists():
        raise FileNotFoundError(f"GT folder not found:\n{gt_dir}")

    mask_dir = TARGET_ROOT / split_name / "GT_binary"
    mask_dir.mkdir(parents=True, exist_ok=True)

    boxes = {}
    stats = {"total": 0, "target_present": 0, "target_absent": 0, "errors": 0}

    for filename in P.list_files(gt_dir, ".png"):
        try:
            gt = P.load_gt(gt_dir / filename)

            mask_1024, bbox, n_512 = build_medsam_target(gt)

            if bbox is None:
                stats["target_absent"] += 1
            else:
                stats["target_present"] += 1

            Image.fromarray(mask_1024 * 255).save(mask_dir / filename)

            boxes[filename] = {
                "bbox": bbox,
                "target_labels": sorted(TARGET_LABELS),
                "original_size": [512, 512],
                "medsam_size": [MEDSAM_SIZE[0], MEDSAM_SIZE[1]],
                "target_pixels_512": n_512,
                "target_pixels_1024": int(mask_1024.sum()),
            }

            stats["total"] += 1

        except Exception as e:
            stats["errors"] += 1
            print(f"\nERROR: {filename}\n{e}")

    json_path = TARGET_ROOT / split_name / "boxes.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(boxes, f, indent=2)

    stats["boxes_json"] = str(json_path)

    return stats


# ============================================================
# VALIDATION  (merges old stage14 + stage16 checks)
# ============================================================

def _validate_image(path):
    """Returns an error string, or None if the image is OK."""
    try:
        with Image.open(path) as img:
            if img.size != MEDSAM_SIZE:
                return f"wrong size {img.size}"
            if img.mode != "RGB":
                return f"wrong mode {img.mode}"
    except Exception as e:
        return f"read error: {e}"

    return None


def _validate_mask(path):
    """
    Returns (error_or_None, has_foreground_bool).
    A valid mask is 1024x1024 with values only in {0, 255}.
    """
    try:
        with Image.open(path) as img:
            if img.size != MEDSAM_SIZE:
                return f"wrong size {img.size}", False

            arr = np.array(img)

            if not np.all(np.isin(np.unique(arr), [0, 255])):
                return "non-binary values", False

            return None, bool(np.count_nonzero(arr))
    except Exception as e:
        return f"read error: {e}", False


def validate_medsam_split(split_name, check_pairing_with=None):
    """
    Full validation of one built split.

    Checks images, masks, pairing, boxes.json format AND
    box == recomputed mask bbox (the old stage14 vs stage16
    mismatch class that made stage15 necessary).

    Returns a dict of error counts (all 0 means PASS).
    """
    image_dir = IMAGE_ROOT / split_name / "INP"
    mask_dir = TARGET_ROOT / split_name / "GT_binary"
    json_path = TARGET_ROOT / split_name / "boxes.json"

    errors = {
        "image_errors": [],
        "mask_errors": [],
        "images_without_masks": 0,
        "masks_without_images": 0,
        "missing_boxes": 0,
        "invalid_boxes": 0,
        "out_of_range_boxes": 0,
        "box_mask_mismatch": 0,
        "boxes_json_missing": False,
    }

    image_files = P.list_files(image_dir, ".png")
    mask_files = P.list_files(mask_dir, ".png")

    for filename in image_files:
        err = _validate_image(image_dir / filename)
        if err:
            errors["image_errors"].append((filename, err))

    foreground_of = {}

    for filename in mask_files:
        err, fg = _validate_mask(mask_dir / filename)
        if err:
            errors["mask_errors"].append((filename, err))
        foreground_of[filename] = fg

    image_set, mask_set = set(image_files), set(mask_files)
    errors["images_without_masks"] = len(image_set - mask_set)
    errors["masks_without_images"] = len(mask_set - image_set)

    if not json_path.exists():
        errors["boxes_json_missing"] = True
        return errors

    with open(json_path, "r", encoding="utf-8") as f:
        boxes = json.load(f)

    for filename in image_files:
        if filename not in boxes:
            errors["missing_boxes"] += 1
            continue

        bbox = boxes[filename].get("bbox")
        has_fg = foreground_of.get(filename, False)

        # Empty target must pair with bbox=None and vice versa.
        if not has_fg or bbox is None:
            if has_fg != (bbox is not None):
                errors["invalid_boxes"] += 1
            continue

        if not P.validate_bbox(bbox, MEDSAM_SIZE):
            errors["out_of_range_boxes"] += 1
            continue

        # Box must be EXACTLY the bbox of its mask.
        try:
            mask = np.array(Image.open(mask_dir / filename))
        except Exception:
            errors["invalid_boxes"] += 1
            continue

        if P.bbox_from_mask(mask) != [int(v) for v in bbox]:
            errors["box_mask_mismatch"] += 1

    errors["n_images"] = len(image_files)
    errors["n_masks"] = len(mask_files)
    errors["n_boxes"] = len(boxes)

    if check_pairing_with is not None:
        errors["pairing"] = P.pairing_report(
            P.list_files(Path(check_pairing_with) / "INP", ".png"),
            P.list_files(Path(check_pairing_with) / "GT", ".png"),
            P.list_files(Path(check_pairing_with) / "LABEL", ".txt"),
        )

    return errors


def split_passed(errors):
    """True when every error count is zero / falsy."""
    return (
        not errors["image_errors"]
        and not errors["mask_errors"]
        and errors["images_without_masks"] == 0
        and errors["masks_without_images"] == 0
        and errors["missing_boxes"] == 0
        and errors["invalid_boxes"] == 0
        and errors["out_of_range_boxes"] == 0
        and errors["box_mask_mismatch"] == 0
        and not errors["boxes_json_missing"]
    )
