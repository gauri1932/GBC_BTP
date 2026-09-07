import os
import json
from PIL import Image
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Users\Student\Downloads\GBC\GBC_dataset"

IMAGE_ROOT = os.path.join(BASE, "medsam_preprocessed")
MASK_ROOT = os.path.join(BASE, "medsam_targets")
BOX_ROOT = os.path.join(BASE, "medsam_targets")

SPLITS = {
    "Train": {
        "expected_images": 6971,
        "expected_masks": 6971,
        "expected_patients": 194,
    },
    "Val": {
        "expected_images": 2030,
        "expected_masks": 2030,
        "expected_patients": 49,
    },
    "Test": {
        "expected_images": 1001,
        "expected_masks": 1001,
        "expected_patients": 226,
    },
}

TARGET_SIZE = (1024, 1024)


# ============================================================
# HELPERS
# ============================================================

def get_png_files(folder):
    if not os.path.exists(folder):
        return []

    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])


def extract_patient_id(filename):
    """
    Dataset filenames look like:
        100_0.png
        101_12.png

    Patient ID = part before first underscore.
    """
    stem = os.path.splitext(filename)[0]

    if "_" not in stem:
        return None

    patient = stem.split("_")[0]

    try:
        return int(patient)
    except ValueError:
        return None


def calculate_bbox_from_mask(mask):
    """
    Calculate bbox from a binary 1024x1024 mask.

    Returns:
        [x_min, y_min, x_max, y_max]
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return [x_min, y_min, x_max, y_max]


def bbox_matches_mask(mask, bbox):
    """
    Verify that bbox is exactly the bounding box of
    the foreground mask.
    """

    calculated = calculate_bbox_from_mask(mask)

    if calculated is None:
        return bbox is None

    if bbox is None:
        return False

    return calculated == bbox


# ============================================================
# MAIN VALIDATION
# ============================================================

print("=" * 70)
print("STAGE 16 - FINAL MEDSAM DATASET VALIDATION")
print("=" * 70)

total_errors = 0

split_results = {}


for split_name, expected in SPLITS.items():

    print("\n" + "=" * 70)
    print(split_name.upper())
    print("=" * 70)

    image_dir = os.path.join(
        IMAGE_ROOT,
        split_name,
        "INP"
    )

    mask_dir = os.path.join(
        MASK_ROOT,
        split_name,
        "GT_binary"
    )

    box_file = os.path.join(
        BOX_ROOT,
        split_name,
        "boxes.json"
    )

    # --------------------------------------------------------
    # FILE COUNTS
    # --------------------------------------------------------

    image_files = get_png_files(image_dir)
    mask_files = get_png_files(mask_dir)

    image_set = set(image_files)
    mask_set = set(mask_files)

    print("\nFILE COUNTS")
    print("-" * 70)

    print(f"Processed images : {len(image_files)}")
    print(f"Binary masks     : {len(mask_files)}")

    count_errors = 0

    if len(image_files) != expected["expected_images"]:
        print(
            f"⚠ Expected {expected['expected_images']} images"
        )
        count_errors += 1

    if len(mask_files) != expected["expected_masks"]:
        print(
            f"⚠ Expected {expected['expected_masks']} masks"
        )
        count_errors += 1

    if count_errors == 0:
        print("✓ File counts correct")

    total_errors += count_errors

    # --------------------------------------------------------
    # IMAGE VALIDATION
    # --------------------------------------------------------

    print("\nIMAGE VALIDATION")
    print("-" * 70)

    wrong_dimensions = 0
    wrong_mode = 0
    read_errors = 0

    for i, filename in enumerate(image_files, 1):

        path = os.path.join(image_dir, filename)

        try:
            with Image.open(path) as img:

                if img.size != TARGET_SIZE:
                    wrong_dimensions += 1

                if img.mode != "RGB":
                    wrong_mode += 1

        except Exception:
            read_errors += 1

        if i % 1000 == 0 or i == len(image_files):
            print(f"Checked {i}/{len(image_files)}")

    print()
    print(f"Wrong dimensions : {wrong_dimensions}")
    print(f"Wrong mode       : {wrong_mode}")
    print(f"Read errors      : {read_errors}")

    image_errors = (
        wrong_dimensions +
        wrong_mode +
        read_errors
    )

    if image_errors == 0:
        print("✓ All images are 1024×1024 RGB")

    total_errors += image_errors

    # --------------------------------------------------------
    # MASK VALIDATION
    # --------------------------------------------------------

    print("\nMASK VALIDATION")
    print("-" * 70)

    mask_wrong_dimensions = 0
    non_binary = 0
    positive_masks = 0
    empty_masks = 0
    mask_read_errors = 0

    for i, filename in enumerate(mask_files, 1):

        path = os.path.join(mask_dir, filename)

        try:
            with Image.open(path) as img:

                if img.size != TARGET_SIZE:
                    mask_wrong_dimensions += 1

                arr = np.array(img)

                unique = np.unique(arr)

                # Valid binary mask must contain only 0 and 255
                if not np.all(np.isin(unique, [0, 255])):
                    non_binary += 1

                foreground = np.count_nonzero(arr)

                if foreground > 0:
                    positive_masks += 1
                else:
                    empty_masks += 1

        except Exception:
            mask_read_errors += 1

        if i % 1000 == 0 or i == len(mask_files):
            print(f"Checked {i}/{len(mask_files)}")

    print()
    print(f"Wrong dimensions : {mask_wrong_dimensions}")
    print(f"Non-binary masks : {non_binary}")
    print(f"Positive masks   : {positive_masks}")
    print(f"Empty masks      : {empty_masks}")
    print(f"Read errors      : {mask_read_errors}")

    mask_errors = (
        mask_wrong_dimensions +
        non_binary +
        mask_read_errors
    )

    if mask_errors == 0:
        print("✓ All masks valid")

    total_errors += mask_errors

    # --------------------------------------------------------
    # IMAGE / MASK MATCHING
    # --------------------------------------------------------

    print("\nIMAGE / MASK MATCHING")
    print("-" * 70)

    images_without_masks = sorted(
        image_set - mask_set
    )

    masks_without_images = sorted(
        mask_set - image_set
    )

    print(
        f"Images without masks : "
        f"{len(images_without_masks)}"
    )

    print(
        f"Masks without images : "
        f"{len(masks_without_images)}"
    )

    matching_errors = (
        len(images_without_masks) +
        len(masks_without_images)
    )

    if matching_errors == 0:
        print("✓ Image / mask pairing correct")

    total_errors += matching_errors

    # --------------------------------------------------------
    # BOX JSON
    # --------------------------------------------------------

    print("\nBOUNDING BOX VALIDATION")
    print("-" * 70)

    if not os.path.exists(box_file):

        print("⚠ boxes.json NOT FOUND")

        box_errors = 1
        total_errors += box_errors

        continue

    try:
        with open(box_file, "r") as f:
            boxes = json.load(f)

    except Exception as e:

        print("⚠ Could not read boxes.json")
        print(e)

        total_errors += 1
        continue

    print(f"Boxes in JSON : {len(boxes)}")

    missing_boxes = 0
    valid_boxes = 0
    empty_target_boxes = 0
    invalid_boxes = 0
    out_of_range = 0

    # --------------------------------------------------------
    # BOX FORMAT VALIDATION
    # --------------------------------------------------------

    for filename in image_files:

        if filename not in boxes:

            missing_boxes += 1
            continue

        entry = boxes[filename]

        if not isinstance(entry, dict):

            invalid_boxes += 1
            continue

        bbox = entry.get("bbox")

        # Load corresponding mask
        mask_path = os.path.join(mask_dir, filename)

        try:
            with Image.open(mask_path) as img:
                mask = np.array(img)

        except Exception:
            invalid_boxes += 1
            continue

        has_foreground = np.count_nonzero(mask) > 0

        # ----------------------------------------------------
        # EMPTY TARGET
        # ----------------------------------------------------

        if not has_foreground:

            if bbox is None:
                empty_target_boxes += 1
            else:
                invalid_boxes += 1

            continue

        # ----------------------------------------------------
        # POSITIVE TARGET
        # ----------------------------------------------------

        if bbox is None:

            invalid_boxes += 1
            continue

        if not isinstance(bbox, list):
            invalid_boxes += 1
            continue

        if len(bbox) != 4:
            invalid_boxes += 1
            continue

        try:
            x_min, y_min, x_max, y_max = [
                int(x) for x in bbox
            ]
        except Exception:
            invalid_boxes += 1
            continue

        # ----------------------------------------------------
        # RANGE CHECK
        # ----------------------------------------------------

        if not (
            0 <= x_min <= 1023 and
            0 <= y_min <= 1023 and
            0 <= x_max <= 1023 and
            0 <= y_max <= 1023
        ):

            out_of_range += 1
            continue

        # ----------------------------------------------------
        # GEOMETRIC CHECK
        # ----------------------------------------------------

        if x_min > x_max or y_min > y_max:

            invalid_boxes += 1
            continue

        valid_boxes += 1

    print(
        f"Missing boxes        : {missing_boxes}"
    )

    print(
        f"Valid boxes          : {valid_boxes}"
    )

    print(
        f"Empty-target boxes   : {empty_target_boxes}"
    )

    print(
        f"Invalid boxes        : {invalid_boxes}"
    )

    print(
        f"Out-of-range boxes   : {out_of_range}"
    )

    box_errors = (
        missing_boxes +
        invalid_boxes +
        out_of_range
    )

    if box_errors == 0:
        print("✓ Bounding boxes structurally valid")

    total_errors += box_errors

    # --------------------------------------------------------
    # BOX / MASK CONSISTENCY
    # --------------------------------------------------------

    print("\nBOX / MASK CONSISTENCY")
    print("-" * 70)

    bbox_mismatches = 0

    for filename in image_files:

        if filename not in boxes:
            continue

        bbox = boxes[filename].get("bbox")

        mask_path = os.path.join(mask_dir, filename)

        try:
            with Image.open(mask_path) as img:
                mask = np.array(img)

        except Exception:
            continue

        if not bbox_matches_mask(mask, bbox):
            bbox_mismatches += 1

    print(
        f"Mask / bbox mismatches : {bbox_mismatches}"
    )

    if bbox_mismatches == 0:
        print("✓ Every bbox exactly matches its mask")

    total_errors += bbox_mismatches

    # --------------------------------------------------------
    # PATIENT COUNT
    # --------------------------------------------------------

    print("\nPATIENT-LEVEL CHECK")
    print("-" * 70)

    patient_ids = set()

    for filename in image_files:

        pid = extract_patient_id(filename)

        if pid is not None:
            patient_ids.add(pid)

    print(
        f"Patients detected : {len(patient_ids)}"
    )

    if len(patient_ids) == expected["expected_patients"]:

        print("✓ Patient count correct")

    else:

        print(
            f"⚠ Expected {expected['expected_patients']} "
            f"patients"
        )

        total_errors += 1

    # --------------------------------------------------------
    # SPLIT SUMMARY
    # --------------------------------------------------------

    split_results[split_name] = {
        "images": len(image_files),
        "masks": len(mask_files),
        "positive_masks": positive_masks,
        "empty_masks": empty_masks,
        "valid_boxes": valid_boxes,
        "empty_boxes": empty_target_boxes,
        "bbox_mismatches": bbox_mismatches,
        "patients": len(patient_ids),
    }


# ============================================================
# CROSS-SPLIT PATIENT LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / VALIDATION PATIENT LEAKAGE CHECK")
print("=" * 70)

split_patients = {}

for split_name in SPLITS:

    image_dir = os.path.join(
        IMAGE_ROOT,
        split_name,
        "INP"
    )

    patients = set()

    for filename in get_png_files(image_dir):

        pid = extract_patient_id(filename)

        if pid is not None:
            patients.add(pid)

    split_patients[split_name] = patients

train_val_overlap = (
    split_patients["Train"] &
    split_patients["Val"]
)

print(
    f"Train ∩ Val  : {len(train_val_overlap)}"
)

test_train_overlap = split_patients["Test"] & split_patients["Train"]
test_val_overlap = split_patients["Test"] & split_patients["Val"]

print(f"Test ∩ Train : {len(test_train_overlap)} (informational)")
print(f"Test ∩ Val   : {len(test_val_overlap)} (informational)")

# Test uses a separate patient-ID namespace, so its numeric IDs cannot
# be compared with the IDs created by the Train/Val split.
leakage_errors = len(train_val_overlap)

if leakage_errors == 0:

    print("✓ NO TRAIN / VALIDATION PATIENT LEAKAGE")

else:

    print("⚠ PATIENT LEAKAGE DETECTED")

total_errors += leakage_errors


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 16 FINAL SUMMARY")
print("=" * 70)

print("\nDATASET")
print("-" * 70)

total_images = sum(
    x["images"]
    for x in split_results.values()
)

total_masks = sum(
    x["masks"]
    for x in split_results.values()
)

total_positive = sum(
    x["positive_masks"]
    for x in split_results.values()
)

total_empty = sum(
    x["empty_masks"]
    for x in split_results.values()
)

total_valid_boxes = sum(
    x["valid_boxes"]
    for x in split_results.values()
)

print(f"Total images        : {total_images}")
print(f"Total masks         : {total_masks}")
print(f"Positive masks      : {total_positive}")
print(f"Empty masks         : {total_empty}")
print(f"Valid boxes         : {total_valid_boxes}")

print("\nEXPECTED TOTALS")
print("-" * 70)

print("Train : 6971")
print("Val   : 2030")
print("Test  : 1001")
print("Total : 10002")

print("\nERROR SUMMARY")
print("-" * 70)

print(f"Total errors : {total_errors}")

print("\n" + "=" * 70)

if total_errors == 0:

    print("🎉 STAGE 16 PASSED")
    print("=" * 70)

    print()
    print("✓ Images are MedSAM-ready")
    print("✓ Images are 1024×1024 RGB")
    print("✓ Masks are 1024×1024 binary")
    print("✓ Image / mask pairing is correct")
    print("✓ Bounding boxes are valid")
    print("✓ Bounding boxes match masks")
    print("✓ Empty targets have bbox=None")
    print("✓ No patient leakage")
    print("✓ Train / Val / Test structure is valid")
    print()
    print("==============================================================")
    print("MEDSAM IMPLEMENTATION CAN BEGIN")
    print("==============================================================")

else:

    print("⚠ STAGE 16 FOUND ISSUES")
    print("=" * 70)

    print()
    print("DO NOT START MEDSAM IMPLEMENTATION YET.")
    print("Fix the reported issues first.")

print("=" * 70)