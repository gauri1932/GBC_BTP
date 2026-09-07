import os
import json
from PIL import Image
import numpy as np

# ============================================================
# STAGE 14 - MEDSAM READY DATA VERIFICATION
# ============================================================

BASE_DIR = r"C:\Users\Student\Downloads\GBC\GBC_dataset"

IMAGE_BASE = os.path.join(BASE_DIR, "medsam_preprocessed")
TARGET_BASE = os.path.join(BASE_DIR, "medsam_targets")

SPLITS = ["Train", "Val", "Test"]

EXPECTED_SIZE = (1024, 1024)
EXPECTED_MODE = "RGB"


def get_png_files(folder):
    if not os.path.exists(folder):
        return []

    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])


def validate_images(folder):
    files = get_png_files(folder)

    wrong_dimensions = 0
    wrong_mode = 0
    read_errors = 0

    for i, filename in enumerate(files, 1):
        path = os.path.join(folder, filename)

        try:
            img = Image.open(path)

            if img.size != EXPECTED_SIZE:
                wrong_dimensions += 1

            if img.mode != EXPECTED_MODE:
                wrong_mode += 1

            img.close()

        except Exception:
            read_errors += 1

        if i % 1000 == 0 or i == len(files):
            print(f"Checked {i}/{len(files)}")

    return files, wrong_dimensions, wrong_mode, read_errors


def validate_masks(folder):
    files = get_png_files(folder)

    wrong_dimensions = 0
    non_binary = 0
    positive = 0
    empty = 0
    read_errors = 0

    for i, filename in enumerate(files, 1):

        path = os.path.join(folder, filename)

        try:
            img = Image.open(path)

            if img.size != EXPECTED_SIZE:
                wrong_dimensions += 1

            arr = np.array(img)

            unique = np.unique(arr)

            # Expected binary mask values are 0 and 255
            if not set(unique).issubset({0, 255}):
                non_binary += 1

            foreground = np.count_nonzero(arr)

            if foreground > 0:
                positive += 1
            else:
                empty += 1

            img.close()

        except Exception:
            read_errors += 1

        if i % 1000 == 0 or i == len(files):
            print(f"Checked {i}/{len(files)}")

    return (
        files,
        wrong_dimensions,
        non_binary,
        positive,
        empty,
        read_errors
    )


def validate_boxes(json_path, image_files, mask_folder):

    with open(json_path, "r") as f:
        boxes = json.load(f)

    valid_boxes = 0
    empty_target = 0
    invalid_boxes = 0
    out_of_range = 0
    missing_boxes = 0
    mask_bbox_mismatch = 0

    image_set = set(image_files)

    for filename in image_files:

        if filename not in boxes:
            missing_boxes += 1
            continue

        info = boxes[filename]
        bbox = info.get("bbox")

        mask_path = os.path.join(mask_folder, filename)

        try:
            mask = np.array(Image.open(mask_path))

            ys, xs = np.where(mask > 0)

            # ------------------------------------------------
            # EMPTY TARGET
            # ------------------------------------------------
            if len(xs) == 0:

                if bbox is not None:
                    invalid_boxes += 1

                else:
                    empty_target += 1

                continue

            # ------------------------------------------------
            # TARGET EXISTS
            # ------------------------------------------------
            if bbox is None:
                invalid_boxes += 1
                continue

            if not isinstance(bbox, list) or len(bbox) != 4:
                invalid_boxes += 1
                continue

            x1, y1, x2, y2 = bbox

            # Must be numbers
            if not all(isinstance(v, (int, float)) for v in bbox):
                invalid_boxes += 1
                continue

            # Range check
            if (
                x1 < 0 or
                y1 < 0 or
                x2 > 1024 or
                y2 > 1024 or
                x1 >= x2 or
                y1 >= y2
            ):
                out_of_range += 1
                continue

            valid_boxes += 1

            # ------------------------------------------------
            # CHECK BBOX AGAINST MASK
            # ------------------------------------------------
            actual_x1 = int(xs.min())
            actual_y1 = int(ys.min())
            actual_x2 = int(xs.max()) + 1
            actual_y2 = int(ys.max()) + 1

            if (
                int(x1) != actual_x1 or
                int(y1) != actual_y1 or
                int(x2) != actual_x2 or
                int(y2) != actual_y2
            ):
                mask_bbox_mismatch += 1

        except Exception:
            invalid_boxes += 1

    return (
        len(boxes),
        missing_boxes,
        valid_boxes,
        empty_target,
        invalid_boxes,
        out_of_range,
        mask_bbox_mismatch
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("STAGE 14 - MEDSAM READY DATA VERIFICATION")
print("=" * 70)

total_images = 0
total_masks = 0
total_positive = 0
total_empty = 0
total_box_errors = 0
total_mask_errors = 0

for split in SPLITS:

    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    image_dir = os.path.join(
        IMAGE_BASE,
        split,
        "INP"
    )

    mask_dir = os.path.join(
        TARGET_BASE,
        split,
        "GT_binary"
    )

    box_path = os.path.join(
        TARGET_BASE,
        split,
        "boxes.json"
    )

    # --------------------------------------------------------
    # FILE COUNTS
    # --------------------------------------------------------

    image_files = get_png_files(image_dir)
    mask_files = get_png_files(mask_dir)

    print("\nFILE COUNTS")
    print("-" * 70)

    print(f"Processed images : {len(image_files)}")
    print(f"Binary masks     : {len(mask_files)}")

    total_images += len(image_files)
    total_masks += len(mask_files)

    # --------------------------------------------------------
    # IMAGE VALIDATION
    # --------------------------------------------------------

    print("\nIMAGE VALIDATION")
    print("-" * 70)

    (
        _,
        wrong_dimensions,
        wrong_mode,
        read_errors
    ) = validate_images(image_dir)

    print(f"\nWrong dimensions : {wrong_dimensions}")
    print(f"Wrong mode       : {wrong_mode}")
    print(f"Read errors      : {read_errors}")

    # --------------------------------------------------------
    # MASK VALIDATION
    # --------------------------------------------------------

    print("\nMASK VALIDATION")
    print("-" * 70)

    (
        _,
        mask_wrong_dimensions,
        non_binary,
        positive,
        empty,
        mask_read_errors
    ) = validate_masks(mask_dir)

    print(f"\nWrong dimensions : {mask_wrong_dimensions}")
    print(f"Non-binary masks : {non_binary}")
    print(f"Positive masks   : {positive}")
    print(f"Empty masks      : {empty}")
    print(f"Read errors      : {mask_read_errors}")

    total_positive += positive
    total_empty += empty

    total_mask_errors += (
        mask_wrong_dimensions +
        non_binary +
        mask_read_errors
    )

    # --------------------------------------------------------
    # IMAGE / MASK MATCHING
    # --------------------------------------------------------

    print("\nIMAGE / MASK MATCHING")
    print("-" * 70)

    image_set = set(image_files)
    mask_set = set(mask_files)

    images_without_masks = len(image_set - mask_set)
    masks_without_images = len(mask_set - image_set)

    print(f"Images without masks : {images_without_masks}")
    print(f"Masks without images : {masks_without_images}")

    # --------------------------------------------------------
    # BOX VALIDATION
    # --------------------------------------------------------

    print("\nBOUNDING BOX VALIDATION")
    print("-" * 70)

    if os.path.exists(box_path):

        (
            number_entries,
            missing_boxes,
            valid_boxes,
            empty_target,
            invalid_boxes,
            out_of_range,
            bbox_mismatch
        ) = validate_boxes(
            box_path,
            image_files,
            mask_dir
        )

        print(f"Boxes in JSON        : {number_entries}")
        print(f"Missing boxes        : {missing_boxes}")
        print(f"Valid boxes          : {valid_boxes}")
        print(f"Empty-target boxes   : {empty_target}")
        print(f"Invalid boxes        : {invalid_boxes}")
        print(f"Out-of-range boxes   : {out_of_range}")

        print("\nBOX / MASK CONSISTENCY")
        print("-" * 70)

        print(f"Mask / bbox mismatches : {bbox_mismatch}")

        total_box_errors += (
            missing_boxes +
            invalid_boxes +
            out_of_range +
            bbox_mismatch
        )

    else:

        print("boxes.json NOT FOUND")

        total_box_errors += 1

    # --------------------------------------------------------
    # SPLIT SUMMARY
    # --------------------------------------------------------

    split_errors = (
        wrong_dimensions +
        wrong_mode +
        read_errors +
        mask_wrong_dimensions +
        non_binary +
        mask_read_errors +
        images_without_masks +
        masks_without_images +
        total_box_errors
    )

    if (
        wrong_dimensions == 0 and
        wrong_mode == 0 and
        read_errors == 0 and
        mask_wrong_dimensions == 0 and
        non_binary == 0 and
        mask_read_errors == 0 and
        images_without_masks == 0 and
        masks_without_images == 0 and
        missing_boxes == 0 and
        invalid_boxes == 0 and
        out_of_range == 0 and
        bbox_mismatch == 0
    ):
        print("\n✓ SPLIT PASSED")

    else:
        print("\n⚠ SPLIT HAS ISSUES")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STAGE 14 FINAL SUMMARY")
print("=" * 70)

print("\nTOTAL FILES")
print("-" * 70)

print(f"Processed images : {total_images}")
print(f"Binary masks     : {total_masks}")

print("\nMASK SUMMARY")
print("-" * 70)

print(f"Positive masks   : {total_positive}")
print(f"Empty masks      : {total_empty}")

print("\nERROR SUMMARY")
print("-" * 70)

print(f"Mask errors      : {total_mask_errors}")
print(f"Box errors       : {total_box_errors}")

if (
    total_images == 10002 and
    total_masks == 10002 and
    total_mask_errors == 0 and
    total_box_errors == 0
):

    print("\n" + "=" * 70)
    print("✓ STAGE 14 PASSED")
    print("=" * 70)

    print("\nMedSAM-ready dataset verified.")
    print("Images : 1024×1024 RGB")
    print("Masks  : 1024×1024 binary")
    print("Boxes  : valid and consistent with masks")
    print("Image-mask pairing : OK")

else:

    print("\n" + "=" * 70)
    print("⚠ STAGE 14 FOUND ISSUES")
    print("=" * 70)

print("\nSTAGE 14 COMPLETE")