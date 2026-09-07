import os
import json
import numpy as np
from PIL import Image

# ============================================================
# STAGE 15 - FIX MEDSAM BOUNDING BOX COORDINATES
# ============================================================

BASE_DIR = r"C:\Users\Student\Downloads\GBC\GBC_dataset"

TARGET_BASE = os.path.join(
    BASE_DIR,
    "medsam_targets"
)

SPLITS = ["Train", "Val", "Test"]

ORIGINAL_SIZE = 512
MEDSAM_SIZE = 1024

print("=" * 70)
print("STAGE 15 - MEDSAM BOUNDING BOX COORDINATE FIX")
print("=" * 70)

print()
print(f"Original image size : {ORIGINAL_SIZE} x {ORIGINAL_SIZE}")
print(f"MedSAM image size   : {MEDSAM_SIZE} x {MEDSAM_SIZE}")
print("Bounding boxes      : recomputed from 1024x1024 masks")
print()
print("Only bounding-box coordinates will be modified.")
print("GT masks and images will NOT be modified.")
print()


for split in SPLITS:

    print("=" * 70)
    print(split.upper())
    print("=" * 70)

    json_path = os.path.join(
        TARGET_BASE,
        split,
        "boxes.json"
    )

    if not os.path.exists(json_path):
        print(f"ERROR: boxes.json not found:")
        print(json_path)
        continue

    # --------------------------------------------------------
    # LOAD JSON
    # --------------------------------------------------------

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Entries found: {len(data)}")

    mask_dir = os.path.join(
        TARGET_BASE,
        split,
        "GT_binary"
    )

    changed = 0
    empty_targets = 0
    invalid = 0

    # --------------------------------------------------------
    # CONVERT BBOXES
    # --------------------------------------------------------

    for filename, info in data.items():

        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            invalid += 1
            continue

        try:
            with Image.open(mask_path) as mask_image:
                mask = np.array(mask_image)
        except Exception:
            invalid += 1
            continue

        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            info["bbox"] = None
            empty_targets += 1
        else:
            info["bbox"] = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ]


        # Update metadata to make coordinate system explicit
        info["bbox_coordinate_system"] = "1024x1024_MedSAM"

        changed += 1

    # --------------------------------------------------------
    # SAVE BACK
    # --------------------------------------------------------

    with open(json_path, "w") as f:
        json.dump(
            data,
            f,
            indent=2
        )

    print()
    print("RESULTS")
    print("-" * 70)

    print(f"Total entries       : {len(data)}")
    print(f"Bounding boxes fixed: {changed}")
    print(f"Empty targets       : {empty_targets}")
    print(f"Invalid entries     : {invalid}")

    print()
    print(f"Updated: {json_path}")

print()
print("=" * 70)
print("STAGE 15 COMPLETE")
print("=" * 70)

print()
print("Bounding boxes are now expressed in 1024x1024")
print("MedSAM image coordinates.")
print()
print("Images were NOT modified.")
print("GT masks were NOT modified.")
print("Only boxes.json was updated.")