import os
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\Users\Student\Downloads\GBC\GBC_dataset")

# Original GT folders
SPLITS = {
    "Train": BASE_DIR / "split" / "Train",
    "Val": BASE_DIR / "split" / "Val",
    "Test": BASE_DIR / "Test",
}

# Output directory
OUTPUT_BASE = BASE_DIR / "medsam_targets"

# MedSAM target labels
TARGET_LABELS = {1, 2}

# Original GT size
ORIGINAL_SIZE = (512, 512)

# MedSAM target size
MEDSAM_SIZE = (1024, 1024)


# ============================================================
# FUNCTIONS
# ============================================================

def get_bbox(binary_mask):
    """
    Calculate bounding box from binary mask.

    Returns:
        [x_min, y_min, x_max, y_max]
    """

    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())

    return [x_min, y_min, x_max, y_max]


def create_target_mask(gt_array):
    """
    Convert multiclass GT into binary target mask.

    Labels 1 and 2 -> foreground
    Everything else -> background
    """

    target = np.isin(gt_array, list(TARGET_LABELS))

    return target.astype(np.uint8)


def resize_mask(mask):
    """
    Resize binary mask using NEAREST NEIGHBOR.

    Important:
    Never use bilinear/bicubic interpolation for segmentation labels.
    """

    pil_mask = Image.fromarray(mask)

    resized = pil_mask.resize(
        MEDSAM_SIZE,
        Image.Resampling.NEAREST
    )

    return np.array(resized, dtype=np.uint8)


def process_split(split_name, split_path):

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name}")
    print("=" * 70)

    gt_dir = split_path / "GT"

    if not gt_dir.exists():
        raise FileNotFoundError(
            f"GT folder not found:\n{gt_dir}"
        )

    # Output directories
    mask_output_dir = OUTPUT_BASE / split_name / "GT_binary"
    mask_output_dir.mkdir(parents=True, exist_ok=True)

    box_output_dir = OUTPUT_BASE / split_name
    box_output_dir.mkdir(parents=True, exist_ok=True)

    # Find GT files
    gt_files = sorted(
        gt_dir.glob("*.png")
    )

    print(f"GT files found: {len(gt_files)}")

    boxes = {}

    statistics = {
        "total": 0,
        "target_present": 0,
        "target_absent": 0,
        "errors": 0,
    }

    for idx, gt_file in enumerate(
        tqdm(gt_files, desc=f"{split_name}")
    ):

        try:

            # ------------------------------------------------
            # LOAD ORIGINAL GT
            # ------------------------------------------------

            gt = np.array(
                Image.open(gt_file)
            )

            if gt.shape != ORIGINAL_SIZE:
                print(
                    f"\nWARNING: {gt_file.name} "
                    f"has shape {gt.shape}"
                )

            # ------------------------------------------------
            # CREATE BINARY TARGET
            # ------------------------------------------------

            binary_mask = create_target_mask(gt)

            # ------------------------------------------------
            # CHECK TARGET
            # ------------------------------------------------

            if binary_mask.sum() > 0:

                statistics["target_present"] += 1

                # ------------------------------------------------
                # RESIZE MASK
                # ------------------------------------------------

                mask_1024 = resize_mask(binary_mask)

                # ------------------------------------------------
                # CALCULATE BBOX ON 1024 MASK
                # ------------------------------------------------

                bbox = get_bbox(mask_1024)

                if bbox is None:
                    raise RuntimeError(
                        "Target disappeared after resizing"
                    )

                boxes[gt_file.name] = {
                    "bbox": bbox,
                    "target_labels": [1, 2],
                    "original_size": [
                        ORIGINAL_SIZE[1],
                        ORIGINAL_SIZE[0]
                    ],
                    "medsam_size": [
                        MEDSAM_SIZE[0],
                        MEDSAM_SIZE[1]
                    ],
                    "target_pixels_512": int(
                        binary_mask.sum()
                    ),
                    "target_pixels_1024": int(
                        mask_1024.sum()
                    ),
                }

            else:

                statistics["target_absent"] += 1

                # Save empty 1024 mask
                mask_1024 = resize_mask(
                    binary_mask
                )

                boxes[gt_file.name] = {
                    "bbox": None,
                    "target_labels": [1, 2],
                    "original_size": [
                        ORIGINAL_SIZE[1],
                        ORIGINAL_SIZE[0]
                    ],
                    "medsam_size": [
                        MEDSAM_SIZE[0],
                        MEDSAM_SIZE[1]
                    ],
                    "target_pixels_512": 0,
                    "target_pixels_1024": 0,
                }

            # ------------------------------------------------
            # SAVE BINARY MASK
            # ------------------------------------------------

            output_mask = mask_output_dir / gt_file.name

            Image.fromarray(
                mask_1024 * 255
            ).save(output_mask)

            statistics["total"] += 1

        except Exception as e:

            statistics["errors"] += 1

            print(
                f"\nERROR: {gt_file.name}"
            )
            print(e)

    # --------------------------------------------------------
    # SAVE BOX JSON
    # --------------------------------------------------------

    json_path = box_output_dir / "boxes.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            boxes,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"{split_name} RESULTS")
    print("-" * 70)

    print(
        f"Total GT files       : {statistics['total']}"
    )

    print(
        f"Target present       : "
        f"{statistics['target_present']}"
    )

    print(
        f"Target absent        : "
        f"{statistics['target_absent']}"
    )

    print(
        f"Errors               : "
        f"{statistics['errors']}"
    )

    print(
        f"Binary masks saved   : "
        f"{mask_output_dir}"
    )

    print(
        f"Bounding boxes saved : "
        f"{json_path}"
    )

    return statistics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 13 - MEDSAM GT TARGET + BOX GENERATION")
    print("=" * 70)

    print("\nTarget labels:")
    print("1 = Gallbladder Mass")
    print("2 = Gallbladder Wall Thickening")

    print("\nConversion:")
    print("Labels 1,2 -> foreground (1)")
    print("Labels 0,3,4,5,6,7,8 -> background (0)")

    print("\nMask resizing:")
    print("512×512 -> 1024×1024")
    print("Interpolation: NEAREST NEIGHBOR")

    print("\nOriginal GT files will NOT be modified.")

    all_stats = {}

    for split_name, split_path in SPLITS.items():

        all_stats[split_name] = process_split(
            split_name,
            split_path
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STAGE 13 COMPLETE")
    print("=" * 70)

    for split_name, stats in all_stats.items():

        print(
            f"\n{split_name}"
        )

        print(
            f"Total       : {stats['total']}"
        )

        print(
            f"Target      : {stats['target_present']}"
        )

        print(
            f"No target   : {stats['target_absent']}"
        )

        print(
            f"Errors      : {stats['errors']}"
        )

    print("\nOutput directory:")
    print(OUTPUT_BASE)

    print("\nOriginal GT remains untouched.")


if __name__ == "__main__":
    main()
    