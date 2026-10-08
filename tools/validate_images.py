from pathlib import Path
from PIL import Image
import numpy as np


# ============================================================
# DATASET
# ============================================================

dataset = Path("GBC_dataset")


# ============================================================
# VALIDATE ONE SPLIT
# ============================================================

def validate_split(split_name):

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} VALIDATION")
    print("=" * 70)

    inp_folder = dataset / split_name / "INP"
    gt_folder = dataset / split_name / "GT"

    inp_files = sorted(inp_folder.glob("*.png"))

    print("Total images:", len(inp_files))

    # Counters
    valid_pairs = 0
    size_mismatch = 0
    unreadable_inp = 0
    unreadable_gt = 0

    positive_slices = 0
    empty_slices = 0

    # Image dimensions
    inp_sizes = {}
    gt_sizes = {}

    # GT pixel values
    gt_values = set()

    # Number of foreground pixels
    foreground_counts = []

    # ========================================================
    # CHECK EVERY IMAGE
    # ========================================================

    for i, inp_file in enumerate(inp_files):

        gt_file = gt_folder / inp_file.name

        # ----------------------------------------------------
        # Check GT exists
        # ----------------------------------------------------

        if not gt_file.exists():

            print(
                "Missing GT:",
                inp_file.name
            )

            continue

        # ----------------------------------------------------
        # Open INP
        # ----------------------------------------------------

        try:

            with Image.open(inp_file) as img:

                inp_array = np.array(img)

                inp_sizes[inp_array.shape] = (
                    inp_sizes.get(inp_array.shape, 0) + 1
                )

        except Exception as e:

            unreadable_inp += 1

            print(
                "Unreadable INP:",
                inp_file.name,
                e
            )

            continue

        # ----------------------------------------------------
        # Open GT
        # ----------------------------------------------------

        try:

            with Image.open(gt_file) as img:

                gt_array = np.array(img)

                gt_sizes[gt_array.shape] = (
                    gt_sizes.get(gt_array.shape, 0) + 1
                )

        except Exception as e:

            unreadable_gt += 1

            print(
                "Unreadable GT:",
                gt_file.name,
                e
            )

            continue

        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        if inp_array.shape != gt_array.shape:

            size_mismatch += 1

            print(
                "SIZE MISMATCH:",
                inp_file.name,
                "INP:",
                inp_array.shape,
                "GT:",
                gt_array.shape
            )

            continue

        valid_pairs += 1

        # ----------------------------------------------------
        # GT unique pixel values
        # ----------------------------------------------------

        unique_values = np.unique(gt_array)

        gt_values.update(
            unique_values.tolist()
        )

        # ----------------------------------------------------
        # Count foreground pixels
        # ----------------------------------------------------

        foreground = np.count_nonzero(gt_array)

        foreground_counts.append(foreground)

        if foreground > 0:

            positive_slices += 1

        else:

            empty_slices += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (i + 1) % 1000 == 0:

            print(
                f"Processed {i + 1}/{len(inp_files)}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)

    print(
        "Valid INP-GT pairs:",
        valid_pairs
    )

    print(
        "Size mismatches:",
        size_mismatch
    )

    print(
        "Unreadable INP:",
        unreadable_inp
    )

    print(
        "Unreadable GT:",
        unreadable_gt
    )

    print("\nINP dimensions:")

    for shape, count in inp_sizes.items():

        print(
            f"  {shape}: {count} images"
        )

    print("\nGT dimensions:")

    for shape, count in gt_sizes.items():

        print(
            f"  {shape}: {count} masks"
        )

    print("\nUnique GT pixel values:")

    print(
        sorted(gt_values)
    )

    print("\nPositive slices:")

    print(
        positive_slices
    )

    print("\nEmpty slices:")

    print(
        empty_slices
    )

    # ========================================================
    # FOREGROUND STATISTICS
    # ========================================================

    if foreground_counts:

        foreground_counts = np.array(
            foreground_counts
        )

        print("\nForeground pixel statistics:")

        print(
            "Minimum:",
            foreground_counts.min()
        )

        print(
            "Maximum:",
            foreground_counts.max()
        )

        print(
            "Mean:",
            round(
                foreground_counts.mean(),
                2
            )
        )

        print(
            "Median:",
            np.median(foreground_counts)
        )


# ============================================================
# RUN
# ============================================================

validate_split("Train")

validate_split("Test")


print("\n" + "=" * 70)
print("STAGE 3 COMPLETE")
print("=" * 70)