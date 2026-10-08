from pathlib import Path
from PIL import Image
import numpy as np


# ============================================================
# DATASET LOCATION
# ============================================================

dataset = Path("GBC_dataset")


# ============================================================
# ANALYZE ONE SPLIT
# ============================================================

def analyze_split(split_name):

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} GT LABEL ANALYSIS")
    print("=" * 70)

    gt_folder = dataset / split_name / "GT"

    gt_files = sorted(gt_folder.glob("*.png"))

    print("Total GT masks:", len(gt_files))

    # Count pixels for every label
    label_pixel_counts = {}

    # Count how many images contain each label
    label_image_counts = {}

    # --------------------------------------------------------
    # Process every GT
    # --------------------------------------------------------

    for i, gt_file in enumerate(gt_files):

        gt = np.array(Image.open(gt_file))

        unique_values = np.unique(gt)

        for value in unique_values:

            value = int(value)

            # Initialize counters
            if value not in label_pixel_counts:
                label_pixel_counts[value] = 0

            if value not in label_image_counts:
                label_image_counts[value] = 0

            # Number of pixels belonging to this label
            pixel_count = np.sum(gt == value)

            label_pixel_counts[value] += int(pixel_count)

            # This image contains this label
            label_image_counts[value] += 1

        # Progress
        if (i + 1) % 1000 == 0:
            print(
                f"Processed {i + 1}/{len(gt_files)}"
            )

    # --------------------------------------------------------
    # Total pixels
    # --------------------------------------------------------

    total_pixels = sum(label_pixel_counts.values())

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("LABEL STATISTICS")
    print("-" * 70)

    print(
        f"{'Label':<10}"
        f"{'Pixel Count':<20}"
        f"{'Pixel %':<15}"
        f"{'Images Containing Label'}"
    )

    print("-" * 70)

    for label in sorted(label_pixel_counts):

        count = label_pixel_counts[label]

        percentage = (
            count / total_pixels
        ) * 100

        images = label_image_counts[label]

        print(
            f"{label:<10}"
            f"{count:<20}"
            f"{percentage:<15.4f}"
            f"{images}"
        )

    print("-" * 70)

    print(
        "Total pixels:",
        total_pixels
    )


# ============================================================
# RUN
# ============================================================

analyze_split("Train")

analyze_split("Test")


print("\n" + "=" * 70)
print("STAGE 4 COMPLETE")
print("=" * 70)