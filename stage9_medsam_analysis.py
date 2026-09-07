from pathlib import Path
from PIL import Image
import numpy as np
import statistics


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\Student\Downloads\GBC\GBC_dataset")

SPLIT_TRAIN = BASE_DIR / "split" / "Train"
SPLIT_VAL = BASE_DIR / "split" / "Val"
TEST_DIR = BASE_DIR / "Test"


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_LIMIT = None
# None = analyze everything
# You can change to 100 for a quick test


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STAGE 9 - MEDSAM-SPECIFIC IMAGE ANALYSIS")
print("=" * 70)


# ============================================================
# IMAGE ANALYSIS FUNCTION
# ============================================================

def analyze_images(folder):

    files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() == ".png"
    ])

    if SAMPLE_LIMIT is not None:
        files = files[:SAMPLE_LIMIT]

    dimensions = {}
    modes = {}
    dtypes = {}

    min_values = []
    max_values = []
    means = []
    medians = []

    for i, file in enumerate(files):

        try:
            img = Image.open(file)

            arr = np.array(img)

            # Dimensions
            dimensions[arr.shape] = dimensions.get(arr.shape, 0) + 1

            # Image mode
            modes[img.mode] = modes.get(img.mode, 0) + 1

            # Datatype
            dtypes[str(arr.dtype)] = dtypes.get(
                str(arr.dtype), 0
            ) + 1

            # Intensity statistics
            min_values.append(float(arr.min()))
            max_values.append(float(arr.max()))
            means.append(float(arr.mean()))
            medians.append(float(np.median(arr)))

        except Exception as e:

            print(f"ERROR reading {file.name}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(files)}")

    return {
        "files": len(files),
        "dimensions": dimensions,
        "modes": modes,
        "dtypes": dtypes,
        "min_values": min_values,
        "max_values": max_values,
        "means": means,
        "medians": medians
    }


# ============================================================
# PRINT IMAGE ANALYSIS
# ============================================================

def print_image_results(name, results):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print()
    print(f"Total images analyzed: {results['files']}")

    print()
    print("-" * 70)
    print("DIMENSIONS")
    print("-" * 70)

    for dim, count in sorted(results["dimensions"].items()):
        print(f"{dim}: {count}")

    print()
    print("-" * 70)
    print("IMAGE MODES")
    print("-" * 70)

    for mode, count in results["modes"].items():
        print(f"{mode}: {count}")

    print()
    print("-" * 70)
    print("DATATYPES")
    print("-" * 70)

    for dtype, count in results["dtypes"].items():
        print(f"{dtype}: {count}")

    print()
    print("-" * 70)
    print("INTENSITY STATISTICS")
    print("-" * 70)

    print(
        f"Minimum pixel intensity across images: "
        f"{min(results['min_values']):.2f}"
    )

    print(
        f"Maximum pixel intensity across images: "
        f"{max(results['max_values']):.2f}"
    )

    print(
        f"Mean image intensity: "
        f"{statistics.mean(results['means']):.2f}"
    )

    print(
        f"Median image intensity: "
        f"{statistics.median(results['medians']):.2f}"
    )


# ============================================================
# RUN IMAGE ANALYSIS
# ============================================================

train_results = analyze_images(
    SPLIT_TRAIN / "INP"
)

print_image_results(
    "TRAIN INP",
    train_results
)


val_results = analyze_images(
    SPLIT_VAL / "INP"
)

print_image_results(
    "VALIDATION INP",
    val_results
)


test_results = analyze_images(
    TEST_DIR / "INP"
)

print_image_results(
    "OFFICIAL TEST INP",
    test_results
)


# ============================================================
# GT ANALYSIS
# ============================================================

def analyze_masks(folder):

    files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() == ".png"
    ])

    if SAMPLE_LIMIT is not None:
        files = files[:SAMPLE_LIMIT]

    dimensions = {}
    modes = {}
    dtypes = {}

    unique_labels = set()

    foreground_pixels = []

    bbox_widths = []
    bbox_heights = []
    bbox_areas = []

    for i, file in enumerate(files):

        try:

            img = Image.open(file)
            mask = np.array(img)

            dimensions[mask.shape] = dimensions.get(
                mask.shape, 0
            ) + 1

            modes[img.mode] = modes.get(
                img.mode, 0
            ) + 1

            dtypes[str(mask.dtype)] = dtypes.get(
                str(mask.dtype), 0
            ) + 1

            labels = np.unique(mask)

            unique_labels.update(
                labels.tolist()
            )

            # Everything except background = foreground
            foreground = mask > 0

            count = int(foreground.sum())

            foreground_pixels.append(count)

            # Bounding box
            if count > 0:

                ys, xs = np.where(foreground)

                x_min = xs.min()
                x_max = xs.max()

                y_min = ys.min()
                y_max = ys.max()

                width = x_max - x_min + 1
                height = y_max - y_min + 1

                bbox_widths.append(width)
                bbox_heights.append(height)
                bbox_areas.append(width * height)

        except Exception as e:

            print(f"ERROR reading {file.name}: {e}")

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(files)}")

    return {
        "files": len(files),
        "dimensions": dimensions,
        "modes": modes,
        "dtypes": dtypes,
        "unique_labels": sorted(unique_labels),
        "foreground_pixels": foreground_pixels,
        "bbox_widths": bbox_widths,
        "bbox_heights": bbox_heights,
        "bbox_areas": bbox_areas
    }


# ============================================================
# PRINT MASK RESULTS
# ============================================================

def print_mask_results(name, results):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print()
    print(f"Total masks analyzed: {results['files']}")

    print()
    print("-" * 70)
    print("DIMENSIONS")
    print("-" * 70)

    for dim, count in sorted(results["dimensions"].items()):
        print(f"{dim}: {count}")

    print()
    print("-" * 70)
    print("MASK MODES")
    print("-" * 70)

    for mode, count in results["modes"].items():
        print(f"{mode}: {count}")

    print()
    print("-" * 70)
    print("DATATYPES")
    print("-" * 70)

    for dtype, count in results["dtypes"].items():
        print(f"{dtype}: {count}")

    print()
    print("-" * 70)
    print("UNIQUE GT LABELS")
    print("-" * 70)

    print(results["unique_labels"])

    print()
    print("-" * 70)
    print("FOREGROUND PIXELS")
    print("-" * 70)

    if results["foreground_pixels"]:

        print(
            f"Minimum: "
            f"{min(results['foreground_pixels'])}"
        )

        print(
            f"Maximum: "
            f"{max(results['foreground_pixels'])}"
        )

        print(
            f"Mean: "
            f"{statistics.mean(results['foreground_pixels']):.2f}"
        )

        print(
            f"Median: "
            f"{statistics.median(results['foreground_pixels']):.2f}"
        )

    print()
    print("-" * 70)
    print("BOUNDING BOX STATISTICS")
    print("-" * 70)

    if results["bbox_widths"]:

        print(
            f"Bounding-box width:"
            f" min={min(results['bbox_widths'])},"
            f" max={max(results['bbox_widths'])},"
            f" mean={statistics.mean(results['bbox_widths']):.2f}"
        )

        print(
            f"Bounding-box height:"
            f" min={min(results['bbox_heights'])},"
            f" max={max(results['bbox_heights'])},"
            f" mean={statistics.mean(results['bbox_heights']):.2f}"
        )

        print(
            f"Bounding-box area:"
            f" min={min(results['bbox_areas'])},"
            f" max={max(results['bbox_areas'])},"
            f" mean={statistics.mean(results['bbox_areas']):.2f}"
        )


# ============================================================
# RUN MASK ANALYSIS
# ============================================================

train_gt_results = analyze_masks(
    SPLIT_TRAIN / "GT"
)

print_mask_results(
    "TRAIN GT",
    train_gt_results
)


val_gt_results = analyze_masks(
    SPLIT_VAL / "GT"
)

print_mask_results(
    "VALIDATION GT",
    val_gt_results
)


test_gt_results = analyze_masks(
    TEST_DIR / "GT"
)

print_mask_results(
    "OFFICIAL TEST GT",
    test_gt_results
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STAGE 9 COMPLETE")
print("=" * 70)

print()
print("No files were modified.")
print("No files were resized.")
print("No files were normalized.")
print("This stage only analyzed the dataset.")

print("=" * 70)