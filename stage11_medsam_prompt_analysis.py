import os
import numpy as np
from PIL import Image
from scipy import ndimage
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Student\Downloads\GBC\GBC_dataset"

SPLITS = {
    "TRAIN": os.path.join(BASE_DIR, "split", "Train"),
    "VAL": os.path.join(BASE_DIR, "split", "Val"),
    "TEST": os.path.join(BASE_DIR, "Test"),
}


# ============================================================
# MEDSAM TARGET LABELS
# ============================================================

# For gallbladder cancer segmentation, the primary target
# labels are:
#
# 1 = Gallbladder Mass
# 2 = Gallbladder Wall Thickening
#
# We analyze these separately and together.

TARGET_LABELS = [1, 2]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_files(folder):
    """Return sorted PNG files from a folder."""
    if not os.path.exists(folder):
        return []

    return sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])


def load_mask(path):
    """Load GT mask as uint8 numpy array."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def get_foreground_bbox(mask):
    """
    Return bounding box of non-zero pixels.

    Output:
        (x_min, y_min, x_max, y_max)

    If mask is empty:
        None
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return x_min, y_min, x_max, y_max


def get_target_bbox(mask):
    """
    Bounding box around target labels 1 and 2 only.
    """

    target_mask = np.isin(mask, TARGET_LABELS)

    ys, xs = np.where(target_mask)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return x_min, y_min, x_max, y_max


def bbox_dimensions(bbox):
    """
    Calculate width, height and area from:

    (x_min, y_min, x_max, y_max)
    """

    if bbox is None:
        return 0, 0, 0

    x_min, y_min, x_max, y_max = bbox

    # +1 because pixel coordinates are inclusive
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    area = width * height

    return width, height, area


def connected_components(mask):
    """
    Count connected components in a binary mask.
    """

    binary = mask > 0

    if not np.any(binary):
        return 0, []

    labeled, num = ndimage.label(binary)

    component_sizes = np.bincount(labeled.ravel())[1:]

    return num, component_sizes.tolist()


def get_patient_id(filename):
    """
    Extract patient ID.

    Example:
        117_21.png -> 117
    """

    try:
        return int(filename.split("_")[0])
    except:
        return None


# ============================================================
# ANALYZE ONE SPLIT
# ============================================================

def analyze_split(split_name, split_path):

    inp_dir = os.path.join(split_path, "INP")
    gt_dir = os.path.join(split_path, "GT")
    label_dir = os.path.join(split_path, "LABEL")

    gt_files = get_files(gt_dir)

    print("\n" + "=" * 70)
    print(split_name)
    print("=" * 70)

    print(f"Total slices: {len(gt_files)}")

    if len(gt_files) == 0:
        print("No GT files found.")
        return

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    target_slice_count = 0

    target_label_counts = Counter()

    target_bbox_widths = []
    target_bbox_heights = []
    target_bbox_areas = []

    component_counts = []

    empty_target_slices = 0

    patient_target_slices = Counter()

    # For prompt suitability
    small_bbox_count = 0
    large_bbox_count = 0

    processed = 0

    # --------------------------------------------------------
    # PROCESS ALL SLICES
    # --------------------------------------------------------

    for filename in gt_files:

        gt_path = os.path.join(gt_dir, filename)

        mask = load_mask(gt_path)

        # ----------------------------------------------------
        # Target mask = labels 1 and 2
        # ----------------------------------------------------

        target_mask = np.isin(mask, TARGET_LABELS)

        target_pixels = np.sum(target_mask)

        # ----------------------------------------------------
        # Count individual target labels
        # ----------------------------------------------------

        for label in TARGET_LABELS:

            count = np.sum(mask == label)

            if count > 0:
                target_label_counts[label] += 1

        # ----------------------------------------------------
        # No target
        # ----------------------------------------------------

        if target_pixels == 0:

            empty_target_slices += 1

        else:

            target_slice_count += 1

            # Patient-level count
            patient_id = get_patient_id(filename)

            if patient_id is not None:
                patient_target_slices[patient_id] += 1

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            bbox = get_target_bbox(mask)

            width, height, area = bbox_dimensions(bbox)

            target_bbox_widths.append(width)
            target_bbox_heights.append(height)
            target_bbox_areas.append(area)

            # ------------------------------------------------
            # Connected components
            # ------------------------------------------------

            num_components, sizes = connected_components(target_mask)

            component_counts.append(num_components)

            # ------------------------------------------------
            # Very small / large target regions
            # ------------------------------------------------

            if area < 25:
                small_bbox_count += 1

            if area > 30000:
                large_bbox_count += 1

        processed += 1

        if processed % 1000 == 0:
            print(f"Processed {processed}/{len(gt_files)}")

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "-" * 70)
    print("TARGET SLICE ANALYSIS")
    print("-" * 70)

    print(f"Target labels: {TARGET_LABELS}")

    print(f"Total slices: {len(gt_files)}")

    print(f"Slices containing Label 1/2: {target_slice_count}")

    print(f"Slices without Label 1/2: {empty_target_slices}")

    if len(gt_files) > 0:

        percentage = (
            target_slice_count / len(gt_files)
        ) * 100

        print(
            f"Target slice percentage: {percentage:.2f}%"
        )

    # ========================================================
    # LABEL PRESENCE
    # ========================================================

    print("\n" + "-" * 70)
    print("TARGET LABEL PRESENCE")
    print("-" * 70)

    for label in TARGET_LABELS:

        name = {
            1: "Gallbladder Mass",
            2: "Gallbladder Wall Thickening"
        }.get(label, "Unknown")

        print(
            f"Label {label} ({name}): "
            f"{target_label_counts[label]} slices"
        )

    # ========================================================
    # BOUNDING BOX
    # ========================================================

    print("\n" + "-" * 70)
    print("TARGET BOUNDING BOX STATISTICS")
    print("-" * 70)

    if target_bbox_widths:

        print(
            f"Width:  "
            f"min={min(target_bbox_widths)}, "
            f"max={max(target_bbox_widths)}, "
            f"mean={np.mean(target_bbox_widths):.2f}, "
            f"median={np.median(target_bbox_widths):.2f}"
        )

        print(
            f"Height: "
            f"min={min(target_bbox_heights)}, "
            f"max={max(target_bbox_heights)}, "
            f"mean={np.mean(target_bbox_heights):.2f}, "
            f"median={np.median(target_bbox_heights):.2f}"
        )

        print(
            f"Area:   "
            f"min={min(target_bbox_areas)}, "
            f"max={max(target_bbox_areas)}, "
            f"mean={np.mean(target_bbox_areas):.2f}, "
            f"median={np.median(target_bbox_areas):.2f}"
        )

    # ========================================================
    # CONNECTED COMPONENTS
    # ========================================================

    print("\n" + "-" * 70)
    print("CONNECTED COMPONENT ANALYSIS")
    print("-" * 70)

    if component_counts:

        print(
            f"Minimum components: "
            f"{min(component_counts)}"
        )

        print(
            f"Maximum components: "
            f"{max(component_counts)}"
        )

        print(
            f"Mean components: "
            f"{np.mean(component_counts):.2f}"
        )

        print(
            f"Median components: "
            f"{np.median(component_counts):.2f}"
        )

        print(
            f"Slices with >1 component: "
            f"{sum(c > 1 for c in component_counts)}"
        )

    # ========================================================
    # PROMPT SUITABILITY
    # ========================================================

    print("\n" + "-" * 70)
    print("MEDSAM PROMPT SUITABILITY")
    print("-" * 70)

    if target_slice_count > 0:

        small_percentage = (
            small_bbox_count / target_slice_count
        ) * 100

        large_percentage = (
            large_bbox_count / target_slice_count
        ) * 100

        print(
            f"Very small target bounding boxes (<25 px area): "
            f"{small_bbox_count} "
            f"({small_percentage:.2f}%)"
        )

        print(
            f"Large target bounding boxes (>30000 px area): "
            f"{large_bbox_count} "
            f"({large_percentage:.2f}%)"
        )

    # ========================================================
    # PATIENT LEVEL
    # ========================================================

    print("\n" + "-" * 70)
    print("PATIENT-LEVEL TARGET ANALYSIS")
    print("-" * 70)

    print(
        f"Patients containing Label 1/2: "
        f"{len(patient_target_slices)}"
    )

    if patient_target_slices:

        values = list(patient_target_slices.values())

        print(
            f"Target slices per patient: "
            f"min={min(values)}, "
            f"max={max(values)}, "
            f"mean={np.mean(values):.2f}, "
            f"median={np.median(values):.2f}"
        )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)

    print(
        "For MedSAM preprocessing, Labels 1 and 2 are treated "
        "as the primary target region."
    )

    print(
        "The bounding box is calculated from the target mask "
        "and can be used as a box prompt."
    )

    print(
        "No image or mask files were modified in this stage."
    )

    print(
        "No resizing, normalization, cropping, or augmentation "
        "was performed."
    )

    return {
        "total_slices": len(gt_files),
        "target_slices": target_slice_count,
        "empty_target_slices": empty_target_slices,
        "patients_with_target": len(patient_target_slices),
        "bbox_widths": target_bbox_widths,
        "bbox_heights": target_bbox_heights,
        "bbox_areas": target_bbox_areas,
        "components": component_counts,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STAGE 11 - MEDSAM PROMPT ANALYSIS")
    print("=" * 70)

    all_results = {}

    for split_name, split_path in SPLITS.items():

        if not os.path.exists(split_path):

            print(
                f"\nWARNING: {split_name} path does not exist:"
            )

            print(split_path)

            continue

        all_results[split_name] = analyze_split(
            split_name,
            split_path
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("STAGE 11 COMPLETE")
    print("=" * 70)

    print(
        "MedSAM target/prompt analysis completed."
    )

    print(
        "No dataset files were modified."
    )

    print(
        "No resizing or normalization was performed."
    )

    print(
        "Target labels analyzed: 1 (Mass), 2 (Wall Thickening)"
    )

    print("=" * 70)