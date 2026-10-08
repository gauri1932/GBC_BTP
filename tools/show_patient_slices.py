from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

DATASET = Path("GBC_dataset")

SPLIT = "Train"

# Patient we want to inspect
PATIENT_ID = 106

# Number of consecutive slices to show
NUM_SLICES = 12


# ============================================================
# FIND FILES FOR ONE PATIENT
# ============================================================

inp_folder = DATASET / SPLIT / "INP"
gt_folder = DATASET / SPLIT / "GT"

files = []

for inp_file in inp_folder.glob("*.png"):

    # Expected filename:
    # 106_63.png

    parts = inp_file.stem.split("_")

    if len(parts) != 2:
        continue

    try:
        patient_id = int(parts[0])
        slice_id = int(parts[1])
    except ValueError:
        continue

    if patient_id == PATIENT_ID:

        gt_file = gt_folder / inp_file.name

        if gt_file.exists():
            files.append(
                (slice_id, inp_file, gt_file)
            )


# ============================================================
# SORT BY SLICE NUMBER
# ============================================================

files.sort(
    key=lambda x: x[0]
)


print("=" * 70)
print(f"PATIENT {PATIENT_ID}")
print("=" * 70)

print("Total available slices:", len(files))

print(
    "Slice numbers:",
    [x[0] for x in files]
)


# ============================================================
# SELECT MIDDLE CONSECUTIVE SLICES
# ============================================================

if len(files) == 0:

    print("\nNo files found for this patient.")
    print("Check PATIENT_ID and SPLIT.")
    exit()


# Start around the middle of the patient's scan
middle = len(files) // 2

start = max(
    0,
    middle - NUM_SLICES // 2
)

selected = files[
    start:start + NUM_SLICES
]


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    len(selected),
    figsize=(3 * len(selected), 9)
)


# ============================================================
# DISPLAY
# ============================================================

for col, (
    slice_id,
    inp_file,
    gt_file
) in enumerate(selected):

    inp = np.array(
        Image.open(inp_file)
    )

    gt = np.array(
        Image.open(gt_file)
    )

    binary_gt = gt > 0

    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

    axes[0, col].imshow(
        inp,
        cmap="gray"
    )

    axes[0, col].set_title(
        f"Slice {slice_id}"
    )

    axes[0, col].axis("off")


    # --------------------------------------------------------
    # ORIGINAL GT
    # --------------------------------------------------------

    axes[1, col].imshow(
        gt
    )

    axes[1, col].set_title(
        f"GT\nLabels: {np.unique(gt).tolist()}"
    )

    axes[1, col].axis("off")


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    axes[2, col].imshow(
        inp,
        cmap="gray"
    )

    axes[2, col].imshow(
        binary_gt,
        alpha=0.5
    )

    axes[2, col].set_title(
        f"Overlay\nFG: {np.count_nonzero(gt)}"
    )

    axes[2, col].axis("off")


# ============================================================
# ROW LABELS
# ============================================================

axes[0, 0].set_ylabel(
    "CT",
    fontsize=14
)

axes[1, 0].set_ylabel(
    "Original GT",
    fontsize=14
)

axes[2, 0].set_ylabel(
    "Overlay",
    fontsize=14
)


plt.suptitle(
    f"Patient {PATIENT_ID} - Consecutive Slices",
    fontsize=16
)

plt.tight_layout()

plt.show()