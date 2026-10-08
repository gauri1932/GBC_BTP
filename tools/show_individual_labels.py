from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


DATASET = Path("GBC_dataset")

SPLIT = "Train"

# We will inspect this slice
FILENAME = "106_63.png"


inp_path = DATASET / SPLIT / "INP" / FILENAME
gt_path = DATASET / SPLIT / "GT" / FILENAME


# ============================================================
# LOAD
# ============================================================

inp = np.array(
    Image.open(inp_path)
)

gt = np.array(
    Image.open(gt_path)
)


print("=" * 70)
print("INDIVIDUAL GT LABEL ANALYSIS")
print("=" * 70)

print("File:", FILENAME)

print("GT labels:", np.unique(gt).tolist())


# ============================================================
# FIND LABELS PRESENT
# ============================================================

labels = [
    int(x)
    for x in np.unique(gt)
    if x != 0
]


print("\nNon-background labels:")

for label in labels:

    count = np.sum(gt == label)

    print(
        f"Label {label}: {count} pixels"
    )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    len(labels) + 1,
    figsize=(4 * (len(labels) + 1), 8)
)


# ============================================================
# ORIGINAL CT
# ============================================================

axes[0, 0].imshow(
    inp,
    cmap="gray"
)

axes[0, 0].set_title(
    "INP\nCT"
)

axes[0, 0].axis("off")


# ============================================================
# ORIGINAL GT
# ============================================================

axes[1, 0].imshow(
    gt
)

axes[1, 0].set_title(
    "Original GT\n0-8"
)

axes[1, 0].axis("off")


# ============================================================
# INDIVIDUAL LABELS
# ============================================================

for i, label in enumerate(labels):

    mask = gt == label

    # Top row: mask
    axes[0, i + 1].imshow(
        mask,
        cmap="gray"
    )

    axes[0, i + 1].set_title(
        f"Label {label}\n"
        f"{np.sum(mask)} pixels"
    )

    axes[0, i + 1].axis("off")


    # Bottom row: overlay
    axes[1, i + 1].imshow(
        inp,
        cmap="gray"
    )

    axes[1, i + 1].imshow(
        mask,
        alpha=0.5
    )

    axes[1, i + 1].set_title(
        f"CT + Label {label}"
    )

    axes[1, i + 1].axis("off")


plt.suptitle(
    f"Individual GT Labels - {FILENAME}",
    fontsize=16
)

plt.tight_layout()

plt.show()