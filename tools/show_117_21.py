from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


DATASET = Path("GBC_dataset")
SPLIT = "Train"
FILENAME = "117_21.png"


inp_path = DATASET / SPLIT / "INP" / FILENAME
gt_path = DATASET / SPLIT / "GT" / FILENAME


inp = np.array(Image.open(inp_path))
gt = np.array(Image.open(gt_path))


labels = sorted(
    int(x)
    for x in np.unique(gt)
    if x != 0
)


print("=" * 70)
print(FILENAME)
print("=" * 70)

print("GT labels:", np.unique(gt).tolist())

for label in labels:
    print(
        f"Label {label}: {np.sum(gt == label)} pixels"
    )


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    len(labels) + 2,
    figsize=(4 * (len(labels) + 2), 8)
)


# CT
axes[0, 0].imshow(inp, cmap="gray")
axes[0, 0].set_title("INP\nCT")
axes[0, 0].axis("off")


# Original GT
axes[1, 0].imshow(gt)
axes[1, 0].set_title("Original GT\n0-8")
axes[1, 0].axis("off")


# All foreground
foreground = gt > 0

axes[0, 1].imshow(foreground, cmap="gray")
axes[0, 1].set_title(
    f"All foreground\n{foreground.sum()} pixels"
)
axes[0, 1].axis("off")


axes[1, 1].imshow(inp, cmap="gray")
axes[1, 1].imshow(
    foreground,
    alpha=0.5
)
axes[1, 1].set_title("CT + All GT")
axes[1, 1].axis("off")


# Individual labels
for i, label in enumerate(labels):

    mask = gt == label

    axes[0, i + 2].imshow(
        mask,
        cmap="gray"
    )

    axes[0, i + 2].set_title(
        f"Label {label}\n"
        f"{mask.sum()} pixels"
    )

    axes[0, i + 2].axis("off")


    axes[1, i + 2].imshow(
        inp,
        cmap="gray"
    )

    axes[1, i + 2].imshow(
        mask,
        alpha=0.5
    )

    axes[1, i + 2].set_title(
        f"CT + Label {label}"
    )

    axes[1, i + 2].axis("off")


plt.suptitle(
    f"GT Label Investigation - {FILENAME}",
    fontsize=16
)

plt.tight_layout()
plt.show()