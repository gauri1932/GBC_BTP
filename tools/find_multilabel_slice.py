from pathlib import Path
from PIL import Image
import numpy as np


DATASET = Path("GBC_dataset")
SPLIT = "Train"

gt_folder = DATASET / SPLIT / "GT"

best_file = None
best_labels = []
best_count = 0


for gt_file in gt_folder.glob("*.png"):

    gt = np.array(Image.open(gt_file))

    labels = sorted(
        int(x)
        for x in np.unique(gt)
        if x != 0
    )

    if len(labels) > best_count:
        best_count = len(labels)
        best_labels = labels
        best_file = gt_file.name


print("=" * 70)
print("SLICE WITH MOST GT LABELS")
print("=" * 70)

print("File:", best_file)
print("Number of non-background labels:", best_count)
print("Labels:", best_labels)

if best_file:
    gt = np.array(Image.open(gt_folder / best_file))

    print("\nPixel counts:")

    for label in best_labels:
        count = np.sum(gt == label)
        print(f"Label {label}: {count} pixels")

print("\nDONE")