import os
import numpy as np
from PIL import Image

BASE = r"C:\Users\Student\Downloads\GBC\GBC_dataset"

paths = [
    os.path.join(BASE, "medsam_targets", "Train", "GT_binary"),
    os.path.join(BASE, "medsam_targets", "Val", "GT_binary"),
    os.path.join(BASE, "medsam_targets", "Test", "GT_binary"),
]

print("=" * 70)
print("MEDSAM BINARY MASK DIAGNOSTIC")
print("=" * 70)

for folder in paths:

    print("\n" + "=" * 70)
    print(folder)
    print("=" * 70)

    files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    )

    if not files:
        print("NO PNG FILES FOUND")
        continue

    # Check first few masks
    for filename in files[:5]:

        path = os.path.join(folder, filename)

        img = Image.open(path)
        arr = np.array(img)

        print("\nFile:", filename)
        print("PIL mode:", img.mode)
        print("Size:", img.size)
        print("NumPy shape:", arr.shape)
        print("NumPy dtype:", arr.dtype)
        print("Unique values:", np.unique(arr).tolist())
        print("Min:", arr.min())
        print("Max:", arr.max())
        print("Foreground pixels:", np.sum(arr == 1))
        print("Non-zero pixels:", np.count_nonzero(arr))

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)