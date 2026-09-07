from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict


BASE = Path(
    r"C:\Users\Student\Downloads\GBC\GBC_dataset"
)

splits = {
    "TRAIN": BASE / "split" / "Train" / "GT",
    "VAL": BASE / "split" / "Val" / "GT",
    "TEST": BASE / "Test" / "GT",
}


TARGETS = [1, 2]


for split_name, folder in splits.items():

    print()
    print("=" * 70)
    print(split_name)
    print("=" * 70)

    total = 0

    label_counts = {
        label: 0
        for label in range(9)
    }

    target_present = 0
    target_absent = 0

    patients_with_target = set()

    for path in folder.glob("*.png"):

        total += 1

        gt = np.array(
            Image.open(path)
        )

        labels = set(
            np.unique(gt).tolist()
        )

        for label in labels:

            if label in label_counts:
                label_counts[label] += 1

        if any(
            label in labels
            for label in TARGETS
        ):

            target_present += 1

            patient_id = path.stem.split("_")[0]

            patients_with_target.add(
                patient_id
            )

        else:

            target_absent += 1

    print()
    print(f"Total slices: {total}")

    print()
    print("Slices containing each label:")

    for label, count in label_counts.items():

        print(
            f"Label {label}: {count}"
        )

    print()
    print(
        "Slices containing Label 1 OR Label 2:",
        target_present
    )

    print(
        "Slices WITHOUT Label 1/2:",
        target_absent
    )

    print()
    print(
        "Patients containing Label 1/2:",
        len(patients_with_target)
    )


print()
print("=" * 70)
print("TARGET ANALYSIS COMPLETE")
print("=" * 70)