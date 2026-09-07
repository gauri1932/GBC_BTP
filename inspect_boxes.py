import json
from pathlib import Path

BASE = Path(r"C:\Users\Student\Downloads\GBC\GBC_dataset\medsam_targets")

for split in ["Train", "Val", "Test"]:
    print("\n" + "=" * 70)
    print(split)
    print("=" * 70)

    path = BASE / split / "boxes.json"

    with open(path, "r") as f:
        data = json.load(f)

    print("JSON type:", type(data))
    print("Number of entries:", len(data))

    # Show first 5 entries
    print("\nFirst 5 entries:")
    for i, (key, value) in enumerate(data.items()):
        print(f"\nKey: {key}")
        print(f"Value: {value}")

        if i >= 4:
            break