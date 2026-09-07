from pathlib import Path


dataset = Path("GBC_dataset")

split = "Train"

patient_id = 106

label_folder = dataset / split / "LABEL"


# Find every LABEL file belonging to patient 106
files = sorted(
    label_folder.glob(f"{patient_id}_*.txt")
)


print("=" * 70)
print(f"LABEL FILES FOR PATIENT {patient_id}")
print("=" * 70)

print("Number of LABEL files:", len(files))

for file in files:

    print("\nFile:", file.name)

    with open(file, "r") as f:
        content = f.read().strip()

    print("Content:", content)