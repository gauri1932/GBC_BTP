from pathlib import Path
from collections import defaultdict
import re


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\Student\Downloads\GBC\GBC_dataset")

TRAIN_DIR = BASE_DIR / "Train"
TEST_DIR = BASE_DIR / "Test"

TRAIN_INP = TRAIN_DIR / "INP"
TEST_INP = TEST_DIR / "INP"


# ============================================================
# FUNCTION TO EXTRACT PATIENT ID
# ============================================================

def extract_patient_id(filename):
    """
    Example:
        106_63.png -> patient ID = 106
        117_21.png -> patient ID = 117
    """

    match = re.match(r"^(\d+)_", filename)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# ANALYZE DATASET
# ============================================================

def analyze_folder(folder):

    patient_slices = defaultdict(list)

    for file in folder.iterdir():

        if file.suffix.lower() != ".png":
            continue

        patient_id = extract_patient_id(file.name)

        if patient_id is not None:
            patient_slices[patient_id].append(file.name)

    return patient_slices


# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("TRAIN PATIENT INVENTORY")
print("=" * 70)

train_patients = analyze_folder(TRAIN_INP)

print("Unique patients:", len(train_patients))
print("Total slices:", sum(len(v) for v in train_patients.values()))

print("\nFirst 20 patients:")

for patient_id in sorted(train_patients)[:20]:

    slices = train_patients[patient_id]

    print(
        f"Patient {patient_id}: "
        f"{len(slices)} slices"
    )


# ============================================================
# TEST
# ============================================================

print("\n")
print("=" * 70)
print("TEST PATIENT INVENTORY")
print("=" * 70)

test_patients = analyze_folder(TEST_INP)

print("Unique patients:", len(test_patients))
print("Total slices:", sum(len(v) for v in test_patients.values()))

print("\nFirst 20 patients:")

for patient_id in sorted(test_patients)[:20]:

    slices = test_patients[patient_id]

    print(
        f"Patient {patient_id}: "
        f"{len(slices)} slices"
    )


# ============================================================
# ID COLLISION CHECK
# ============================================================

train_ids = set(train_patients.keys())
test_ids = set(test_patients.keys())

overlap = train_ids.intersection(test_ids)

print("\n")
print("=" * 70)
print("TRAIN / TEST ID OVERLAP")
print("=" * 70)

print("Train IDs:", len(train_ids))
print("Test IDs :", len(test_ids))
print("Same numeric IDs in both:", len(overlap))

if overlap:

    print("\nWARNING:")
    print(
        "The same numeric patient IDs occur in both Train and Test."
    )

    print("\nFirst 30 overlapping IDs:")

    print(sorted(overlap)[:30])

else:

    print("\nNo numeric ID overlap found.")


# ============================================================
# SLICE DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("SLICE DISTRIBUTION")
print("=" * 70)

train_counts = [len(v) for v in train_patients.values()]
test_counts = [len(v) for v in test_patients.values()]

print("\nTRAIN")
print("Minimum slices per patient:", min(train_counts))
print("Maximum slices per patient:", max(train_counts))
print("Average slices per patient:", sum(train_counts) / len(train_counts))

print("\nTEST")
print("Minimum slices per patient:", min(test_counts))
print("Maximum slices per patient:", max(test_counts))
print("Average slices per patient:", sum(test_counts) / len(test_counts))


print("\n")
print("=" * 70)
print("STAGE 6 COMPLETE")
print("=" * 70)