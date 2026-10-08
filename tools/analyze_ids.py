from pathlib import Path
from collections import defaultdict


# ============================================================
# DATASET LOCATION
# ============================================================

dataset = Path("GBC_dataset")


# ============================================================
# FUNCTION TO ANALYZE ONE SPLIT
# ============================================================

def analyze_split(split_name):

    print("\n" + "=" * 60)
    print(split_name)
    print("=" * 60)

    inp_folder = dataset / split_name / "INP"

    # Dictionary:
    # patient_id -> list of slice_ids
    patients = defaultdict(list)

    # Look at all PNG files
    for file in inp_folder.glob("*.png"):

        # Example:
        # 9_0.png
        # file.stem = "9_0"

        name = file.stem

        # Split at "_"
        parts = name.split("_")

        # We expect something like:
        # ["9", "0"]

        if len(parts) != 2:
            print("Unexpected filename:", file.name)
            continue

        patient_id = parts[0]
        slice_id = parts[1]

        patients[patient_id].append(slice_id)

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    print("\nTotal unique first IDs:", len(patients))

    print("\nFirst 20 IDs and number of slices:")

    for patient_id in sorted(
        patients.keys(),
        key=lambda x: int(x)
    )[:20]:

        print(
            f"Patient/ID {patient_id}: "
            f"{len(patients[patient_id])} slices"
        )

    # ========================================================
    # SHOW SLICE IDs FOR FIRST FEW PATIENTS
    # ========================================================

    print("\nDetailed view of first 5 IDs:")

    for patient_id in sorted(
        patients.keys(),
        key=lambda x: int(x)
    )[:5]:

        slices = patients[patient_id]

        # Convert to integers for proper sorting
        slices = sorted(
            [int(x) for x in slices]
        )

        print(
            f"\nID {patient_id}"
        )

        print(
            "Number of slices:",
            len(slices)
        )

        print(
            "First slices:",
            slices[:10]
        )

        print(
            "Last slices:",
            slices[-10:]
        )


# ============================================================
# RUN FOR TRAIN AND TEST
# ============================================================

analyze_split("Train")
analyze_split("Test")


print("\n" + "=" * 60)
print("STAGE 2 COMPLETE")
print("=" * 60)