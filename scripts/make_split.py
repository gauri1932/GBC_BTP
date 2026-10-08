from pathlib import Path
import re
import random
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gbc.config import DATASET_DIR

BASE_DIR = DATASET_DIR
TRAIN_DIR = BASE_DIR / "Train"

INP_DIR = TRAIN_DIR / "INP"
GT_DIR = TRAIN_DIR / "GT"
LABEL_DIR = TRAIN_DIR / "LABEL"

OUTPUT_DIR = BASE_DIR / "Split"

TRAIN_OUT = OUTPUT_DIR / "Train"
VAL_OUT = OUTPUT_DIR / "Val"


# ============================================================
# SETTINGS
# ============================================================

VAL_RATIO = 0.20
SEED = 42


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for folder in [
    TRAIN_OUT / "INP",
    TRAIN_OUT / "GT",
    TRAIN_OUT / "LABEL",
    VAL_OUT / "INP",
    VAL_OUT / "GT",
    VAL_OUT / "LABEL",
]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXTRACT PATIENT ID
# ============================================================

def get_patient_id(filename):

    match = re.match(r"^(\d+)_", filename)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# GET PATIENT IDs FROM INP
# ============================================================

patient_ids = set()

for file in INP_DIR.iterdir():

    if file.suffix.lower() == ".png":

        patient_id = get_patient_id(file.name)

        if patient_id is not None:
            patient_ids.add(patient_id)


patient_ids = sorted(patient_ids)


# ============================================================
# PRINT INITIAL INFORMATION
# ============================================================

print("=" * 70)
print("STAGE 7 - PATIENT LEVEL TRAIN / VALIDATION SPLIT")
print("=" * 70)

print()
print(f"Total TRAIN patients: {len(patient_ids)}")


# ============================================================
# SHUFFLE PATIENTS
# ============================================================

random.seed(SEED)

shuffled_patients = patient_ids.copy()
random.shuffle(shuffled_patients)


# ============================================================
# SPLIT PATIENTS
# ============================================================

val_count = round(len(patient_ids) * VAL_RATIO)

val_patients = sorted(shuffled_patients[:val_count])
train_patients = sorted(shuffled_patients[val_count:])


# ============================================================
# DISPLAY SPLIT
# ============================================================

print()
print("-" * 70)
print("PATIENT SPLIT")
print("-" * 70)

print(f"Training patients:   {len(train_patients)}")
print(f"Validation patients: {len(val_patients)}")

print()
print("Validation patient IDs:")
print(val_patients)


# ============================================================
# CHECK FOR PATIENT OVERLAP
# ============================================================

overlap = set(train_patients) & set(val_patients)

print()
print("-" * 70)
print("LEAKAGE CHECK")
print("-" * 70)

print(f"Patient overlap: {len(overlap)}")

if overlap:
    print("ERROR: Patient leakage detected!")
    print(overlap)
    raise SystemExit

else:
    print("✓ No patient overlap")


# ============================================================
# FUNCTION TO COPY A PATIENT
# ============================================================

def copy_patient(patient_id, destination):

    copied = {
        "INP": 0,
        "GT": 0,
        "LABEL": 0
    }

    # --------------------------------------------------------
    # INP
    # --------------------------------------------------------

    for file in INP_DIR.iterdir():

        if file.suffix.lower() != ".png":
            continue

        if get_patient_id(file.name) == patient_id:

            shutil.copy2(
                file,
                destination / "INP" / file.name
            )

            copied["INP"] += 1

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    for file in GT_DIR.iterdir():

        if file.suffix.lower() != ".png":
            continue

        if get_patient_id(file.name) == patient_id:

            shutil.copy2(
                file,
                destination / "GT" / file.name
            )

            copied["GT"] += 1

    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    for file in LABEL_DIR.iterdir():

        if file.suffix.lower() != ".txt":
            continue

        if get_patient_id(file.name) == patient_id:

            shutil.copy2(
                file,
                destination / "LABEL" / file.name
            )

            copied["LABEL"] += 1

    return copied


# ============================================================
# COPY TRAINING PATIENTS
# ============================================================

print()
print("-" * 70)
print("COPYING TRAINING DATA")
print("-" * 70)

train_counts = {
    "INP": 0,
    "GT": 0,
    "LABEL": 0
}

for i, patient_id in enumerate(train_patients, 1):

    counts = copy_patient(patient_id, TRAIN_OUT)

    for key in train_counts:
        train_counts[key] += counts[key]

    if i % 20 == 0 or i == len(train_patients):
        print(
            f"Processed {i}/{len(train_patients)} patients"
        )


# ============================================================
# COPY VALIDATION PATIENTS
# ============================================================

print()
print("-" * 70)
print("COPYING VALIDATION DATA")
print("-" * 70)

val_counts = {
    "INP": 0,
    "GT": 0,
    "LABEL": 0
}

for i, patient_id in enumerate(val_patients, 1):

    counts = copy_patient(patient_id, VAL_OUT)

    for key in val_counts:
        val_counts[key] += counts[key]

    if i % 10 == 0 or i == len(val_patients):
        print(
            f"Processed {i}/{len(val_patients)} patients"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 70)
print("STAGE 7 RESULTS")
print("=" * 70)

print()
print("PATIENTS")
print("-" * 70)

print(f"Original TRAIN patients : {len(patient_ids)}")
print(f"New TRAIN patients      : {len(train_patients)}")
print(f"Validation patients     : {len(val_patients)}")


print()
print("FILES")
print("-" * 70)

print("TRAIN:")
print(f"  INP   : {train_counts['INP']}")
print(f"  GT    : {train_counts['GT']}")
print(f"  LABEL : {train_counts['LABEL']}")

print()
print("VALIDATION:")
print(f"  INP   : {val_counts['INP']}")
print(f"  GT    : {val_counts['GT']}")
print(f"  LABEL : {val_counts['LABEL']}")


# ============================================================
# PAIRING CHECK
# ============================================================

print()
print("-" * 70)
print("PAIRING CHECK")
print("-" * 70)


def get_stems(folder, extension):

    return {
        file.stem
        for file in folder.iterdir()
        if file.suffix.lower() == extension
    }


for split_name, split_dir in [
    ("TRAIN", TRAIN_OUT),
    ("VALIDATION", VAL_OUT)
]:

    inp = get_stems(split_dir / "INP", ".png")
    gt = get_stems(split_dir / "GT", ".png")
    label = get_stems(split_dir / "LABEL", ".txt")

    print()
    print(split_name)

    print(f"INP   : {len(inp)}")
    print(f"GT    : {len(gt)}")
    print(f"LABEL : {len(label)}")

    print(f"INP without GT    : {len(inp - gt)}")
    print(f"INP without LABEL : {len(inp - label)}")
    print(f"GT without INP    : {len(gt - inp)}")
    print(f"LABEL without INP : {len(label - inp)}")


# ============================================================
# TEST REMINDER
# ============================================================

print()
print("=" * 70)
print("OFFICIAL TEST SET")
print("=" * 70)

print("The original Test folder was NOT modified.")
print("Test patients remain completely untouched.")


print()
print("=" * 70)
print("STAGE 7 COMPLETE")
print("=" * 70)