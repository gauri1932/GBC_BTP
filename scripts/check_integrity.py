from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gbc.config import DATASET_DIR

# Windows consoles default to cp1252, which cannot print ✓/⚠.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = DATASET_DIR

ORIGINAL_TRAIN = BASE_DIR / "Train"
ORIGINAL_TEST = BASE_DIR / "Test"

SPLIT_TRAIN = BASE_DIR / "split" / "Train"
SPLIT_VAL = BASE_DIR / "split" / "Val"


# ============================================================
# HELPERS
# ============================================================

def get_patient_id(filename):
    match = re.match(r"^(\d+)_", filename)
    return int(match.group(1)) if match else None


def get_files(folder, extension):
    return {
        f.name
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == extension
    }


def get_patients(folder, extension):
    patients = set()

    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() == extension:
            pid = get_patient_id(f.name)

            if pid is not None:
                patients.add(pid)

    return patients


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STAGE 8 - FINAL DATASET INTEGRITY CHECK")
print("=" * 70)


# ============================================================
# 1. CHECK REQUIRED FOLDERS
# ============================================================

print()
print("-" * 70)
print("1. FOLDER EXISTENCE")
print("-" * 70)

folders = [
    ORIGINAL_TRAIN / "INP",
    ORIGINAL_TRAIN / "GT",
    ORIGINAL_TRAIN / "LABEL",

    ORIGINAL_TEST / "INP",
    ORIGINAL_TEST / "GT",
    ORIGINAL_TEST / "LABEL",

    SPLIT_TRAIN / "INP",
    SPLIT_TRAIN / "GT",
    SPLIT_TRAIN / "LABEL",

    SPLIT_VAL / "INP",
    SPLIT_VAL / "GT",
    SPLIT_VAL / "LABEL",
]

folder_error = False

for folder in folders:

    if folder.exists():
        print(f"✓ {folder}")
    else:
        print(f"✗ MISSING: {folder}")
        folder_error = True


if folder_error:
    print()
    print("ERROR: Required folders are missing.")
    raise SystemExit


# ============================================================
# 2. ORIGINAL TRAIN COUNTS
# ============================================================

print()
print("-" * 70)
print("2. ORIGINAL TRAIN DATA")
print("-" * 70)

orig_train_inp = get_files(ORIGINAL_TRAIN / "INP", ".png")
orig_train_gt = get_files(ORIGINAL_TRAIN / "GT", ".png")
orig_train_label = get_files(ORIGINAL_TRAIN / "LABEL", ".txt")

print(f"INP   : {len(orig_train_inp)}")
print(f"GT    : {len(orig_train_gt)}")
print(f"LABEL : {len(orig_train_label)}")


# ============================================================
# 3. SPLIT COUNTS
# ============================================================

print()
print("-" * 70)
print("3. SPLIT COUNTS")
print("-" * 70)

train_inp = get_files(SPLIT_TRAIN / "INP", ".png")
train_gt = get_files(SPLIT_TRAIN / "GT", ".png")
train_label = get_files(SPLIT_TRAIN / "LABEL", ".txt")

val_inp = get_files(SPLIT_VAL / "INP", ".png")
val_gt = get_files(SPLIT_VAL / "GT", ".png")
val_label = get_files(SPLIT_VAL / "LABEL", ".txt")

print()
print("SPLIT TRAIN")
print(f"INP   : {len(train_inp)}")
print(f"GT    : {len(train_gt)}")
print(f"LABEL : {len(train_label)}")

print()
print("VALIDATION")
print(f"INP   : {len(val_inp)}")
print(f"GT    : {len(val_gt)}")
print(f"LABEL : {len(val_label)}")


# ============================================================
# 4. TRAIN + VAL = ORIGINAL TRAIN
# ============================================================

print()
print("-" * 70)
print("4. TRAIN + VALIDATION COMPLETENESS")
print("-" * 70)

combined_inp = train_inp | val_inp
combined_gt = train_gt | val_gt
combined_label = train_label | val_label

print(f"Original INP : {len(orig_train_inp)}")
print(f"Split INP    : {len(combined_inp)}")

print(f"Original GT  : {len(orig_train_gt)}")
print(f"Split GT     : {len(combined_gt)}")

print(f"Original LABEL : {len(orig_train_label)}")
print(f"Split LABEL    : {len(combined_label)}")

if combined_inp == orig_train_inp:
    print("✓ All original INP files accounted for")
else:
    print("✗ INP mismatch")

if combined_gt == orig_train_gt:
    print("✓ All original GT files accounted for")
else:
    print("✗ GT mismatch")

if combined_label == orig_train_label:
    print("✓ All original LABEL files accounted for")
else:
    print("✗ LABEL mismatch")


# ============================================================
# 5. TRAIN / VAL FILE OVERLAP
# ============================================================

print()
print("-" * 70)
print("5. TRAIN / VALIDATION FILE OVERLAP")
print("-" * 70)

inp_overlap = train_inp & val_inp
gt_overlap = train_gt & val_gt
label_overlap = train_label & val_label

print(f"INP overlap   : {len(inp_overlap)}")
print(f"GT overlap    : {len(gt_overlap)}")
print(f"LABEL overlap : {len(label_overlap)}")

if not inp_overlap and not gt_overlap and not label_overlap:
    print("✓ No file overlap between Train and Validation")
else:
    print("✗ FILE LEAKAGE DETECTED")


# ============================================================
# 6. PATIENT OVERLAP
# ============================================================

print()
print("-" * 70)
print("6. PATIENT-LEVEL LEAKAGE CHECK")
print("-" * 70)

train_patients = get_patients(SPLIT_TRAIN / "INP", ".png")
val_patients = get_patients(SPLIT_VAL / "INP", ".png")

patient_overlap = train_patients & val_patients

print(f"Train patients : {len(train_patients)}")
print(f"Val patients   : {len(val_patients)}")
print(f"Overlap        : {len(patient_overlap)}")

if not patient_overlap:
    print("✓ NO PATIENT LEAKAGE")
else:
    print("✗ PATIENT LEAKAGE DETECTED")
    print(sorted(patient_overlap))


# ============================================================
# 7. INP / GT / LABEL MATCHING
# ============================================================

print()
print("-" * 70)
print("7. INP / GT / LABEL MATCHING")
print("-" * 70)


def png_to_label_stem(filename):
    return Path(filename).stem + ".txt"


def check_matching(name, inp, gt, label):

    expected_labels = {
        Path(x).stem + ".txt"
        for x in inp
    }

    missing_gt = inp - gt
    missing_label = expected_labels - label

    extra_gt = gt - inp
    extra_label = {
        Path(x).stem + ".png"
        for x in label
    } - inp

    print()
    print(name)

    print(f"INP files   : {len(inp)}")
    print(f"GT files    : {len(gt)}")
    print(f"LABEL files : {len(label)}")

    print(f"INP without GT    : {len(missing_gt)}")
    print(f"INP without LABEL : {len(missing_label)}")
    print(f"GT without INP    : {len(extra_gt)}")
    print(f"LABEL without INP : {len(extra_label)}")

    if not missing_gt and not missing_label:
        print("✓ INP / GT / LABEL alignment OK")
    else:
        print("✗ ALIGNMENT ERROR")


check_matching(
    "SPLIT TRAIN",
    train_inp,
    train_gt,
    train_label
)

check_matching(
    "VALIDATION",
    val_inp,
    val_gt,
    val_label
)


# ============================================================
# 8. OFFICIAL TEST DATA
# ============================================================

print()
print("-" * 70)
print("8. OFFICIAL TEST SET")
print("-" * 70)

test_inp = get_files(ORIGINAL_TEST / "INP", ".png")
test_gt = get_files(ORIGINAL_TEST / "GT", ".png")
test_label = get_files(ORIGINAL_TEST / "LABEL", ".txt")

test_patients = get_patients(ORIGINAL_TEST / "INP", ".png")

print(f"Test patients : {len(test_patients)}")
print(f"Test INP      : {len(test_inp)}")
print(f"Test GT       : {len(test_gt)}")
print(f"Test LABEL    : {len(test_label)}")

print()
print("✓ Official Test folder remains separate.")


# ============================================================
# 9. PATIENT COUNT SANITY
# ============================================================

print()
print("-" * 70)
print("9. PATIENT COUNT SANITY")
print("-" * 70)

original_patients = get_patients(
    ORIGINAL_TRAIN / "INP",
    ".png"
)

print(f"Original Train patients : {len(original_patients)}")
print(f"Split Train patients    : {len(train_patients)}")
print(f"Validation patients     : {len(val_patients)}")

if train_patients | val_patients == original_patients:
    print("✓ Every original TRAIN patient is accounted for")
else:
    print("✗ Patient accounting mismatch")


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STAGE 8 COMPLETE")
print("=" * 70)

print()
print("Dataset integrity checks finished.")
print("If all checks show ✓, the dataset is ready for")
print("MedSAM-specific preprocessing.")
print("=" * 70)