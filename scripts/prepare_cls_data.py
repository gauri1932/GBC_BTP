# ============================================================
# PREPARE CLASSIFICATION DATA (reuses gbc_preprocess)
# ============================================================
#
# Proof of "recursive" reuse: this script builds a light,
# model-agnostic classification input using ONLY the generic
# library -- zero MedSAM specifics, zero code duplication.
#
# Output (cls_data/<Split>/):
#   INP/*.png   - grayscale slices resized to CLS_SIZE
#   labels.csv  - filename, patient_id, bit1..bit8
#
# labels.csv loads straight into sklearn / torch for the
# frozen-feature + linear-head pipeline (option A).
#
# Usage:
#   python prepare_classification_data.py
#   python prepare_classification_data.py --size 224 --split Val
#
# ============================================================

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gbc import data as P
from gbc.config import REPO_ROOT as REPO

ROOT = REPO
OUTPUT_BASE = ROOT / "cls_data"

CLS_SIZE = (256, 256)


def prepare_split(split_name, size):
    split_dir = P.SPLIT_DIRS[split_name]

    out_dir = OUTPUT_BASE / split_name / "INP"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_BASE / split_name / "labels.csv"

    inp_names = P.list_files(split_dir / "INP", ".png")

    print(f"\n{split_name}: {len(inp_names)} slices -> {size[0]}px gray")

    rows = []
    errors = []

    for filename in tqdm(inp_names, desc=split_name):
        try:
            img = P.load_inp_gray(split_dir / "INP" / filename)
            small = P.resize_image(img, size)
            small.save(out_dir / filename)

            bits = P.load_class_bits(split_dir / "LABEL" / f"{P.stem(filename)}.txt")

            rows.append(
                [filename, P.get_patient_id(filename), *bits.tolist()]
            )

        except Exception as e:
            errors.append((filename, str(e)))

    header = ["filename", "patient_id"] + [f"bit{i}" for i in P.CLASS_BITS]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  saved: {len(rows)} images + {csv_path.name}")
    print(f"  errors: {len(errors)}")
    for name, msg in errors[:10]:
        print(f"  ERROR {name}: {msg}")

    # Class balance preview (needed for loss weighting later).
    if rows:
        import numpy as np

        positives = np.array([r[2:] for r in rows]).sum(axis=0)
        print(f"  positives per bit1..8: {positives.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=CLS_SIZE[0])
    parser.add_argument("--split", choices=list(P.SPLIT_DIRS), default=None)
    args = parser.parse_args()

    size = (args.size, args.size)

    print("=" * 70)
    print("PREPARE CLASSIFICATION DATA (generic lib reuse)")
    print("=" * 70)
    print(f"\nImage size: {size[0]}x{size[1]} grayscale")
    print(f"Labels: bits 1..8 (bit 0 dropped, always 1)")

    splits = [args.split] if args.split else list(P.SPLIT_DIRS)

    for split_name in splits:
        prepare_split(split_name, size)

    print("\n" + "=" * 70)
    print(f"CLASSIFICATION DATA READY in {OUTPUT_BASE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
