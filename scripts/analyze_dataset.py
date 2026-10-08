# ============================================================
# ANALYZE DATASET (single EDA entry point)
# ============================================================
#
# Runs every dataset analysis in one go (old stages 6/9/10/11
# plus the label scripts, now merged in gbc.eda):
#   1. patient inventory + train/val/test leakage check
#   2. INP image stats (dims, mode, intensity)
#   3. GT mask stats (label histogram, foreground, bboxes)
#   4. classification-label distribution (bits 1..8)
#   5. segmentation-target analysis (labels 1+2, bboxes)
#
# Read-only: never modifies the dataset.
#
# Usage (run from repo root):
#   python scripts/analyze_dataset.py
#   python scripts/analyze_dataset.py --quick 200
#
# ============================================================

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gbc import eda
from gbc.config import SPLIT_DIRS


def _print(title, items):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    for key, value in items:
        print(f"{key:<28}: {value}")


def _summary(values):
    if not values:
        return "n/a"
    return (
        f"min={values['min']}, max={values['max']}, "
        f"mean={values['mean']:.1f}, median={values['median']:.1f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", type=int, default=None,
        help="Analyze only the first N files per folder.",
    )
    parser.add_argument(
        "--split", choices=list(SPLIT_DIRS), default=None,
        help="Analyze a single split (default: all).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GBC DATASET ANALYSIS")
    print("=" * 70)

    splits = [args.split] if args.split else list(SPLIT_DIRS)

    # --------------------------------------------------------
    # 1. PATIENT INVENTORY + LEAKAGE CHECK
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("PATIENT INVENTORY")
    print("=" * 70)

    for split in splits:
        inv = eda.patient_inventory(SPLIT_DIRS[split] / "INP")
        _print(split, [
            ("patients", inv["n_patients"]),
            ("slices", inv["n_slices"]),
            ("slices/patient",
             f"min={inv['min_slices']}, max={inv['max_slices']}, "
             f"mean={inv['mean_slices']:.1f}"),
            ("malformed names", inv["bad_names"]),
        ])

    if len(splits) > 1:
        ids = eda.split_patient_ids(
            {s: SPLIT_DIRS[s] for s in splits}
        )
        names = list(ids)
        leaked = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = ids[names[i]] & ids[names[j]]
                if overlap:
                    leaked = True
                    print(f"\nLEAK: {names[i]} x {names[j]}: "
                          f"{sorted(overlap)}")
        print("\nPatient leakage: "
              + ("FOUND (see above)" if leaked else "none - OK"))

    # --------------------------------------------------------
    # 2-5. PER-SPLIT ANALYSIS
    # --------------------------------------------------------
    for split in splits:
        split_dir = SPLIT_DIRS[split]
        print("\n" + "=" * 70)
        print(f"{split.upper()} DETAIL")
        print("=" * 70)

        imgs = eda.image_stats(split_dir / "INP", limit=args.quick)
        _print("INP IMAGES", [
            ("files", imgs["n_files"]),
            ("shapes", imgs["dims"]),
            ("modes", imgs["modes"]),
            ("intensity",
             f"min={imgs['global_min']}, max={imgs['global_max']}, "
             f"mean={imgs['mean_intensity']:.1f}"),
        ])

        masks = eda.mask_stats(split_dir / "GT", limit=args.quick)
        _print("GT MASKS", [
            ("files", masks["n_files"]),
            ("shapes", masks["dims"]),
            ("pixel counts", masks["pixel_counts"]),
            ("images w/ label", masks["image_counts"]),
            ("foreground px", _summary(masks["foreground_pixels"])),
        ])

        labels = eda.label_distribution(split_dir)
        _print("CLASS LABELS (bits 1..8)", [
            ("files", labels["n_files"]),
            ("unique vectors", labels["n_unique_vectors"]),
            ("positives/bit", labels["bit_positives"]),
            ("zero-positive bits", labels["zero_bits"] or "none"),
        ])

        target = eda.target_analysis(split_dir)
        _print("SEG TARGET (labels 1+2)", [
            ("slices", target["n_slices"]),
            ("with target", target["target_slices"]),
            ("empty", target["empty_slices"]),
            ("label hits", target["label_hits"]),
            ("bbox area", _summary(target["bbox_area"])),
            ("mean components", target["components_mean"]),
            ("tiny/huge boxes",
             f"{target['tiny_boxes']}/{target['huge_boxes']}"),
            ("patients w/ target", target["patients_with_target"]),
        ])

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE (read-only, nothing modified)")
    print("=" * 70)


if __name__ == "__main__":
    main()
