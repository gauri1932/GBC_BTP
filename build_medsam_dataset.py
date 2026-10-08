# ============================================================
# BUILD MEDSAM DATASET (merges old stage12 + stage13 + stage15)
# ============================================================
#
# One script builds everything MedSAM needs:
#   1. images : INP 512 gray  -> 1024 RGB  (medsam_preprocessed/)
#   2. targets: GT 512 multic -> 1024 binary + boxes.json
#                                    (medsam_targets/)
#
# The old stage15 box fix is baked in: bboxes are computed
# directly on the final 1024 masks, so no fix-up pass exists.
#
# Original dataset is NEVER modified.
# Re-running is safe (outputs are overwritten in place).
#
# Usage:
#   python build_medsam_dataset.py            # all splits
#   python build_medsam_dataset.py --split Val
#
# ============================================================

import argparse

from tqdm import tqdm

import gbc_preprocess as P
import gbc_medsam as M


def build_split(split_name):
    print("\n" + "=" * 70)
    print(f"BUILDING {split_name.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. IMAGES
    # --------------------------------------------------------
    src_dir = P.SPLIT_DIRS[split_name] / "INP"
    files = P.list_files(src_dir, ".png")

    print(f"\n[1/2] Images: {len(files)} slices -> 1024 RGB")

    processed, errors = M.build_medsam_images(split_name)

    print(f"  processed: {processed}, errors: {len(errors)}")
    for name, msg in errors[:10]:
        print(f"  ERROR {name}: {msg}")

    # --------------------------------------------------------
    # 2. TARGETS
    # --------------------------------------------------------
    print(f"\n[2/2] Targets: labels {sorted(M.TARGET_LABELS)} -> binary + boxes")

    stats = M.build_medsam_targets(split_name)

    print(f"  total         : {stats['total']}")
    print(f"  target present: {stats['target_present']}")
    print(f"  target absent : {stats['target_absent']}")
    print(f"  errors        : {stats['errors']}")
    print(f"  boxes.json    : {stats['boxes_json']}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=list(P.SPLIT_DIRS),
        default=None,
        help="Build a single split (default: all).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("BUILD MEDSAM DATASET (stages 12+13+15 merged)")
    print("=" * 70)
    print("\nTarget labels: 1 = Mass, 2 = Wall Thickening")
    print("Original dataset will NOT be modified.")

    splits = [args.split] if args.split else list(P.SPLIT_DIRS)

    for split_name in tqdm(splits, desc="Splits"):
        split_dir = P.SPLIT_DIRS[split_name]
        if not (split_dir / "INP").exists():
            print(f"\nWARNING: missing {split_dir / 'INP'}, skipped.")
            continue
        build_split(split_name)

    print("\n" + "=" * 70)
    print("BUILD COMPLETE -> run validate_medsam_dataset.py next")
    print("=" * 70)


if __name__ == "__main__":
    main()
