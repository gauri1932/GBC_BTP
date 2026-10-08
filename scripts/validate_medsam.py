# ============================================================
# VALIDATE MEDSAM DATASET (merges old stage14 + stage16)
# ============================================================
#
# Single validation pass over medsam_preprocessed/ +
# medsam_targets/, checking everything both old stages
# checked (and the box==mask exactness that stage15 fixed):
#   - images: 1024x1024 RGB, readable
#   - masks : 1024x1024, values only {0, 255}
#   - image/mask filename pairing
#   - boxes.json: present, in range, bbox None <-> empty mask,
#                 bbox EXACTLY equal to mask bbox
#
# Usage:
#   python validate_medsam_dataset.py
#
# ============================================================

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gbc import data as P
from models.medsam import prep as M


def main():
    print("=" * 70)
    print("VALIDATE MEDSAM DATASET (stages 14+16 merged)")
    print("=" * 70)

    all_passed = True

    for split_name, split_dir in P.SPLIT_DIRS.items():
        print("\n" + "=" * 70)
        print(split_name.upper())
        print("=" * 70)

        errors = M.validate_medsam_split(
            split_name, check_pairing_with=split_dir
        )

        print(f"\nImages : {errors['n_images']}")
        print(f"Masks  : {errors['n_masks']}")
        print(f"Boxes  : {errors['n_boxes']}")

        print(f"\nImage file errors    : {len(errors['image_errors'])}")
        for name, msg in errors["image_errors"][:10]:
            print(f"  {name}: {msg}")

        print(f"Mask file errors     : {len(errors['mask_errors'])}")
        for name, msg in errors["mask_errors"][:10]:
            print(f"  {name}: {msg}")

        print(
            f"Images without masks : "
            f"{errors['images_without_masks']}"
        )
        print(
            f"Masks without images : "
            f"{errors['masks_without_images']}"
        )
        print(f"Missing boxes        : {errors['missing_boxes']}")
        print(f"Invalid boxes        : {errors['invalid_boxes']}")
        print(
            f"Out-of-range boxes   : "
            f"{errors['out_of_range_boxes']}"
        )
        print(
            f"Box/mask mismatches  : "
            f"{errors['box_mask_mismatch']}"
        )

        if M.split_passed(errors):
            print(f"\n[SPLIT {split_name}] PASS")
        else:
            print(f"\n[SPLIT {split_name}] FAIL")
            all_passed = False

    print("\n" + "=" * 70)
    print("VALIDATION PASSED" if all_passed else "VALIDATION FOUND ISSUES")
    print("=" * 70)


if __name__ == "__main__":
    main()
