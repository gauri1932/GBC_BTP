# ============================================================
# LITEMEDSAM PREP (reuses the shared MedSAM-preprocessed data)
# ============================================================
#
# LiteMedSAM needs NOTHING new on disk: it trains and runs on
# the exact same files MedSAM uses --
#   medsam_preprocessed/<Split>/INP   (1024 RGB)
#   medsam_targets/<Split>/GT_binary  (1024 binary)
#   medsam_targets/<Split>/boxes.json (1024 coords, inclusive)
#
# The 1024 -> 256 downscale (+ box / 4) happens in the data
# loader at runtime, so both models always see identical data.
#
# This module therefore only exposes constants + a readiness
# check (same validator MedSAM uses).
#
# ============================================================

from models.litemedsam.model import COORD_SCALE, INPUT_SIZE
from models.medsam import prep as M

# Same roots MedSAM builds/consumes.
IMAGE_ROOT = M.IMAGE_ROOT
TARGET_ROOT = M.TARGET_ROOT

__all__ = [
    "IMAGE_ROOT",
    "TARGET_ROOT",
    "INPUT_SIZE",
    "COORD_SCALE",
    "build_inputs",
    "validate_split",
    "passed",
]


def build_inputs(split):
    """
    Nothing to build: verify the shared MedSAM data exists
    and is valid for this split.

    Returns (n_images, errors).
    """
    errors = M.validate_medsam_split(split)
    n = errors.get("n_images", 0)
    problems = (
        len(errors["image_errors"])
        + len(errors["mask_errors"])
        + errors["images_without_masks"]
        + errors["masks_without_images"]
        + errors["missing_boxes"]
        + errors["invalid_boxes"]
        + errors["out_of_range_boxes"]
        + errors["box_mask_mismatch"]
    )
    return n, problems


def validate_split(split):
    """Same full validation MedSAM uses (shared files)."""
    return M.validate_medsam_split(split)


def passed(errors):
    return M.split_passed(errors)
