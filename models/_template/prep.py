# ============================================================
# NEW-MODEL PREP TEMPLATE
# ============================================================
# Copy this folder to models/<your_model>/ and implement the
# three functions below. Shared logic (loading, resizing,
# bboxes, pairing, validation helpers) comes from gbc.data --
# do NOT reimplement it here.
#
# Contract:
#   build_inputs(split)  : raw split -> model-ready inputs
#                          on disk. Return (n_done, errors).
#   validate_split(split): check built inputs. Return an
#                          errors dict; all-empty means PASS.
#   passed(errors)       : True when the errors dict is clean.
# ============================================================

from gbc import data as D


# e.g. MODEL_SIZE = (224, 224); MODEL_MEAN = ...; MODEL_STD = ...

INPUT_ROOT = None   # e.g. DATASET_DIR / "<model>_preprocessed"
TARGET_ROOT = None  # e.g. DATASET_DIR / "<model>_targets"


def build_inputs(split):
    """Build model-ready inputs for one split."""
    raise NotImplementedError


def validate_split(split):
    """Validate built inputs for one split."""
    raise NotImplementedError


def passed(errors):
    """True when the errors dict is clean."""
    raise NotImplementedError
