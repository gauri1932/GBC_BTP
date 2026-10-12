# ============================================================
# EfficientNet-B0 prep (GBC_BTP compatible, standalone-safe)
# ============================================================
# Works two ways:
#   1) Inside GBC_BTP repo  -> reuses `gbc.data / gbc.config`
#   2) Standalone (this folder) -> reads cls_data/ directly
#
# Expected input (built by scripts/prepare_cls_data.py):
#   cls_data/<Split>/INP/*.png        256px grayscale
#   cls_data/<Split>/labels.csv       filename,patient_id,bit1..bit8
#
# Binary target (default): malignant = bit1 OR bit2
#   (mass + wall thickening, cf. gbc/config.py TARGET_LABELS={1,2})
# Multilabel target: --mode multilabel uses all 8 bits.
#
# Usage (run from repo root):
#   python models/efficient_b0/prep.py --data-root cls_data
# ============================================================

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from gbc import data as P  # type: ignore
    HAS_GBC = True
except Exception:
    HAS_GBC = False

SPLITS = ("Train", "Val", "Test")
CLASS_BITS = [f"bit{i}" for i in range(1, 9)]


def read_labels_csv(csv_path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def to_binary_label(row):
    """Malignant if bit1==1 or bit2==1."""
    return int(int(row["bit1"]) == 1 or int(row["bit2"]) == 1)


def to_multilabel(row):
    return np.array([int(row[f"bit{i}"]) for i in range(1, 9)], dtype=np.int64)


def validate_split(data_root, split):
    """Check INP/*.png <-> labels.csv pairing + patient leakage helper."""
    split_dir = Path(data_root) / split
    inp_dir = split_dir / "INP"
    csv_path = split_dir / "labels.csv"
    report = {"split": split, "ok": False, "errors": []}

    if not inp_dir.is_dir():
        report["errors"].append(f"missing dir: {inp_dir}")
        return report
    if not csv_path.is_file():
        report["errors"].append(f"missing file: {csv_path}")
        return report

    rows = read_labels_csv(csv_path)
    csv_names = {r["filename"] for r in rows}
    disk_names = {p.name for p in inp_dir.glob("*.png")}

    missing_on_disk = sorted(csv_names - disk_names)
    missing_in_csv = sorted(disk_names - csv_names)
    report.update({
        "n_csv": len(rows),
        "n_png": len(disk_names),
        "missing_on_disk": len(missing_on_disk),
        "missing_in_csv": len(missing_in_csv),
    })
    if missing_on_disk[:5]:
        report["errors"].append(f"in csv not on disk: {missing_on_disk[:5]}")
    if missing_in_csv[:5]:
        report["errors"].append(f"on disk not in csv: {missing_in_csv[:5]}")

    # class balance preview (needed for pos_weight)
    if rows:
        y_bin = np.array([to_binary_label(r) for r in rows])
        y_ml = np.array([to_multilabel(r) for r in rows])
        report["n_positive_binary"] = int(y_bin.sum())
        report["n_negative_binary"] = int((1 - y_bin).sum())
        report["positives_per_bit1_8"] = y_ml.sum(axis=0).tolist()

    report["ok"] = len(report["errors"]) == 0
    return report


def check_patient_leakage(data_root):
    """Train/Val/Test must not share patient_id."""
    ids = {}
    for s in SPLITS:
        csv_path = Path(data_root) / s / "labels.csv"
        if not csv_path.is_file():
            continue
        ids[s] = {r["patient_id"] for r in read_labels_csv(csv_path)}
    leaks = {}
    keys = list(ids)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            overlap = ids[keys[i]] & ids[keys[j]]
            if overlap:
                leaks[f"{keys[i]}<->{keys[j]}"] = sorted(overlap)[:10]
    return {"patient_counts": {k: len(v) for k, v in ids.items()}, "leaks": leaks}


def compute_pos_weight(data_root, mode="binary"):
    """pos_weight for BCEWithLogitsLoss = neg/pos per class."""
    rows = read_labels_csv(Path(data_root) / "Train" / "labels.csv")
    if mode == "binary":
        y = np.array([to_binary_label(r) for r in rows], dtype=np.float64)
        neg, pos = (1 - y).sum(), y.sum()
        w = float(neg / max(pos, 1))
        return [w]
    y = np.array([to_multilabel(r) for r in rows], dtype=np.float64)
    neg = (1 - y).sum(axis=0)
    pos = y.sum(axis=0)
    return (neg / np.maximum(pos, 1)).tolist()


def build_inputs(data_root, out_path=None):
    """Validate all splits + leakage; write summary json (read-only otherwise)."""
    data_root = Path(data_root)
    reports = [validate_split(data_root, s) for s in SPLITS]
    leakage = check_patient_leakage(data_root)
    passed = all(r["ok"] for r in reports) and len(leakage["leaks"]) == 0
    summary = {
        "data_root": str(data_root),
        "has_gbc_lib": HAS_GBC,
        "splits": reports,
        "leakage": leakage,
        "pos_weight_binary": compute_pos_weight(data_root, "binary"),
        "pos_weight_multilabel": compute_pos_weight(data_root, "multilabel"),
        "passed": passed,
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
    return summary


def passed(summary):
    return bool(summary.get("passed", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="cls_data")
    ap.add_argument("--out", default="models/efficient_b0/outputs/prep_summary.json")
    args = ap.parse_args()

    summary = build_inputs(args.data_root, args.out)
    print(json.dumps(summary, indent=2))
    print("\nPREP %s -> %s" % ("PASS" if passed(summary) else "FAIL", args.out))


if __name__ == "__main__":
    main()
