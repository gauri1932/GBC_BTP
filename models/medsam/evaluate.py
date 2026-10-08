# ============================================================
# MEDSAM EVALUATION (merges old stage17 + stage18)
# ============================================================
#
# Two modes, one shared inference core (identical math to the
# old stages, so published baselines still reproduce):
#
#   smoke : N positive slices + 4-panel figures (old stage17)
#   full  : all positive slices + metrics + summary (old stage18)
#
# Usage (run from repo root):
#   python models/medsam/evaluate.py --mode smoke --n 5
#   python models/medsam/evaluate.py --mode full --split Val
#   python models/medsam/evaluate.py --mode full \
#       --checkpoint finetuned
#
# ============================================================

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from segment_anything import sam_model_registry

from models.medsam import prep as M

def _first_existing(*paths):
    for p in paths:
        if p.is_file():
            return p
    return paths[0]


CHECKPOINTS = {
    "pretrained": REPO_ROOT / "checkpoints" / "medsam_vit_b.pth",
    # New finetunes land in models/medsam/outputs/; fall back to the
    # historic stage19_results/ location.
    "finetuned": _first_existing(
        Path(__file__).resolve().parent / "outputs" / "best_medsam.pth",
        REPO_ROOT / "stage19_results" / "best_medsam.pth",
    ),
}

OUTPUT_BASE = Path(__file__).resolve().parent / "outputs"

IMAGE_SIZE = 1024


# ============================================================
# SHARED CORE (identical to old stage17/stage18 math)
# ============================================================

def load_model(checkpoint, device):
    print("Loading MedSAM...")
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model = model.to(device)
    model.eval()
    print("MedSAM loaded")
    return model


@torch.no_grad()
def predict_with_box(model, image_np, bbox, device):
    """
    Box-prompted binary prediction at 1024x1024.

    image_np : HxWx3 uint8 (already MedSAM resolution).
    bbox     : [x_min, y_min, x_max, y_max], 1024 coords.
    """
    if image_np.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Unexpected image shape: {image_np.shape}")

    image_float = image_np.astype(np.float32)
    image_float = (image_float - image_float.min()) / max(
        image_float.max() - image_float.min(), 1e-8
    )

    image_tensor = (
        torch.from_numpy(image_float)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )

    box_tensor = torch.as_tensor(
        [bbox], dtype=torch.float32, device=device
    )[:, None, :]

    image_embedding = model.image_encoder(image_tensor)

    sparse, dense = model.prompt_encoder(
        points=None, boxes=box_tensor, masks=None
    )

    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )

    prob = torch.sigmoid(
        F.interpolate(
            low_res_logits,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    )

    return (prob.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def calculate_metrics(pred, gt):
    """Same fields as the old stage18 results JSON."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = int(np.logical_and(pred, gt).sum())
    pred_pixels = int(pred.sum())
    gt_pixels = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())

    dice = (
        1.0
        if pred_pixels + gt_pixels == 0
        else 2.0 * intersection / (pred_pixels + gt_pixels)
    )
    iou = 1.0 if union == 0 else intersection / union
    precision = (
        (1.0 if gt_pixels == 0 else 0.0)
        if pred_pixels == 0
        else intersection / pred_pixels
    )
    recall = (
        (1.0 if pred_pixels == 0 else 0.0)
        if gt_pixels == 0
        else intersection / gt_pixels
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "intersection": intersection,
    }


def positive_samples(split):
    """Sorted (filename, bbox) list of target-positive slices."""
    with open(M.TARGET_ROOT / split / "boxes.json") as f:
        boxes = json.load(f)

    samples = [
        (name, info["bbox"])
        for name, info in boxes.items()
        if info.get("bbox") is not None
    ]
    samples.sort(key=lambda x: x[0])

    with open(M.TARGET_ROOT / split / "boxes.json") as f:
        n_empty = sum(
            1 for info in json.load(f).values()
            if info.get("bbox") is None
        )

    return samples, len(boxes), n_empty


def load_pair(split, filename):
    image = np.array(
        Image.open(M.IMAGE_ROOT / split / "INP" / filename).convert("RGB")
    )
    gt = (
        np.array(Image.open(M.TARGET_ROOT / split / "GT_binary" / filename))
        > 0
    ).astype(np.uint8)
    return image, gt


# ============================================================
# SMOKE MODE (old stage17: figures + small results JSON)
# ============================================================

def save_figure(image, gt, pred, bbox, out_path, dice, iou):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(gt, alpha=0.5)
    x1, y1, x2, y2 = bbox
    axes[1].add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                      fill=False, linewidth=2)
    )
    axes[1].set_title("Ground Truth + BBox")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(pred, alpha=0.5)
    axes[2].add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                      fill=False, linewidth=2)
    )
    axes[2].set_title("MedSAM Prediction")
    axes[2].axis("off")

    comparison = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    gt_bool, pred_bool = gt > 0, pred > 0
    comparison[gt_bool & pred_bool] = [255, 255, 255]      # TP
    comparison[gt_bool & (~pred_bool)] = [255, 0, 0]       # FN
    comparison[(~gt_bool) & pred_bool] = [0, 255, 0]       # FP
    axes[3].imshow(comparison)
    axes[3].set_title(f"Comparison\nDice={dice:.4f} | IoU={iou:.4f}")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_smoke(split, checkpoint, n, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint, device)

    samples, _, _ = positive_samples(split)
    selected = samples[:n]
    print(f"Positive available: {len(samples)}, selected: {len(selected)}")

    results = []
    for filename, bbox in selected:
        image, gt = load_pair(split, filename)
        pred = predict_with_box(model, image, bbox, device)
        m = calculate_metrics(pred, gt)
        print(f"{filename}: Dice={m['dice']:.4f} IoU={m['iou']:.4f}")

        save_figure(image, gt, pred, bbox,
                    out_dir / f"result_{filename}", m["dice"], m["iou"])

        results.append({
            "filename": filename,
            "bbox": bbox,
            "gt_pixels": m["gt_pixels"],
            "prediction_pixels": m["pred_pixels"],
            "dice": m["dice"],
            "iou": m["iou"],
        })

    with open(out_dir / "smoke_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Mean Dice: {np.mean([r['dice'] for r in results]):.4f}")
    print(f"Figures + JSON in {out_dir}")


# ============================================================
# FULL MODE (old stage18: all positives + summary JSON)
# ============================================================

def run_full(split, checkpoint, out_dir, device):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(checkpoint, device)
    samples, n_total, n_empty = positive_samples(split)
    print(f"Total: {n_total}, positive: {len(samples)}, empty: {n_empty}")

    results = []
    start = time.time()

    for filename, bbox in tqdm(samples, desc="MedSAM validation"):
        image, gt = load_pair(split, filename)
        pred = predict_with_box(model, image, bbox, device)

        m = calculate_metrics(pred, gt)
        m["filename"] = filename
        m["bbox"] = bbox
        results.append(m)

        Image.fromarray((pred * 255).astype(np.uint8)).save(
            pred_dir / filename
        )

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    dice = np.array([r["dice"] for r in results])
    iou = np.array([r["iou"] for r in results])

    summary = {
        "dataset": "GBC",
        "split": split,
        "total_validation_slices": n_total,
        "positive_target_slices": len(samples),
        "empty_target_slices": n_empty,
        "evaluated_slices": len(results),
        "mean_dice": float(dice.mean()),
        "median_dice": float(np.median(dice)),
        "std_dice": float(dice.std()),
        "mean_iou": float(iou.mean()),
        "median_iou": float(np.median(iou)),
        "std_iou": float(iou.std()),
        "mean_precision": float(
            np.mean([r["precision"] for r in results])
        ),
        "mean_recall": float(
            np.mean([r["recall"] for r in results])
        ),
        "min_dice": float(dice.min()),
        "max_dice": float(dice.max()),
        "runtime_seconds": time.time() - start,
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Mean Dice: {summary['mean_dice']:.4f} "
          f"IoU: {summary['mean_iou']:.4f}")
    print(f"Results in {out_dir}")


# ============================================================
# CLI
# ============================================================

def resolve_checkpoint(name_or_path):
    if name_or_path in CHECKPOINTS:
        path = CHECKPOINTS[name_or_path]
    else:
        path = Path(name_or_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found:\n{path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"],
                        default="smoke")
    parser.add_argument("--split", default=None,
                        help="Default: Train for smoke, Val for full.")
    parser.add_argument("--checkpoint", default="pretrained",
                        help="'pretrained', 'finetuned', or a .pth path.")
    parser.add_argument("--n", type=int, default=5,
                        help="Smoke mode sample count.")
    parser.add_argument("--out", default=None,
                        help="Output dir (default under models/medsam/outputs).")
    args = parser.parse_args()

    split = args.split or ("Train" if args.mode == "smoke" else "Val")
    out_dir = (
        Path(args.out)
        if args.out
        else OUTPUT_BASE / f"eval_{args.mode}_{split.lower()}"
    )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    checkpoint = resolve_checkpoint(args.checkpoint)

    if args.mode == "smoke":
        run_smoke(split, checkpoint, args.n, out_dir, device)
    else:
        run_full(split, checkpoint, out_dir, device)


if __name__ == "__main__":
    main()
