# ============================================================
# LITEMEDSAM FINE-TUNING (on shared MedSAM-preprocessed data)
# ============================================================
#
# Data flow per slice (nothing new written to disk):
#   1024 RGB  -> resize 256 -> min-max [0,1] -> TinyViT
#   1024 mask -> resize 256 (nearest)  -> BCE+Dice loss
#   1024 box  -> /4 (+ shift jitter in training)
#
# Starts from the official lite_medsam.pth (auto-downloaded
# once into checkpoints/) unless --from-scratch is given.
# The whole 9.8M-param model is fine-tuned (nothing frozen).
#
# Usage (run from repo root):
#   python models/litemedsam/train.py
#   python models/litemedsam/train.py --epochs 20 --batch-size 16
#   python models/litemedsam/train.py --smoke-test   # 5 batches, no saving
#
# ============================================================

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.litemedsam import prep as P
from models.litemedsam.model import (
    COORD_SCALE,
    INPUT_SIZE,
    build_litemedsam,
    count_parameters,
    ensure_pretrained,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED = REPO_ROOT / "checkpoints" / "lite_medsam.pth"


# ============================================================
# DATASET
# ============================================================

class LiteGBCDataset(Dataset):
    """Target-positive slices; 1024 shared files -> 256 tensors."""

    def __init__(self, split, max_samples=None, train=False,
                 bbox_shift=5):
        self.split = split
        self.train = train
        self.bbox_shift = bbox_shift

        with open(P.TARGET_ROOT / split / "boxes.json") as f:
            boxes = json.load(f)

        self.samples = []
        for filename, info in boxes.items():
            if info.get("bbox") is None:
                continue  # no prompt possible
            img = P.IMAGE_ROOT / split / "INP" / filename
            mask = P.TARGET_ROOT / split / "GT_binary" / filename
            if img.is_file() and mask.is_file():
                self.samples.append((filename, img, mask, info["bbox"]))

        random.shuffle(self.samples)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"{split}: {len(self.samples)} positive samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, img_path, mask_path, bbox = self.samples[idx]

        image = Image.open(img_path).convert("RGB").resize(
            (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
        )
        image = np.array(image).astype(np.float32)
        image = (image - image.min()) / max(
            image.max() - image.min(), 1e-8
        )

        mask = Image.open(mask_path).convert("L").resize(
            (INPUT_SIZE, INPUT_SIZE), Image.Resampling.NEAREST
        )
        mask = (np.array(mask) > 0).astype(np.float32)

        x_min, y_min, x_max, y_max = [v * COORD_SCALE for v in bbox]

        if self.train and self.bbox_shift > 0:
            s = self.bbox_shift
            x_min = max(0.0, x_min - random.randint(0, s))
            y_min = max(0.0, y_min - random.randint(0, s))
            x_max = min(INPUT_SIZE - 1, x_max + random.randint(0, s))
            y_max = min(INPUT_SIZE - 1, y_max + random.randint(0, s))

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        box = torch.tensor(
            [[x_min, y_min, x_max, y_max]], dtype=torch.float32
        )

        return image, mask, box, filename


# ============================================================
# LOSS (BCE + Dice, same recipe as models/medsam/train.py)
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target, smooth=1.0):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    dice = (2 * inter + smooth) / (
        pred.sum() + target.sum() + smooth
    )
    return 1 - dice


def combined_loss(pred, target):
    return bce_loss(pred, target) + dice_loss(pred, target)


# ============================================================
# METRICS (computed at 256, fast)
# ============================================================

def dice_iou(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    union = torch.logical_or(pred.bool(), target.bool()).sum().item()
    dice = 1.0 if denom == 0 else 2 * inter / denom
    iou = 1.0 if union == 0 else inter / union
    return dice, iou


# ============================================================
# TRAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bbox-shift", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--pretrained", default=str(PRETRAINED),
                        help="lite_medsam.pth path (auto-downloaded).")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a few batches only, save nothing.")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    train_set = LiteGBCDataset(
        "Train", max_samples=args.max_train,
        train=True, bbox_shift=args.bbox_shift,
    )
    val_set = LiteGBCDataset("Val", max_samples=args.max_val)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    if args.from_scratch:
        print("Training from scratch (no pretrained weights).")
        model = build_litemedsam()
    else:
        ckpt = ensure_pretrained(args.pretrained)
        model = build_litemedsam(checkpoint=ckpt)

    model = model.to(device)

    total, trainable = count_parameters(model)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")

    # --------------------------------------------------------
    # Optimizer (official LiteMedSAM settings)
    # --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=5
    )

    start_epoch, best_dice = 1, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        raw = state["model"] if "model" in state else state
        model.load_state_dict(raw, strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_dice = state.get("best_dice", -1.0)
        print(f"Resumed from epoch {state['epoch']}")

    # --------------------------------------------------------
    # Smoke test: a few batches, then exit
    # --------------------------------------------------------
    if args.smoke_test:
        model.train()
        for step, (image, mask, box, _) in enumerate(train_loader):
            image, mask, box = (
                image.to(device), mask.to(device), box.to(device)
            )
            logits, _ = model(image, box)
            loss = combined_loss(logits, mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"batch {step}: loss={loss.item():.4f} "
                  f"logits {tuple(logits.shape)}")
            if step >= 4:
                break
        print("SMOKE TEST OK (no checkpoints saved)")
        return

    # --------------------------------------------------------
    # Epochs
    # --------------------------------------------------------
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0

        for image, mask, box, _ in tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}"
        ):
            image = image.to(device)
            mask = mask.to(device)
            box = box.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(image, box)
            loss = combined_loss(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)

        # ---------------- validation ----------------
        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for image, mask, box, _ in val_loader:
                image = image.to(device)
                mask = mask.to(device)
                box = box.to(device)
                logits, _ = model(image, box)
                for l, m in zip(logits, mask):
                    d, i = dice_iou(
                        l.unsqueeze(0), m.unsqueeze(0)
                    )
                    dices.append(d)
                    ious.append(i)

        mean_dice = float(np.mean(dices)) if dices else 0.0
        mean_iou = float(np.mean(ious)) if ious else 0.0
        scheduler.step(train_loss)

        print(f"Epoch {epoch}: loss={train_loss:.4f} "
              f"val_dice={mean_dice:.4f} val_iou={mean_iou:.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_dice": mean_dice, "val_iou": mean_iou,
            "lr": optimizer.param_groups[0]["lr"],
        })

        torch.save(
            {"epoch": epoch,
             "model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "val_dice": mean_dice, "val_iou": mean_iou},
            OUTPUT_DIR / f"litemedsam_epoch_{epoch}.pth",
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(
                {"epoch": epoch,
                 "model": model.state_dict(),
                 "val_dice": mean_dice, "val_iou": mean_iou},
                OUTPUT_DIR / "best_litemedsam.pth",
            )
            print(f"NEW BEST (dice={mean_dice:.4f})")

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"Done. Best val dice: {best_dice:.4f} -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
