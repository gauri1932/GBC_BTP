# ============================================================
# EfficientNet-B0 classification training (GBC ultrasound)
# ============================================================
# Accuracy + low compute pick to pair with LiteMedSAM:
#   EfficientNet-B0 ~5.3M params, ~0.39 GFLOPs at 224-256px.
#
# Data: cls_data/<Split>/INP/*.png (256 gray) + labels.csv
#   labels.csv cols: filename,patient_id,bit1..bit8
#   binary default: malignant = bit1 OR bit2
#   multilabel   : all 8 bits (--mode multilabel)
#
# Pipeline (matches models/_template convention):
#   1) frozen backbone + linear head (--freeze-epochs, default 2)
#   2) unfreeze last blocks + fine-tune with lower LR
#
# Recommended two-stage boost (no extra params):
#   LiteMedSAM mask -> bbox crop + 10% pad -> resize 256 -> this model.
#   (GBCNet paper: +8-12% over full-image classifier.)
#
# Usage (run from repo root):
#   pip install torch timm scikit-learn pillow tqdm numpy
#   python models/efficient_b0/train.py --data-root cls_data --quick
#   python models/efficient_b0/train.py --data-root cls_data --epochs 15 --batch-size 32
#   python models/efficient_b0/train.py --data-root cls_data --mode multilabel
#   python models/efficient_b0/train.py --smoke-test   # 5 batches, no saving
# ============================================================

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------- dataset ----------------

class GBCDataset(Dataset):
    def __init__(self, data_root, split, mode="binary", train=False, img_size=IMG_SIZE):
        self.img_dir = Path(data_root) / split / "INP"
        self.mode = mode
        self.train = train
        self.img_size = img_size
        with open(Path(data_root) / split / "labels.csv", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"empty labels.csv for split {split}")

    def __len__(self):
        return len(self.rows)

    def _label(self, r):
        if self.mode == "binary":
            y = 1.0 if (int(r["bit1"]) == 1 or int(r["bit2"]) == 1) else 0.0
            return torch.tensor([y], dtype=torch.float32)
        return torch.tensor([float(r[f"bit{i}"]) for i in range(1, 9)], dtype=torch.float32)

    def _augment(self, img):
        # light US-safe augmentation: flip + small rotation
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.25:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, resample=Image.BILINEAR)
        return img

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(self.img_dir / r["filename"]).convert("L")
        if self.train:
            img = self._augment(img)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = img.convert("RGB")  # gray -> 3ch for ImageNet weights
        x = np.array(img).astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        x = (x - mean) / std
        x = torch.from_numpy(x).permute(2, 0, 1)
        return x, self._label(r), r["filename"], str(r["patient_id"])


def build_model(mode="binary", pretrained=True):
    if not HAS_TIMM:
        raise ImportError("pip install timm  # required for efficientnet_b0")
    n = 1 if mode == "binary" else 8
    model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=n)
    return model


def set_backbone_frozen(model, frozen=True):
    # freeze everything except classifier head
    for name, p in model.named_parameters():
        if "classifier" in name or "fc" in name or "head" in name:
            p.requires_grad = True
        else:
            p.requires_grad = not frozen


# ---------------- metrics ----------------

def slice_metrics(y_true, y_prob, thresh=0.5):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y_pred = (y_prob >= thresh).astype(int)
    m = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        m["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        m["auc"] = float("nan")
    return m


def patient_metrics(filenames_ids, y_true, y_prob, thresh=0.5):
    """Mean-pool slices per patient, then score (slices are correlated)."""
    from collections import defaultdict
    bag = defaultdict(list)
    bag_true = {}
    for pid, yt, yp in zip(filenames_ids, y_true, y_prob):
        bag[pid].append(yp)
        bag_true[pid] = yt  # slices of one patient share image-level label only for binary approx
    # NOTE: for multilabel / mixed patients this is an approximation;
    # strict patient scoring needs patient-level biopsy label.
    p_true = np.array([bag_true[k] for k in bag])
    p_prob = np.array([float(np.mean(bag[k])) for k in bag])
    return slice_metrics(p_true, p_prob, thresh)


# ---------------- train ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="cls_data")
    ap.add_argument("--mode", choices=["binary", "multilabel"], default="binary")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--finetune-lr", type=float, default=3e-5)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true",
                    help="2 epochs, 400 train / 100 val samples, batch 16")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 2
        args.batch_size = 16
        print("QUICK MODE: 2 epochs, batch 16")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | mode={args.mode} | timm={HAS_TIMM}")

    train_set = GBCDataset(args.data_root, "Train", args.mode, train=True, img_size=args.img_size)
    val_set = GBCDataset(args.data_root, "Val", args.mode, train=False, img_size=args.img_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # pos_weight = neg/pos from Train (handles imbalance in bit1..8)
    with open(Path(args.data_root) / "Train" / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.mode == "binary":
        y = np.array([1 if (int(r["bit1"]) == 1 or int(r["bit2"]) == 1) else 0 for r in rows],
                     dtype=np.float64)
        pw = torch.tensor([(1 - y).sum() / max(y.sum(), 1)], dtype=torch.float32).to(device)
        print(f"Train binary pos={int(y.sum())} neg={int((1-y).sum())} pos_weight={pw.item():.3f}")
    else:
        y = np.array([[float(r[f"bit{i}"]) for i in range(1, 9)] for r in rows])
        pw = torch.tensor((1 - y).sum(0) / np.maximum(y.sum(0), 1), dtype=torch.float32).to(device)
        print(f"Train multilabel pos_per_bit={y.sum(0).tolist()}")

    model = build_model(args.mode, pretrained=not args.no_pretrained).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"EfficientNet-B0 params: {n_total:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    optimizer = None
    scheduler = None

    def make_opt(lr):
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr, weight_decay=args.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
        return opt, sch

    # stage 1: frozen backbone
    set_backbone_frozen(model, frozen=True)
    optimizer, scheduler = make_opt(args.lr)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        print(f"Resumed {args.resume}")

    if args.smoke_test:
        model.train()
        for step, (x, yb, _, _) in enumerate(train_loader):
            x, yb = x.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), yb)
            (scaler.scale(loss).backward() if use_amp else loss.backward())
            (scaler.step(optimizer), scaler.update()) if use_amp else optimizer.step()
            print(f"batch {step}: loss={loss.item():.4f}")
            if step >= 4:
                break
        print("SMOKE TEST OK (nothing saved)")
        return

    history, best_f1, start = [], -1.0, 1
    for epoch in range(start, args.epochs + 1):
        # stage 2: unfreeze after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            set_backbone_frozen(model, frozen=False)
            optimizer, scheduler = make_opt(args.finetune_lr)
            print(f"Unfroze backbone, lr={args.finetune_lr}")

        model.train()
        running = 0.0
        for x, yb, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, yb = x.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), yb)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            running += loss.item()
        train_loss = running / max(len(train_loader), 1)

        # val
        model.eval()
        all_p, all_t, all_pid = [], [], []
        with torch.no_grad():
            for x, yb, _, pid in val_loader:
                logits = model(x.to(device))
                prob = torch.sigmoid(logits).cpu().numpy()
                all_p.append(prob)
                all_t.append(yb.numpy())
                all_pid.extend(list(pid))
        all_p = np.concatenate(all_p, axis=0)
        all_t = np.concatenate(all_t, axis=0)

        if args.mode == "binary":
            m = slice_metrics(all_t.ravel(), all_p.ravel())
        else:
            ms = [slice_metrics(all_t[:, c], all_p[:, c]) for c in range(8)]
            m = {"acc": float(np.mean([v["acc"] for v in ms])),
                 "f1": float(np.mean([v["f1"] for v in ms])),
                 "auc": float(np.nanmean([v["auc"] for v in ms]))}
        scheduler.step(train_loss)
        print(f"Epoch {epoch}: loss={train_loss:.4f} val_acc={m['acc']:.4f} "
              f"val_f1={m['f1']:.4f} val_auc={m['auc']:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in m.items()}})
        torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args), **m},
                   OUTPUT_DIR / f"efficient_b0_epoch_{epoch}.pth")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": vars(args), **m},
                       OUTPUT_DIR / "best_efficient_b0.pth")
            print(f"NEW BEST (f1={best_f1:.4f})")

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val F1: {best_f1:.4f} -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
