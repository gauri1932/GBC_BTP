# ============================================================
# EfficientNet-B0 evaluation (smoke + full)
# ============================================================
# Usage (run from repo root):
#   python models/efficient_b0/evaluate.py --mode smoke
#   python models/efficient_b0/evaluate.py --mode full --split Val
#   python models/efficient_b0/evaluate.py --mode full --split Test
# Outputs:
#   models/efficient_b0/outputs/metrics_<split>.json
#   models/efficient_b0/outputs/confusion_<split>.png
#   models/efficient_b0/outputs/roc_<split>.png  (binary only)
# ============================================================

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def load_model(ckpt_path, mode):
    import timm
    n = 1 if mode == "binary" else 8
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=n)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model, ckpt


def preprocess(png_path, img_size=IMG_SIZE):
    img = Image.open(png_path).convert("L").resize((img_size, img_size), Image.BILINEAR)
    img = img.convert("RGB")
    x = np.array(img).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1)


def predict_split(data_root, split, model, device, mode, max_samples=None, img_size=IMG_SIZE):
    import csv
    split_dir = Path(data_root) / split
    with open(split_dir / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if max_samples:
        rows = rows[:max_samples]
    model.eval()
    probs, trues, names, pids = [], [], [], []
    with torch.no_grad():
        for r in rows:
            x = preprocess(split_dir / "INP" / r["filename"], img_size).unsqueeze(0).to(device)
            p = torch.sigmoid(model(x)).cpu().numpy()[0]
            probs.append(p)
            if mode == "binary":
                t = 1.0 if (int(r["bit1"]) == 1 or int(r["bit2"]) == 1) else 0.0
                trues.append([t])
            else:
                trues.append([float(r[f"bit{i}"]) for i in range(1, 9)])
            names.append(r["filename"])
            pids.append(str(r["patient_id"]))
    return np.array(trues), np.array(probs), names, pids


def save_confusion(y_true, y_pred, out_path, title="confusion"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"skip confusion fig: {e}")


def save_roc(y_true, y_prob, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot(fpr, tpr, label=f"AUC={auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"skip roc fig: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--split", default="Val")
    ap.add_argument("--data-root", default="cls_data")
    ap.add_argument("--ckpt", default="models/efficient_b0/outputs/best_efficient_b0.pth")
    ap.add_argument("--cls-mode", choices=["binary", "multilabel"], default="binary")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.ckpt, args.cls_mode)
    model = model.to(device)
    print(f"Loaded {args.ckpt} (epoch {ckpt.get('epoch', '?')}) on {device}")

    max_samples = 5 if args.mode == "smoke" else None
    split = args.split if args.mode == "full" else "Val"
    y_true, y_prob, names, pids = predict_split(
        args.data_root, split, model, device, args.cls_mode, max_samples, args.img_size)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if args.cls_mode == "binary":
        yt, yp = y_true.ravel(), y_prob.ravel()
        pred = (yp >= args.thresh).astype(int)
        metrics = {
            "split": split, "n": len(yt),
            "acc": float(accuracy_score(yt, pred)),
            "f1": float(f1_score(yt, pred, zero_division=0)),
            "thresh": args.thresh,
        }
        try:
            metrics["auc"] = float(roc_auc_score(yt, yp))
        except ValueError:
            metrics["auc"] = float("nan")
        # patient-aggregated (mean prob per patient)
        from collections import defaultdict
        bag, bag_t = defaultdict(list), {}
        for pid, t, p in zip(pids, yt, yp):
            bag[pid].append(float(p))
            bag_t[pid] = int(t)
        pt = np.array([bag_t[k] for k in bag])
        pp = np.array([np.mean(bag[k]) for k in bag])
        ppred = (pp >= args.thresh).astype(int)
        metrics["patient_n"] = len(pt)
        metrics["patient_acc"] = float(accuracy_score(pt, ppred))
        metrics["patient_f1"] = float(f1_score(pt, ppred, zero_division=0))
        save_confusion(yt, pred, OUTPUT_DIR / f"confusion_{split}.png", f"EfficientNet-B0 {split}")
        save_roc(yt, yp, OUTPUT_DIR / f"roc_{split}.png")
        if args.mode == "smoke":
            for n, t, p in zip(names, yt, yp):
                print(f"{n}: true={int(t)} prob={float(p):.3f}")
    else:
        per_c, accs, f1s, aucs = {}, [], [], []
        for c in range(8):
            yt, yp = y_true[:, c], y_prob[:, c]
            pred = (yp >= args.thresh).astype(int)
            accs.append(accuracy_score(yt, pred))
            f1s.append(f1_score(yt, pred, zero_division=0))
            try:
                aucs.append(roc_auc_score(yt, yp))
            except ValueError:
                aucs.append(float("nan"))
            per_c[f"bit{c+1}"] = {"acc": float(accs[-1]), "f1": float(f1s[-1]),
                                  "auc": float(aucs[-1]), "pos": int(yt.sum())}
        metrics = {"split": split, "n": len(y_true), "mean_acc": float(np.mean(accs)),
                   "mean_f1": float(np.mean(f1s)), "mean_auc": float(np.nanmean(aucs)),
                   "per_class": per_c}

    print(json.dumps(metrics, indent=2))
    out = OUTPUT_DIR / f"metrics_{split}.json"
    if args.mode == "full":
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved -> {out}")
    else:
        print("SMOKE OK (figures/metrics not saved except preview)")


if __name__ == "__main__":
    main()
