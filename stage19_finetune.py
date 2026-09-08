import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry


# ============================================================
# STAGE 19 - MEDSAM FINE-TUNING
# ============================================================

ROOT = r"C:\Users\Student\Downloads\GBC"

TRAIN_IMG = os.path.join(
    ROOT, "GBC_dataset", "medsam_preprocessed", "Train", "INP"
)

TRAIN_MASK = os.path.join(
    ROOT, "GBC_dataset", "medsam_targets", "Train", "GT_binary"
)

TRAIN_BOX = os.path.join(
    ROOT, "GBC_dataset", "medsam_targets", "Train", "boxes.json"
)

VAL_IMG = os.path.join(
    ROOT, "GBC_dataset", "medsam_preprocessed", "Val", "INP"
)

VAL_MASK = os.path.join(
    ROOT, "GBC_dataset", "medsam_targets", "Val", "GT_binary"
)

VAL_BOX = os.path.join(
    ROOT, "GBC_dataset", "medsam_targets", "Val", "boxes.json"
)

CHECKPOINT = os.path.join(
    ROOT, "checkpoints", "medsam_vit_b.pth"
)

OUTPUT = os.path.join(ROOT, "stage19_results")
os.makedirs(OUTPUT, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FAST RESEARCH RUN
EPOCHS = 3
BATCH_SIZE = 1
LR = 1e-5

# To keep the 2-hour window manageable.
MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 500


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


print("=" * 70)
print("STAGE 19 - MEDSAM FINE-TUNING")
print("=" * 70)

print("\nDEVICE")
print("-" * 70)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DATASET
# ============================================================

class MedSAMDataset(Dataset):

    def __init__(self, image_dir, mask_dir, box_json, max_samples=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        with open(box_json, "r") as f:
            self.boxes = json.load(f)

        self.samples = []

        for filename, info in self.boxes.items():

            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            if not os.path.exists(image_path):
                continue

            if not os.path.exists(mask_path):
                continue

            # For training, skip empty masks because
            # there is no bounding-box prompt.
            if info["bbox"] is None:
                continue

            self.samples.append(
                (filename, image_path, mask_path, info["bbox"])
            )

        random.shuffle(self.samples)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(
            f"Dataset: {len(self.samples)} positive samples"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        filename, image_path, mask_path, bbox = self.samples[idx]

        image = np.array(
            Image.open(image_path).convert("RGB")
        )

        mask = np.array(
            Image.open(mask_path).convert("L")
        )

        image = torch.from_numpy(
            image.astype(np.float32)
        ).permute(2, 0, 1)

        # Normalize to [0,1]
        image = image / 255.0

        mask = (mask > 0).astype(np.float32)

        mask = torch.from_numpy(mask).unsqueeze(0)

        bbox = torch.tensor(
            bbox,
            dtype=torch.float32
        )

        return image, mask, bbox, filename


# ============================================================
# LOAD DATA
# ============================================================

print("\nLOADING DATA")
print("-" * 70)

train_dataset = MedSAMDataset(
    TRAIN_IMG,
    TRAIN_MASK,
    TRAIN_BOX,
    MAX_TRAIN_SAMPLES
)

val_dataset = MedSAMDataset(
    VAL_IMG,
    VAL_MASK,
    VAL_BOX,
    MAX_VAL_SAMPLES
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MEDSAM
# ============================================================

print("\nLOADING MEDSAM")
print("-" * 70)

sam = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT
)

sam = sam.to(DEVICE)

print("✓ MedSAM loaded")


# ============================================================
# IMPORTANT
# ============================================================
#
# MedSAM uses:
#
# image encoder
#      ↓
# image embeddings
#      ↓
# prompt encoder
#      ↓
# mask decoder
#
# For fast fine-tuning we train the MASK DECODER.
#
# The pretrained image encoder remains frozen.
#
# ============================================================


for param in sam.image_encoder.parameters():
    param.requires_grad = False

for param in sam.prompt_encoder.parameters():
    param.requires_grad = False

for param in sam.mask_decoder.parameters():
    param.requires_grad = True


trainable = sum(
    p.numel()
    for p in sam.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in sam.parameters()
)

print("\nMODEL")
print("-" * 70)
print(f"Total parameters     : {total:,}")
print(f"Trainable parameters : {trainable:,}")


# ============================================================
# LOSS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(pred, target, smooth=1.0):

    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        pred.sum() + target.sum() + smooth
    )

    return 1 - dice


def combined_loss(pred, target):

    bce = bce_loss(pred, target)

    dice = dice_loss(pred, target)

    return bce + dice


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    sam.mask_decoder.parameters(),
    lr=LR,
    weight_decay=1e-4
)


# ============================================================
# METRICS
# ============================================================

def calculate_dice(pred, target):

    pred = pred > 0.5
    target = target > 0.5

    intersection = np.logical_and(
        pred, target
    ).sum()

    denominator = pred.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection
        / denominator
    )


def calculate_iou(pred, target):

    pred = pred > 0.5
    target = target > 0.5

    intersection = np.logical_and(
        pred, target
    ).sum()

    union = np.logical_or(
        pred, target
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# FORWARD PASS
# ============================================================

def predict_mask(image, bbox):

    # SAM expects image encoder input at 1024x1024.
    #
    # Images are already 1024x1024.
    #

    image_embeddings = sam.image_encoder(image)

    sparse_embeddings, dense_embeddings = (
        sam.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None
        )
    )

    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False
    )

    masks = torch.nn.functional.interpolate(
        low_res_masks,
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False
    )

    return masks


# ============================================================
# TRAINING
# ============================================================

history = []

best_dice = -1

print("\n")
print("=" * 70)
print("STARTING FINE-TUNING")
print("=" * 70)

for epoch in range(1, EPOCHS + 1):

    sam.train()

    # Keep frozen modules in eval mode.
    sam.image_encoder.eval()
    sam.prompt_encoder.eval()

    running_loss = 0.0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{EPOCHS}"
    )

    for image, mask, bbox, filename in progress:

        image = image.to(
            DEVICE,
            non_blocking=True
        )

        mask = mask.to(
            DEVICE,
            non_blocking=True
        )

        bbox = bbox.to(
            DEVICE,
            non_blocking=True
        )

        # SAM image encoder expects approximately
        # ImageNet-normalized input.
        image = (
            image - torch.tensor(
                [0.485, 0.456, 0.406],
                device=DEVICE
            ).view(1, 3, 1, 1)
        ) / torch.tensor(
            [0.229, 0.224, 0.225],
            device=DEVICE
        ).view(1, 3, 1, 1)

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():

            image_embeddings = (
                sam.image_encoder(image)
            )

            sparse_embeddings, dense_embeddings = (
                sam.prompt_encoder(
                    points=None,
                    boxes=bbox,
                    masks=None
                )
            )

        low_res_masks, _ = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        pred = torch.nn.functional.interpolate(
            low_res_masks,
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False
        )

        loss = combined_loss(
            pred,
            mask
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            sam.mask_decoder.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = (
        running_loss / len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    sam.eval()

    dice_scores = []
    iou_scores = []

    with torch.no_grad():

        for image, mask, bbox, filename in tqdm(
            val_loader,
            desc="Validation"
        ):

            image = image.to(DEVICE)
            mask = mask.to(DEVICE)
            bbox = bbox.to(DEVICE)

            image = (
                image - torch.tensor(
                    [0.485, 0.456, 0.406],
                    device=DEVICE
                ).view(1, 3, 1, 1)
            ) / torch.tensor(
                [0.229, 0.224, 0.225],
                device=DEVICE
            ).view(1, 3, 1, 1)

            pred = predict_mask(
                image,
                bbox
            )

            pred = torch.sigmoid(
                pred
            )

            pred_np = (
                pred[0, 0]
                .cpu()
                .numpy()
            )

            mask_np = (
                mask[0, 0]
                .cpu()
                .numpy()
            )

            dice_scores.append(
                calculate_dice(
                    pred_np,
                    mask_np
                )
            )

            iou_scores.append(
                calculate_iou(
                    pred_np,
                    mask_np
                )
            )


    mean_dice = float(
        np.mean(dice_scores)
    )

    mean_iou = float(
        np.mean(iou_scores)
    )


    print("\n")
    print("=" * 70)
    print(f"EPOCH {epoch} RESULTS")
    print("=" * 70)

    print(f"Training Loss : {avg_loss:.6f}")
    print(f"Validation Dice : {mean_dice:.6f}")
    print(f"Validation IoU  : {mean_iou:.6f}")


    record = {
        "epoch": epoch,
        "train_loss": avg_loss,
        "val_dice": mean_dice,
        "val_iou": mean_iou
    }

    history.append(record)


    # ========================================================
    # SAVE EVERY EPOCH
    # ========================================================

    checkpoint_path = os.path.join(
        OUTPUT,
        f"medsam_epoch_{epoch}.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": sam.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": mean_dice,
            "val_iou": mean_iou
        },
        checkpoint_path
    )

    print(
        f"Checkpoint saved: {checkpoint_path}"
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    if mean_dice > best_dice:

        best_dice = mean_dice

        best_path = os.path.join(
            OUTPUT,
            "best_medsam.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": sam.state_dict(),
                "val_dice": mean_dice,
                "val_iou": mean_iou
            },
            best_path
        )

        print(
            f"✓ NEW BEST MODEL"
        )
        print(
            f"  Dice = {mean_dice:.6f}"
        )


# ============================================================
# SAVE HISTORY
# ============================================================

history_path = os.path.join(
    OUTPUT,
    "training_history.json"
)

with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 19 COMPLETE")
print("=" * 70)

print(f"Epochs completed : {EPOCHS}")
print(f"Best Val Dice    : {best_dice:.6f}")

print("\nOutputs:")
print(
    f"Checkpoints : {OUTPUT}"
)

print(
    f"History     : {history_path}"
)

print(
    f"Best model  : {os.path.join(OUTPUT, 'best_medsam.pth')}"
)

print("=" * 70)
