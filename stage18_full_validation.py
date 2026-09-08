import os
import json
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

from segment_anything import sam_model_registry


# ============================================================
# STAGE 18 - FULL MEDSAM VALIDATION BASELINE
# ============================================================

ROOT = r"C:\Users\Student\Downloads\GBC"

IMAGE_DIR = os.path.join(
    ROOT,
    "GBC_dataset",
    "medsam_preprocessed",
    "Val",
    "INP"
)

MASK_DIR = os.path.join(
    ROOT,
    "GBC_dataset",
    "medsam_targets",
    "Val",
    "GT_binary"
)

BOX_FILE = os.path.join(
    ROOT,
    "GBC_dataset",
    "medsam_targets",
    "Val",
    "boxes.json"
)

CHECKPOINT = os.path.join(
    ROOT,
    "checkpoints",
    "medsam_vit_b.pth"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "stage18_results"
)

PRED_DIR = os.path.join(
    OUTPUT_DIR,
    "predictions"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("STAGE 18 - FULL MEDSAM VALIDATION BASELINE")
print("=" * 70)

print("\nDEVICE")
print("-" * 70)
print("PyTorch CUDA available :", torch.cuda.is_available())
print("Device                  :", device)

if torch.cuda.is_available():
    print(
        "GPU                     :",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# PATH VALIDATION
# ============================================================

print("\nPATH VALIDATION")
print("-" * 70)

paths = {
    "Images": IMAGE_DIR,
    "Masks": MASK_DIR,
    "Boxes": BOX_FILE,
    "Checkpoint": CHECKPOINT
}

for name, path in paths.items():

    exists = os.path.exists(path)

    print(
        f"{name:<12}:",
        "✓" if exists else "✗",
        path
    )

    if not exists:
        raise FileNotFoundError(path)


# ============================================================
# LOAD BOXES
# ============================================================

print("\nLOADING VALIDATION BOXES")
print("-" * 70)

with open(BOX_FILE, "r") as f:
    boxes = json.load(f)

print("Total JSON entries :", len(boxes))

positive_files = [
    name
    for name, data in boxes.items()
    if data.get("bbox") is not None
]

empty_files = [
    name
    for name, data in boxes.items()
    if data.get("bbox") is None
]

print("Positive samples    :", len(positive_files))
print("Empty targets       :", len(empty_files))


# ============================================================
# LOAD MEDSAM
# ============================================================

print("\nLOADING MEDSAM")
print("-" * 70)

print("Loading ViT-B checkpoint...")

medsam_model = sam_model_registry["vit_b"](
    checkpoint=CHECKPOINT
)

medsam_model = medsam_model.to(device)
medsam_model.eval()

print("✓ MedSAM loaded")


# ============================================================
# MEDSAM INFERENCE
# ============================================================

@torch.no_grad()
def medsam_inference(
    model,
    image_tensor,
    box,
    H,
    W
):

    # --------------------------------------------------------
    # Image embedding
    # --------------------------------------------------------

    image_embedding = model.image_encoder(
        image_tensor
    )

    # --------------------------------------------------------
    # Bounding box prompt
    # --------------------------------------------------------

    box_torch = torch.as_tensor(
        [box],
        dtype=torch.float32,
        device=device
    )

    box_torch = box_torch[:, None, :]

    sparse_embeddings, dense_embeddings = (
        model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None
        )
    )

    # --------------------------------------------------------
    # Mask decoder
    # --------------------------------------------------------

    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,

        image_pe=model.prompt_encoder.get_dense_pe(),

        sparse_prompt_embeddings=sparse_embeddings,

        dense_prompt_embeddings=dense_embeddings,

        multimask_output=False
    )

    # --------------------------------------------------------
    # Resize prediction to 1024 × 1024
    # --------------------------------------------------------

    prediction = F.interpolate(
        low_res_logits,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )

    prediction = torch.sigmoid(prediction)

    prediction = prediction.squeeze().cpu().numpy()

    prediction_mask = (
        prediction > 0.5
    ).astype(np.uint8)

    return prediction_mask


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(
        pred,
        gt
    ).sum()

    pred_pixels = pred.sum()
    gt_pixels = gt.sum()

    union = np.logical_or(
        pred,
        gt
    ).sum()

    # Dice

    if pred_pixels + gt_pixels == 0:

        dice = 1.0

    else:

        dice = (
            2.0 * intersection
            /
            (pred_pixels + gt_pixels)
        )

    # IoU

    if union == 0:

        iou = 1.0

    else:

        iou = intersection / union

    # Precision

    if pred_pixels == 0:

        precision = 1.0 if gt_pixels == 0 else 0.0

    else:

        precision = intersection / pred_pixels

    # Recall

    if gt_pixels == 0:

        recall = 1.0 if pred_pixels == 0 else 0.0

    else:

        recall = intersection / gt_pixels

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "gt_pixels": int(gt_pixels),
        "pred_pixels": int(pred_pixels),
        "intersection": int(intersection)
    }


# ============================================================
# RUN VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("RUNNING FULL VALIDATION")
print("=" * 70)

results = []

start_time = time.time()

for index, filename in enumerate(
    tqdm(
        positive_files,
        desc="MedSAM Validation"
    ),
    start=1
):

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    mask_path = os.path.join(
        MASK_DIR,
        filename
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    if image.shape != (1024, 1024, 3):

        raise ValueError(
            f"Wrong image shape for {filename}: "
            f"{image.shape}"
        )

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    gt = np.array(
        Image.open(mask_path)
    )

    if gt.shape != (1024, 1024):

        raise ValueError(
            f"Wrong GT shape for {filename}: "
            f"{gt.shape}"
        )

    gt = (
        gt > 0
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Get bounding box
    # --------------------------------------------------------

    box = boxes[filename]["bbox"]

    if box is None:

        raise ValueError(
            f"Positive sample has no bbox: {filename}"
        )

    # --------------------------------------------------------
    # Image normalization
    #
    # Images are already 1024x1024 RGB.
    # We preserve the existing preprocessing.
    # --------------------------------------------------------

    image_float = image.astype(
        np.float32
    )

    image_float = (
        image_float - image_float.min()
    ) / max(
        image_float.max() - image_float.min(),
        1e-8
    )

    image_tensor = torch.from_numpy(
        image_float
    ).permute(
        2, 0, 1
    ).unsqueeze(
        0
    ).float().to(device)

    # --------------------------------------------------------
    # MedSAM
    # --------------------------------------------------------

    pred = medsam_inference(
        medsam_model,
        image_tensor,
        box,
        1024,
        1024
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        pred,
        gt
    )

    metrics["filename"] = filename
    metrics["bbox"] = box

    results.append(metrics)

    # --------------------------------------------------------
    # Save prediction
    # --------------------------------------------------------

    pred_image = (
        pred * 255
    ).astype(np.uint8)

    Image.fromarray(
        pred_image
    ).save(
        os.path.join(
            PRED_DIR,
            filename
        )
    )

    # Free GPU cache occasionally

    if index % 100 == 0:

        torch.cuda.empty_cache()


# ============================================================
# SUMMARY
# ============================================================

elapsed = time.time() - start_time

dice_values = [
    x["dice"]
    for x in results
]

iou_values = [
    x["iou"]
    for x in results
]

precision_values = [
    x["precision"]
    for x in results
]

recall_values = [
    x["recall"]
    for x in results
]


summary = {

    "dataset": "GBC",

    "split": "Val",

    "total_validation_slices":
        len(boxes),

    "positive_target_slices":
        len(positive_files),

    "empty_target_slices":
        len(empty_files),

    "evaluated_slices":
        len(results),

    "mean_dice":
        float(np.mean(dice_values)),

    "median_dice":
        float(np.median(dice_values)),

    "std_dice":
        float(np.std(dice_values)),

    "mean_iou":
        float(np.mean(iou_values)),

    "median_iou":
        float(np.median(iou_values)),

    "std_iou":
        float(np.std(iou_values)),

    "mean_precision":
        float(np.mean(precision_values)),

    "mean_recall":
        float(np.mean(recall_values)),

    "min_dice":
        float(np.min(dice_values)),

    "max_dice":
        float(np.max(dice_values)),

    "runtime_seconds":
        float(elapsed)
}


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "stage18_results.json"
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "stage18_summary.json"
)

with open(results_file, "w") as f:

    json.dump(
        results,
        f,
        indent=2
    )

with open(summary_file, "w") as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 18 RESULTS")
print("=" * 70)

print(
    f"\nValidation slices       : {len(boxes)}"
)

print(
    f"Positive target slices  : {len(positive_files)}"
)

print(
    f"Empty target slices     : {len(empty_files)}"
)

print(
    f"Evaluated slices        : {len(results)}"
)

print("\nPOSITIVE-TARGET BASELINE")
print("-" * 70)

print(
    f"Mean Dice               : "
    f"{summary['mean_dice']:.4f}"
)

print(
    f"Median Dice             : "
    f"{summary['median_dice']:.4f}"
)

print(
    f"Std Dice                : "
    f"{summary['std_dice']:.4f}"
)

print(
    f"Mean IoU                : "
    f"{summary['mean_iou']:.4f}"
)

print(
    f"Median IoU              : "
    f"{summary['median_iou']:.4f}"
)

print(
    f"Mean Precision          : "
    f"{summary['mean_precision']:.4f}"
)

print(
    f"Mean Recall             : "
    f"{summary['mean_recall']:.4f}"
)

print(
    f"Minimum Dice            : "
    f"{summary['min_dice']:.4f}"
)

print(
    f"Maximum Dice            : "
    f"{summary['max_dice']:.4f}"
)

print(
    f"\nRuntime                 : "
    f"{elapsed / 60:.2f} minutes"
)

print("\nOUTPUTS")
print("-" * 70)

print(
    "Predictions :",
    PRED_DIR
)

print(
    "Results     :",
    results_file
)

print(
    "Summary     :",
    summary_file
)

print("\n")
print("=" * 70)
print("STAGE 18 COMPLETE")
print("=" * 70)