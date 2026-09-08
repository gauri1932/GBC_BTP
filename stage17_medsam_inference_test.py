import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry


# ============================================================
# STAGE 17 - MEDSAM INFERENCE TEST
# ============================================================

BASE_DIR = r"C:\Users\Student\Downloads\GBC"

CHECKPOINT = os.path.join(
    BASE_DIR,
    "checkpoints",
    "medsam_vit_b.pth"
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "GBC_dataset"
)

IMAGE_DIR = os.path.join(
    DATASET_DIR,
    "medsam_preprocessed",
    "Train",
    "INP"
)

MASK_DIR = os.path.join(
    DATASET_DIR,
    "medsam_targets",
    "Train",
    "GT_binary"
)

BOX_FILE = os.path.join(
    DATASET_DIR,
    "medsam_targets",
    "Train",
    "boxes.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "stage17_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================
# SETTINGS
# ============================================================

NUM_POSITIVE_SAMPLES = 5
IMAGE_SIZE = 1024


# ============================================================
# CHECK ENVIRONMENT
# ============================================================

print("=" * 70)
print("STAGE 17 - MEDSAM INFERENCE TEST")
print("=" * 70)

print()
print("DEVICE")
print("-" * 70)
print("PyTorch CUDA available :", torch.cuda.is_available())
print("Device                  :", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU                     :",
        torch.cuda.get_device_name(0)
    )

print()
print("CHECKPOINT")
print("-" * 70)
print(CHECKPOINT)

if not os.path.isfile(CHECKPOINT):
    raise FileNotFoundError(
        f"MedSAM checkpoint not found:\n{CHECKPOINT}"
    )

print("✓ Checkpoint found")


# ============================================================
# CHECK DATASET PATHS
# ============================================================

print()
print("DATASET PATHS")
print("-" * 70)

for path in [
    IMAGE_DIR,
    MASK_DIR,
    BOX_FILE
]:
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )

print("✓ Dataset paths found")


# ============================================================
# LOAD BOXES
# ============================================================

print()
print("LOADING BOUNDING BOXES")
print("-" * 70)

with open(BOX_FILE, "r") as f:
    boxes_data = json.load(f)

print("Total JSON entries:", len(boxes_data))


# ============================================================
# SELECT POSITIVE SAMPLES
# ============================================================

positive_samples = []

for filename, info in boxes_data.items():

    bbox = info.get("bbox")

    if bbox is None:
        continue

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    mask_path = os.path.join(
        MASK_DIR,
        filename
    )

    if not os.path.isfile(image_path):
        continue

    if not os.path.isfile(mask_path):
        continue

    positive_samples.append(
        (filename, bbox)
    )

positive_samples = sorted(
    positive_samples,
    key=lambda x: x[0]
)

if len(positive_samples) < NUM_POSITIVE_SAMPLES:
    raise RuntimeError(
        "Not enough positive samples found."
    )

selected_samples = positive_samples[
    :NUM_POSITIVE_SAMPLES
]

print(
    "Positive samples available:",
    len(positive_samples)
)

print()
print("Selected samples:")

for filename, bbox in selected_samples:
    print(
        f"  {filename} -> bbox={bbox}"
    )


# ============================================================
# LOAD MEDSAM
# ============================================================

print()
print("LOADING MEDSAM")
print("-" * 70)

print("Loading ViT-B checkpoint...")

medsam_model = sam_model_registry[
    "vit_b"
](
    checkpoint=CHECKPOINT
)

medsam_model = medsam_model.to(
    DEVICE
)

medsam_model.eval()

print("✓ MedSAM loaded")


# ============================================================
# MEDSAM INFERENCE FUNCTION
# ============================================================

@torch.no_grad()
def run_medsam(
    model,
    image_np,
    bbox
):

    # --------------------------------------------------------
    # Image should already be 1024 x 1024 RGB
    # --------------------------------------------------------

    if image_np.shape != (
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    ):
        raise ValueError(
            f"Unexpected image shape: "
            f"{image_np.shape}"
        )

    # --------------------------------------------------------
    # Normalize image to [0, 1]
    #
    # This follows the official MedSAM inference approach.
    # --------------------------------------------------------

    image_float = image_np.astype(
        np.float32
    )

    image_min = image_float.min()
    image_max = image_float.max()

    image_float = (
        image_float - image_min
    ) / max(
        image_max - image_min,
        1e-8
    )

    # --------------------------------------------------------
    # HWC -> CHW -> batch
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image_float
    ).permute(
        2, 0, 1
    ).unsqueeze(0).float().to(
        DEVICE
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # bbox is ALREADY in 1024x1024 coordinates.
    #
    # Do NOT scale it again.
    # --------------------------------------------------------

    box_np = np.array(
        [bbox],
        dtype=np.float32
    )

    box_tensor = torch.as_tensor(
        box_np,
        dtype=torch.float32,
        device=DEVICE
    )

    if len(box_tensor.shape) == 2:
        box_tensor = box_tensor[:, None, :]

    # --------------------------------------------------------
    # IMAGE ENCODER
    # --------------------------------------------------------

    image_embedding = model.image_encoder(
        image_tensor
    )

    # --------------------------------------------------------
    # PROMPT ENCODER
    # --------------------------------------------------------

    sparse_embeddings, dense_embeddings = (
        model.prompt_encoder(
            points=None,
            boxes=box_tensor,
            masks=None
        )
    )

    # --------------------------------------------------------
    # MASK DECODER
    # --------------------------------------------------------

    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False
    )

    # --------------------------------------------------------
    # Convert logits -> probability
    # --------------------------------------------------------

    low_res_pred = torch.sigmoid(
        low_res_logits
    )

    # --------------------------------------------------------
    # Resize prediction to 1024x1024
    # --------------------------------------------------------

    prediction = F.interpolate(
        low_res_pred,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    prediction = prediction.squeeze().cpu().numpy()

    # --------------------------------------------------------
    # Binary prediction
    # --------------------------------------------------------

    prediction_binary = (
        prediction > 0.5
    ).astype(np.uint8)

    return prediction_binary


# ============================================================
# METRICS
# ============================================================

def calculate_dice(
    prediction,
    ground_truth
):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    prediction_area = prediction.sum()
    ground_truth_area = ground_truth.sum()

    denominator = (
        prediction_area +
        ground_truth_area
    )

    if denominator == 0:
        return 1.0

    return (
        2.0 * intersection
    ) / denominator


def calculate_iou(
    prediction,
    ground_truth
):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(
    image,
    ground_truth,
    prediction,
    bbox,
    filename,
    dice,
    iou
):

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    axes[0].imshow(image)
    axes[0].set_title(
        "Input Image"
    )
    axes[0].axis("off")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    axes[1].imshow(image)
    axes[1].imshow(
        ground_truth,
        alpha=0.5
    )

    x1, y1, x2, y2 = bbox

    rectangle = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2
    )

    axes[1].add_patch(
        rectangle
    )

    axes[1].set_title(
        "Ground Truth + BBox"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # MedSAM prediction
    # --------------------------------------------------------

    axes[2].imshow(image)
    axes[2].imshow(
        prediction,
        alpha=0.5
    )

    rectangle = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        linewidth=2
    )

    axes[2].add_patch(
        rectangle
    )

    axes[2].set_title(
        "MedSAM Prediction"
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    )

    gt_bool = ground_truth > 0
    pred_bool = prediction > 0

    # True positive
    tp = gt_bool & pred_bool

    # Ground truth only
    fn = gt_bool & (~pred_bool)

    # Prediction only
    fp = (~gt_bool) & pred_bool

    comparison[tp] = [255, 255, 255]
    comparison[fn] = [255, 0, 0]
    comparison[fp] = [0, 255, 0]

    axes[3].imshow(
        comparison
    )

    axes[3].set_title(
        f"Comparison\nDice={dice:.4f} | IoU={iou:.4f}"
    )

    axes[3].axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "result_" + filename
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# RUN TEST
# ============================================================

print()
print("=" * 70)
print("RUNNING MEDSAM")
print("=" * 70)

results = []


for index, (
    filename,
    bbox
) in enumerate(
    selected_samples,
    start=1
):

    print()
    print(
        f"[{index}/{len(selected_samples)}] "
        f"{filename}"
    )

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
        Image.open(image_path)
    )

    print(
        "Image shape:",
        image.shape
    )

    print(
        "Image dtype:",
        image.dtype
    )

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    gt = np.array(
        Image.open(mask_path)
    )

    # Convert 0/255 -> 0/1
    gt_binary = (
        gt > 0
    ).astype(np.uint8)

    print(
        "GT foreground pixels:",
        int(gt_binary.sum())
    )

    print(
        "Bounding box:",
        bbox
    )

    # --------------------------------------------------------
    # Run MedSAM
    # --------------------------------------------------------

    prediction = run_medsam(
        medsam_model,
        image,
        bbox
    )

    print(
        "Prediction foreground pixels:",
        int(prediction.sum())
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    dice = calculate_dice(
        prediction,
        gt_binary
    )

    iou = calculate_iou(
        prediction,
        gt_binary
    )

    print(
        f"Dice : {dice:.4f}"
    )

    print(
        f"IoU  : {iou:.4f}"
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    save_visualization(
        image=image,
        ground_truth=gt_binary,
        prediction=prediction,
        bbox=bbox,
        filename=filename,
        dice=dice,
        iou=iou
    )

    results.append({
        "filename": filename,
        "bbox": bbox,
        "gt_pixels": int(gt_binary.sum()),
        "prediction_pixels": int(prediction.sum()),
        "dice": float(dice),
        "iou": float(iou)
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "stage17_results.json"
)

with open(
    results_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("STAGE 17 RESULTS")
print("=" * 70)

print()

for result in results:

    print(
        f"{result['filename']:<20} "
        f"Dice={result['dice']:.4f} "
        f"IoU={result['iou']:.4f}"
    )

mean_dice = np.mean([
    r["dice"]
    for r in results
])

mean_iou = np.mean([
    r["iou"]
    for r in results
])

print()
print("-" * 70)

print(
    f"Mean Dice : {mean_dice:.4f}"
)

print(
    f"Mean IoU  : {mean_iou:.4f}"
)

print()
print(
    "Visualizations saved to:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Results saved to:"
)

print(
    results_file
)

print()
print("=" * 70)
print("STAGE 17 COMPLETE")
print("=" * 70)