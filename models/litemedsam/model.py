# ============================================================
# LITEMEDSAM MODEL (official architecture, GBC wiring)
# ============================================================
#
# TinyViT-5M encoder (vendored in tiny_vit_sam.py) +
# SAM prompt encoder + mask decoder from the local
# segment_anything package (same classes MedSAM uses).
#
# Hyperparameters are exactly the official LiteMedSAM ones,
# so lite_medsam.pth loads with strict=True.
#
# Official source:
#   https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM
#
# ============================================================

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import (
    MaskDecoder,
    PromptEncoder,
    TwoWayTransformer,
)

from models.litemedsam.tiny_vit_sam import TinyViT


# ============================================================
# CONFIG (official LiteMedSAM values -- do not change,
# otherwise lite_medsam.pth will not load)
# ============================================================

INPUT_SIZE = 256          # network resolution (embeddings 64x64)
PROMPT_EMBED_DIM = 256

TINY_VIT_KWARGS = dict(
    img_size=256,
    in_chans=3,
    embed_dims=[64, 128, 160, 320],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.0,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8,
)

# Official pretrained checkpoint (Google Drive, from the
# LiteMedSAM README). Auto-downloaded on first use.
GDRIVE_FILE_ID = "18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z"

# 1024 preprocessed coords -> 256 network coords.
COORD_SCALE = INPUT_SIZE / 1024.0


# ============================================================
# MODEL
# ============================================================

class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, boxes):
        # image: (B, 3, 256, 256) in [0, 1]
        # boxes: (B, 1, 4) in 256 coordinates
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )  # masks: (B, 1, 256, 256)

        return low_res_masks, iou_predictions


def build_litemedsam(checkpoint=None, device=None):
    """Build MedSAM_Lite; optionally strict-load a checkpoint."""
    model = MedSAM_Lite(
        image_encoder=TinyViT(**TINY_VIT_KWARGS),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=PROMPT_EMBED_DIM,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=PROMPT_EMBED_DIM,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=PROMPT_EMBED_DIM,
            image_embedding_size=(64, 64),
            input_image_size=(INPUT_SIZE, INPUT_SIZE),
            mask_in_chans=16,
        ),
    )

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        # Official checkpoints store raw state dicts;
        # our own train.py stores {"model": ...}.
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        print(f"Loaded LiteMedSAM weights: {checkpoint}")

    if device is not None:
        model = model.to(device)

    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total, trainable


# ============================================================
# CHECKPOINT DOWNLOAD
# ============================================================

def download_gdrive(file_id, dst_path):
    """Download a Google Drive file via gdown (handles confirm tokens)."""
    import gdown

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(dst_path), quiet=False)

    if not dst_path.is_file() or dst_path.stat().st_size == 0:
        raise RuntimeError("gdown produced an empty file")

    size_mb = dst_path.stat().st_size / 1e6
    print(f"Downloaded {size_mb:.1f} MB -> {dst_path}")
    return dst_path


def ensure_pretrained(dst_path):
    """Return dst_path, downloading official lite_medsam.pth if needed."""
    dst_path = Path(dst_path)

    if dst_path.is_file() and dst_path.stat().st_size > 0:
        return dst_path

    print("Official LiteMedSAM checkpoint not found locally.")
    print(f"Downloading to {dst_path} ...")

    try:
        return download_gdrive(GDRIVE_FILE_ID, dst_path)
    except Exception as e:
        if dst_path.exists():
            os.remove(dst_path)
        raise RuntimeError(
            "Auto-download failed. Manually download lite_medsam.pth "
            "from https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM "
            f"(README link) and place it at {dst_path}\nReason: {e}"
        )
