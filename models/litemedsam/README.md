# LiteMedSAM (GBC)

Official LiteMedSAM architecture (TinyViT-5M encoder + SAM prompt/mask
decoder, ~9.8M params vs MedSAM ViT-B ~91M), trained on the **same**
`medsam_preprocessed/` + `medsam_targets/` files MedSAM uses. No new
preprocessing: the 1024 → 256 downscale (box / 4) happens in the loader.

Architecture: https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM
(`tiny_vit_sam.py` vendored verbatim so `lite_medsam.pth` loads strictly).

## Files

| File | Job |
|---|---|
| `tiny_vit_sam.py` | Vendored TinyViT encoder (do not edit) |
| `model.py` | `MedSAM_Lite` assembly + `build_litemedsam()` + checkpoint download |
| `prep.py` | Readiness check on the shared MedSAM data (builds nothing new) |
| `train.py` | Full fine-tune, BCE+Dice, AdamW 5e-5, bbox jitter |
| `evaluate.py` | `smoke` (figures) / `full` (metrics JSON, same schema as MedSAM) |

## Setup (once)

```bash
.\medsam_env\Scripts\python.exe -m pip install timm   # TinyViT dependency
python -c "from models.litemedsam.model import ensure_pretrained; \
  ensure_pretrained('checkpoints/lite_medsam.pth')"
```

`ensure_pretrained` downloads the official `lite_medsam.pth` (~40 MB,
Google Drive link from the LiteMedSAM README) into `checkpoints/`
(git-ignored). If auto-download fails, download manually and place it there.

## Train

```bash
python models/litemedsam/train.py --smoke-test          # 5 batches, saves nothing
python models/litemedsam/train.py                       # 10 epochs, batch 8
python models/litemedsam/train.py --epochs 20 --batch-size 16 --lr 5e-5
python models/litemedsam/train.py --from-scratch        # no pretrained init
python models/litemedsam/train.py --resume models/litemedsam/outputs/litemedsam_epoch_3.pth
```

Checkpoints land in `models/litemedsam/outputs/` (`best_litemedsam.pth` +
per-epoch + `training_history.json`).

## Evaluate

```bash
python models/litemedsam/evaluate.py --mode smoke --n 5
python models/litemedsam/evaluate.py --mode full --split Val
python models/litemedsam/evaluate.py --mode full \
  --checkpoint models/litemedsam/outputs/best_litemedsam.pth
```

## Data contract (shared with MedSAM)

- Image: `medsam_preprocessed/<Split>/INP/*.png` (1024 RGB)
- Mask: `medsam_targets/<Split>/GT_binary/*.png` (1024, {0,255})
- Box: `medsam_targets/<Split>/boxes.json` (inclusive 1024 coords; `/4` → 256)
- Only box-positive slices are used (empty targets have no prompt).
