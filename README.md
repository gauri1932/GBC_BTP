# GBC BTP - Gallbladder-Cancer Image Analysis

Ultrasound-slice pipeline: shared preprocessing + EDA, with MedSAM as the
first segmentation model. New models plug into `models/` reusing the same code.

## Layout

```
gbc/                  shared library (model-agnostic)
  config.py           paths + constants (single source of truth)
  data.py             loaders, transforms, bbox, pairing, patient split
  eda.py              dataset analyses (inventory, labels, targets)
models/
  _template/          skeleton for a new model (prep/train/evaluate)
  medsam/             MedSAM plug-in: prep.py, train.py, evaluate.py
scripts/              runnable pipeline steps (run from repo root)
  make_split.py       patient-level Train/Val split (one-time)
  analyze_dataset.py  full EDA report (read-only)
  check_integrity.py  raw-data integrity check
  build_medsam.py     build MedSAM inputs + targets
  validate_medsam.py  validate MedSAM inputs + targets
  prepare_cls_data.py light classification inputs (256px + labels.csv)
tools/                one-off diagnostics / visualizations
```

Large data (`GBC_dataset/`, `cls_data/`), checkpoints (`*.pth`) and model
outputs are git-ignored; only code is versioned.

## Typical flow

```bash
python scripts/analyze_dataset.py --quick 200   # EDA smoke test
python scripts/analyze_dataset.py              # full EDA
python scripts/build_medsam.py                 # MedSAM inputs (1024 RGB + masks + boxes)
python scripts/validate_medsam.py              # must PASS before training
python models/medsam/train.py                  # fine-tune (GPU)
python models/medsam/evaluate.py --mode smoke  # 5 samples + figures
python models/medsam/evaluate.py --mode full   # whole Val + metrics
```

See `models/README.md` for adding another model on the same EDA/preprocessed data.
