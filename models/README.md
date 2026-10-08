# Models

Each model lives in its own folder and reuses the shared `gbc` library
(loaders, transforms, EDA, splits). Nothing model-specific belongs in `gbc/`.

## Adding a new model

```bash
cp -r models/_template models/<your_model>
```

Then implement the three files (see `models/medsam/` for a full example):

| File | Job |
|---|---|
| `prep.py` | Raw split → model-ready inputs on disk (`build_inputs`), plus `validate_split` / `passed` |
| `train.py` | Training loop; save best checkpoint to `models/<your_model>/outputs/` |
| `evaluate.py` | `smoke` mode (few samples + figures) and `full` mode (whole split + metrics JSON) |

Conventions:

- Import shared code only: `from gbc import config, data, eda`.
- Paths come from `gbc.config` (`DATASET_DIR`, `SPLIT_DIRS`, ...), never hardcoded.
- Keep outputs inside your model folder (`outputs/`, git-ignored via `*.pth` etc.).
- Report metrics at slice level **and** patient-aggregated (slices from one
  patient are correlated).
- `medsam/prep.py` owns the `medsam_preprocessed/` + `medsam_targets/` layout;
  a new model must use its own directory names.
