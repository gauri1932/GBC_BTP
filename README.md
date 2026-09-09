# GBC BTP - Medical Image Segmentation Pipeline

This repository contains a comprehensive pipeline for medical image analysis and segmentation using MedSAM (Medical Segment Anything Model).

## 📁 Project Structure

```
GBC_BTP/
├── analysis/              # Data analysis and exploration scripts
├── preprocessing/         # Data preparation and staging pipeline
├── validation/           # Validation and inspection utilities
├── inference/            # Model inference and testing
└── models/              # Model training and fine-tuning
```

### Folder Descriptions

#### `analysis/`
Scripts for analyzing dataset characteristics, labels, and patient data.

#### `preprocessing/`
Data preparation pipeline stages (stages 6-15):
- Patient inventory and splitting
- Integrity checks
- MedSAM preprocessing and analysis
- Box generation and validation

#### `validation/`
Data validation and inspection utilities:
- Image validation
- Dataset inspection
- Label visualization
- Mask checking

#### `inference/`
Model inference and testing scripts:
- MedSAM inference testing
- Full validation pipeline
- Result generation and analysis

#### `models/`
Model training and fine-tuning:
- MedSAM model fine-tuning

---

**Note**: This project is part of the Bachelor Thesis Project (BTP) on medical image segmentation.
