# ============================================================
# NEW-MODEL TRAIN TEMPLATE
# ============================================================
# Copy to models/<your_model>/train.py and implement.
# Reuse gbc.data for loading; see models/medsam/train.py
# for a full example (frozen encoder + trainable head).
# ============================================================

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4

# TODO: dataset class, model, loss, train/val loop,
#       save best checkpoint to models/<your_model>/outputs/


if __name__ == "__main__":
    raise NotImplementedError("Implement the training loop first.")
