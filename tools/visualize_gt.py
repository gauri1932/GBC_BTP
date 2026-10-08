from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


dataset = Path("GBC_dataset")


def find_largest_masks(split_name, top_n=5):

    inp_folder = dataset / split_name / "INP"
    gt_folder = dataset / split_name / "GT"

    samples = []

    for inp_file in inp_folder.glob("*.png"):

        gt_file = gt_folder / inp_file.name

        if not gt_file.exists():
            continue

        gt = np.array(Image.open(gt_file))

        foreground_pixels = np.count_nonzero(gt)

        samples.append(
            (
                foreground_pixels,
                inp_file,
                gt_file
            )
        )

    # Largest masks first
    samples.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return samples[:top_n]


def visualize_split(split_name):

    print("\n" + "=" * 70)
    print(f"LARGEST GT MASKS: {split_name.upper()}")
    print("=" * 70)

    samples = find_largest_masks(
        split_name,
        top_n=5
    )

    for rank, (
        foreground_pixels,
        inp_file,
        gt_file
    ) in enumerate(samples, start=1):

        inp = np.array(
            Image.open(inp_file)
        )

        gt = np.array(
            Image.open(gt_file)
        )

        binary_gt = gt > 0

        print("\n" + "-" * 60)
        print(f"RANK {rank}")
        print("File:", inp_file.name)
        print("Foreground pixels:", foreground_pixels)
        print("GT labels:", np.unique(gt).tolist())

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(20, 5)
        )

        # ----------------------------------------------------
        # INP
        # ----------------------------------------------------

        axes[0].imshow(
            inp,
            cmap="gray"
        )

        axes[0].set_title(
            f"INP\n{inp_file.name}"
        )

        axes[0].axis("off")

        # ----------------------------------------------------
        # ORIGINAL GT
        # ----------------------------------------------------

        axes[1].imshow(
            gt
        )

        axes[1].set_title(
            "Original GT\nValues 0-8"
        )

        axes[1].axis("off")

        # ----------------------------------------------------
        # BINARY GT
        # ----------------------------------------------------

        axes[2].imshow(
            binary_gt,
            cmap="gray"
        )

        axes[2].set_title(
            f"Binary GT\n{foreground_pixels} pixels"
        )

        axes[2].axis("off")

        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        axes[3].imshow(
            inp,
            cmap="gray"
        )

        axes[3].imshow(
            binary_gt,
            alpha=0.5
        )

        axes[3].set_title(
            "INP + GT Overlay"
        )

        axes[3].axis("off")

        plt.tight_layout()

        plt.show()


# ============================================================
# RUN TRAIN
# ============================================================

visualize_split("Train")


# ============================================================
# RUN TEST
# ============================================================

visualize_split("Test")


print("\n" + "=" * 70)
print("STAGE 5 COMPLETE")
print("=" * 70)