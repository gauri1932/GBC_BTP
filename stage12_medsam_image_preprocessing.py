import os
from PIL import Image
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Student\Downloads\GBC\GBC_dataset"

SOURCE_DIRS = {
    "Train": os.path.join(BASE_DIR, "split", "Train", "INP"),
    "Val": os.path.join(BASE_DIR, "split", "Val", "INP"),
    "Test": os.path.join(BASE_DIR, "Test", "INP"),
}

OUTPUT_BASE = os.path.join(
    BASE_DIR,
    "medsam_preprocessed"
)


# ============================================================
# MEDSAM IMAGE SIZE
# ============================================================

MEDSAM_SIZE = (1024, 1024)


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split_name, source_dir):

    output_dir = os.path.join(
        OUTPUT_BASE,
        split_name,
        "INP"
    )

    os.makedirs(output_dir, exist_ok=True)

    files = sorted([
        f for f in os.listdir(source_dir)
        if f.lower().endswith(".png")
    ])

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name}")
    print("=" * 70)

    print(f"Source images: {len(files)}")
    print(f"Output size: {MEDSAM_SIZE}")
    print("Conversion: Grayscale -> RGB")

    processed = 0
    errors = []

    for filename in tqdm(files):

        source_path = os.path.join(
            source_dir,
            filename
        )

        output_path = os.path.join(
            output_dir,
            filename
        )

        try:

            # ------------------------------------------------
            # Load original image
            # ------------------------------------------------

            image = Image.open(source_path)

            # ------------------------------------------------
            # Convert grayscale -> RGB
            # ------------------------------------------------

            image = image.convert("RGB")

            # ------------------------------------------------
            # Resize to MedSAM input resolution
            # ------------------------------------------------

            image = image.resize(
                MEDSAM_SIZE,
                Image.Resampling.BILINEAR
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            image.save(
                output_path,
                format="PNG"
            )

            processed += 1

        except Exception as e:

            errors.append(
                (filename, str(e))
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "-" * 70)
    print(f"{split_name} RESULTS")
    print("-" * 70)

    print(f"Source images : {len(files)}")
    print(f"Processed     : {processed}")
    print(f"Errors        : {len(errors)}")

    if errors:

        print("\nErrors:")

        for filename, error in errors[:20]:

            print(
                f"{filename}: {error}"
            )

    return processed, errors


# ============================================================
# VALIDATE OUTPUT
# ============================================================

def validate_output(split_name):

    output_dir = os.path.join(
        OUTPUT_BASE,
        split_name,
        "INP"
    )

    files = sorted([
        f for f in os.listdir(output_dir)
        if f.lower().endswith(".png")
    ])

    print("\n" + "-" * 70)
    print(f"VALIDATING {split_name}")
    print("-" * 70)

    print(
        f"Output files: {len(files)}"
    )

    dimension_counts = {}
    mode_counts = {}

    bad_files = []

    for filename in tqdm(files):

        path = os.path.join(
            output_dir,
            filename
        )

        try:

            image = Image.open(path)

            dimension_counts[
                image.size
            ] = dimension_counts.get(
                image.size,
                0
            ) + 1

            mode_counts[
                image.mode
            ] = mode_counts.get(
                image.mode,
                0
            ) + 1

            if image.size != MEDSAM_SIZE:

                bad_files.append(
                    (
                        filename,
                        f"Wrong size: {image.size}"
                    )
                )

            if image.mode != "RGB":

                bad_files.append(
                    (
                        filename,
                        f"Wrong mode: {image.mode}"
                    )
                )

            image.close()

        except Exception as e:

            bad_files.append(
                (filename, str(e))
            )

    print("\nDimensions:")

    for size, count in dimension_counts.items():

        print(
            f"  {size}: {count}"
        )

    print("\nModes:")

    for mode, count in mode_counts.items():

        print(
            f"  {mode}: {count}"
        )

    print(
        f"\nValidation errors: "
        f"{len(bad_files)}"
    )

    if bad_files:

        print("\nFirst errors:")

        for filename, error in bad_files[:20]:

            print(
                f"{filename}: {error}"
            )

    else:

        print(
            "✓ All processed images are "
            "1024×1024 RGB"
        )

    return bad_files


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STAGE 12 - MEDSAM IMAGE PREPROCESSING")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Original dataset will NOT be modified."
    )

    print(
        "Only processed INP images will be created."
    )

    print(
        f"MedSAM image size: {MEDSAM_SIZE}"
    )

    print(
        "Interpolation: Bilinear"
    )

    print(
        "Image conversion: Grayscale -> RGB"
    )

    # --------------------------------------------------------
    # Create output base directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_BASE,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process all splits
    # --------------------------------------------------------

    results = {}

    for split_name, source_dir in SOURCE_DIRS.items():

        if not os.path.exists(source_dir):

            print(
                f"\nWARNING: Source directory missing:"
            )

            print(source_dir)

            continue

        results[split_name] = process_split(
            split_name,
            source_dir
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OUTPUT VALIDATION")
    print("=" * 70)

    validation_results = {}

    for split_name in results:

        validation_results[
            split_name
        ] = validate_output(
            split_name
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("STAGE 12 COMPLETE")
    print("=" * 70)

    print("\nProcessed images:")

    for split_name, result in results.items():

        processed, errors = result

        print(
            f"{split_name}: "
            f"{processed} processed, "
            f"{len(errors)} errors"
        )

    print("\nValidation:")

    for split_name, errors in validation_results.items():

        if len(errors) == 0:

            print(
                f"{split_name}: ✓ PASS"
            )

        else:

            print(
                f"{split_name}: ✗ "
                f"{len(errors)} errors"
            )

    print(
        "\nOriginal dataset remains untouched."
    )

    print(
        "Processed images are stored in:"
    )

    print(
        OUTPUT_BASE
    )

    print("=" * 70)