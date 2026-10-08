from pathlib import Path


# Your dataset location
dataset = Path("GBC_dataset")


# Train and Test
for split in ["Train", "Test"]:

    print("\n==============================")
    print(split)
    print("==============================")

    split_path = dataset / split

    # Three folders
    label = split_path / "LABEL"
    inp = split_path / "INP"
    gt = split_path / "GT"

    # Get filenames without extensions
    label_files = {file.stem for file in label.glob("*")}
    inp_files = {file.stem for file in inp.glob("*")}
    gt_files = {file.stem for file in gt.glob("*")}

    print("LABEL files:", len(label_files))
    print("INP files:  ", len(inp_files))
    print("GT files:   ", len(gt_files))

    # Check common files
    common = label_files & inp_files & gt_files

    print("Complete samples:", len(common))

    # Check missing files
    missing_inp = label_files - inp_files
    missing_gt = label_files - gt_files

    print("LABEL without INP:", len(missing_inp))
    print("LABEL without GT: ", len(missing_gt))
    