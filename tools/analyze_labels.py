from pathlib import Path
from collections import Counter


DATASET = Path("GBC_dataset")


def analyze_split(split):

    label_folder = DATASET / split / "LABEL"

    counter = Counter()

    total_files = 0

    for file in label_folder.glob("*.txt"):

        with open(file, "r") as f:
            content = f.read().strip()

        if content:
            counter[content] += 1
            total_files += 1

    print("\n" + "=" * 70)
    print(f"{split.upper()} LABEL ANALYSIS")
    print("=" * 70)

    print("Total LABEL files:", total_files)
    print("Unique LABEL vectors:", len(counter))

    print("\nVector                         Count")
    print("-" * 50)

    for vector, count in counter.most_common():

        print(
            f"{vector:<35} {count}"
        )


analyze_split("Train")
analyze_split("Test")


print("\n" + "=" * 70)
print("LABEL ANALYSIS COMPLETE")
print("=" * 70)