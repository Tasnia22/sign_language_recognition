import os

DATASETS = {
    "Training": "dataset/train",
    "Validation": "dataset/validation",
    "Testing": "dataset/test"
}

print("=" * 60)
print("DATASET SPLIT VERIFICATION")
print("=" * 60)

grand_total = 0

for dataset_name, dataset_dir in DATASETS.items():

    print(f"\n{dataset_name}")

    print("-" * 40)

    dataset_total = 0

    classes = sorted(
        [
            folder
            for folder in os.listdir(dataset_dir)
            if os.path.isdir(
                os.path.join(
                    dataset_dir,
                    folder
                )
            )
        ],
        key=lambda x: int(x)
    )

    for class_name in classes:

        class_dir = os.path.join(
            dataset_dir,
            class_name
        )

        images = [
            file
            for file in os.listdir(class_dir)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        count = len(images)

        dataset_total += count

        print(
            f"Class {class_name:>2}: {count:>3}"
        )

    print(
        f"Total {dataset_name.lower()}: "
        f"{dataset_total}"
    )

    grand_total += dataset_total

print("\n" + "=" * 60)

print(
    f"Grand Total: {grand_total}"
)

print("=" * 60)