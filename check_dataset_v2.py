import os
from PIL import Image

DATASET_DIR = "dataset_v2/raw"

print("=" * 60)
print("V2 DATASET VERIFICATION")
print("=" * 60)

if not os.path.exists(DATASET_DIR):
    print(f"ERROR: Dataset directory not found:")
    print(DATASET_DIR)
    exit()

classes = sorted(
    [
        folder
        for folder in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, folder))
    ],
    key=lambda x: int(x)
)

if not classes:
    print("No classes found.")
    exit()

total_images = 0
total_valid = 0
total_corrupted = 0

for class_name in classes:

    class_path = os.path.join(
        DATASET_DIR,
        class_name
    )

    files = [
        f
        for f in os.listdir(class_path)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    valid = 0
    corrupted = 0

    for file in files:

        file_path = os.path.join(
            class_path,
            file
        )

        try:
            with Image.open(file_path) as img:

                # Verify image integrity
                img.verify()

            # Open again because verify() closes the file
            with Image.open(file_path) as img:

                width, height = img.size

                if width != 64 or height != 64:
                    print(
                        f"⚠️ Wrong size: "
                        f"{class_name}/{file} "
                        f"({width}x{height})"
                    )

                valid += 1

        except Exception as e:

            print(
                f"❌ Corrupted image: "
                f"{class_name}/{file}"
            )

            corrupted += 1

    total_images += len(files)
    total_valid += valid
    total_corrupted += corrupted

    print(
        f"Class {class_name:>2}: "
        f"Images = {len(files):>3} | "
        f"Valid = {valid:>3} | "
        f"Corrupted = {corrupted:>3}"
    )

print("=" * 60)
print(f"Total images     : {total_images}")
print(f"Valid images     : {total_valid}")
print(f"Corrupted images : {total_corrupted}")
print("=" * 60)