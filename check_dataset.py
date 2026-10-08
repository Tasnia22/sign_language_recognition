import os
from PIL import Image

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

classes = sorted(
    [folder for folder in os.listdir(TRAIN_DIR)
     if os.path.isdir(os.path.join(TRAIN_DIR, folder))],
    key=lambda x: int(x)
)

print("=" * 50)
print("DATASET VERIFICATION")
print("=" * 50)

total_train = 0
total_test = 0

for class_name in classes:

    train_path = os.path.join(TRAIN_DIR, class_name)
    test_path = os.path.join(TEST_DIR, class_name)

    train_files = [
        f for f in os.listdir(train_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    test_files = [
        f for f in os.listdir(test_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    valid_train = 0
    valid_test = 0

    for file in train_files:
        try:
            with Image.open(os.path.join(train_path, file)) as img:
                img.verify()
            valid_train += 1
        except Exception:
            print(f"❌ Corrupted train image: {class_name}/{file}")

    for file in test_files:
        try:
            with Image.open(os.path.join(test_path, file)) as img:
                img.verify()
            valid_test += 1
        except Exception:
            print(f"❌ Corrupted test image: {class_name}/{file}")

    total_train += valid_train
    total_test += valid_test

    print(
        f"Class {class_name:>2}: "
        f"Train = {valid_train:>3} | "
        f"Test = {valid_test:>3}"
    )

print("=" * 50)
print(f"Total training images : {total_train}")
print(f"Total testing images  : {total_test}")
print(f"Total images          : {total_train + total_test}")
print("=" * 50)