import os
import shutil
import random

# ============================================================
# SETTINGS
# ============================================================

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

VALIDATION_PER_CLASS = 80

RANDOM_SEED = 42

# ============================================================
# SET RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)

# ============================================================
# CREATE VALIDATION DIRECTORY
# ============================================================

os.makedirs(
    VALIDATION_DIR,
    exist_ok=True
)

# ============================================================
# FIND CLASSES
# ============================================================

classes = sorted(
    [
        folder
        for folder in os.listdir(TRAIN_DIR)
        if os.path.isdir(
            os.path.join(
                TRAIN_DIR,
                folder
            )
        )
    ],
    key=lambda x: int(x)
)

print("=" * 60)
print("CREATING VALIDATION DATASET")
print("=" * 60)

# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in classes:

    train_class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    validation_class_dir = os.path.join(
        VALIDATION_DIR,
        class_name
    )

    os.makedirs(
        validation_class_dir,
        exist_ok=True
    )

    images = [
        file
        for file in os.listdir(
            train_class_dir
        )
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    # Shuffle images
    random.shuffle(images)

    # Select validation images
    validation_images = images[
        :VALIDATION_PER_CLASS
    ]

    # Move them
    for image in validation_images:

        source = os.path.join(
            train_class_dir,
            image
        )

        destination = os.path.join(
            validation_class_dir,
            image
        )

        shutil.move(
            source,
            destination
        )

    print(
        f"Class {class_name:>2}: "
        f"Train = {len(images) - VALIDATION_PER_CLASS:>3} | "
        f"Validation = {VALIDATION_PER_CLASS:>3}"
    )

# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)
print("VALIDATION SPLIT COMPLETE")
print("=" * 60)