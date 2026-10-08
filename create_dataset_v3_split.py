import os
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

RAW_DIR = "dataset_v2/raw"
OUTPUT_DIR = "dataset_v2"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VALIDATION_DIR = os.path.join(OUTPUT_DIR, "validation")
TEST_DIR = os.path.join(OUTPUT_DIR, "test")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ============================================================
# VALIDATE RATIOS
# ============================================================

if TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO != 1.0:
    raise ValueError("Train/validation/test ratios must add up to 1.0")

# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)

# ============================================================
# CHECK RAW DATASET
# ============================================================

if not os.path.exists(RAW_DIR):
    print(f"ERROR: Raw dataset not found: {RAW_DIR}")
    exit()

classes = sorted(
    [
        folder
        for folder in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, folder))
    ],
    key=lambda x: int(x)
)

if not classes:
    print("ERROR: No classes found.")
    exit()

print("=" * 60)
print("CREATING V3 DATASET SPLIT")
print("=" * 60)

print(f"Raw dataset : {RAW_DIR}")
print(f"Train       : {TRAIN_RATIO * 100:.0f}%")
print(f"Validation  : {VALIDATION_RATIO * 100:.0f}%")
print(f"Test        : {TEST_RATIO * 100:.0f}%")
print(f"Random seed : {RANDOM_SEED}")
print("=" * 60)

# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR
]:
    os.makedirs(directory, exist_ok=True)

# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_train = 0
total_validation = 0
total_test = 0

for class_name in classes:

    source_dir = os.path.join(
        RAW_DIR,
        class_name
    )

    # --------------------------------------------------------
    # Get image files
    # --------------------------------------------------------

    files = [
        file
        for file in os.listdir(source_dir)
        if file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ]

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(files)

    total = len(files)

    train_count = int(
        total * TRAIN_RATIO
    )

    validation_count = int(
        total * VALIDATION_RATIO
    )

    test_count = (
        total
        - train_count
        - validation_count
    )

    train_files = files[
        :train_count
    ]

    validation_files = files[
        train_count:
        train_count + validation_count
    ]

    test_files = files[
        train_count + validation_count:
    ]

    # --------------------------------------------------------
    # Create class directories
    # --------------------------------------------------------

    train_class_dir = os.path.join(
        TRAIN_DIR,
        class_name
    )

    validation_class_dir = os.path.join(
        VALIDATION_DIR,
        class_name
    )

    test_class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    os.makedirs(
        train_class_dir,
        exist_ok=True
    )

    os.makedirs(
        validation_class_dir,
        exist_ok=True
    )

    os.makedirs(
        test_class_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copy files
    # --------------------------------------------------------

    for file in train_files:

        shutil.copy2(
            os.path.join(source_dir, file),
            os.path.join(train_class_dir, file)
        )

    for file in validation_files:

        shutil.copy2(
            os.path.join(source_dir, file),
            os.path.join(validation_class_dir, file)
        )

    for file in test_files:

        shutil.copy2(
            os.path.join(source_dir, file),
            os.path.join(test_class_dir, file)
        )

    # --------------------------------------------------------
    # Update totals
    # --------------------------------------------------------

    total_train += len(train_files)
    total_validation += len(validation_files)
    total_test += len(test_files)

    print(
        f"Class {class_name:>2}: "
        f"Train = {len(train_files):>3} | "
        f"Validation = {len(validation_files):>3} | "
        f"Test = {len(test_files):>3}"
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)

print(f"Training images   : {total_train}")
print(f"Validation images : {total_validation}")
print(f"Testing images    : {total_test}")
print(
    f"Total images      : "
    f"{total_train + total_validation + total_test}"
)

print("=" * 60)