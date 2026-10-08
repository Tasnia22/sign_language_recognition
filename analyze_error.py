import os
import shutil
import tensorflow as tf
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 64
BATCH_SIZE = 32

TEST_DIR = "dataset/test"
MODEL_PATH = "models/sign_language_model.keras"

OUTPUT_DIR = "models/evaluation/misclassified"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded.")


# ============================================================
# LOAD TEST DATASET
# ============================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names


# ============================================================
# GET IMAGE FILE PATHS
# ============================================================

file_paths = test_dataset.file_paths


# ============================================================
# NORMALIZE DATA
# ============================================================

test_dataset = test_dataset.map(
    lambda images, labels: (
        tf.cast(images, tf.float32) / 255.0,
        labels
    )
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND MISCLASSIFIED IMAGES
# ============================================================

image_index = 0
error_count = 0

print("\nAnalyzing test images...\n")

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    true_labels = labels.numpy()

    for i in range(len(labels)):

        actual = true_labels[i]
        predicted = predicted_labels[i]

        if actual != predicted:

            actual_name = class_names[actual]
            predicted_name = class_names[predicted]

            folder_name = (
                f"actual_{actual_name}"
                f"_predicted_{predicted_name}"
            )

            output_folder = os.path.join(
                OUTPUT_DIR,
                folder_name
            )

            os.makedirs(
                output_folder,
                exist_ok=True
            )

            source_path = file_paths[image_index]

            filename = os.path.basename(
                source_path
            )

            destination_path = os.path.join(
                output_folder,
                filename
            )

            shutil.copy2(
                source_path,
                destination_path
            )

            error_count += 1

        image_index += 1


# ============================================================
# FINISHED
# ============================================================

print("=" * 60)
print("ERROR ANALYSIS COMPLETE")
print("=" * 60)

print(f"Misclassified images: {error_count}")

print("\nSaved to:")
print(OUTPUT_DIR)