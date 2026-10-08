import os
import shutil

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = "dataset_v2/test"
MODEL_PATH = "models/sign_language_model_v3.keras"

OUTPUT_DIR = "models/evaluation_v3"
MISCLASSIFIED_DIR = os.path.join(
    OUTPUT_DIR,
    "misclassified"
)

IMAGE_SIZE = 64
BATCH_SIZE = 32


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    MISCLASSIFIED_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING V3 MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")
print(f"Model input shape  : {model.input_shape}")
print(f"Model output shape : {model.output_shape}")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print()
print("=" * 60)
print("LOADING TEST DATASET")
print("=" * 60)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print()
print("Class order:")
print(class_names)


# ============================================================
# PREFETCH
# ============================================================

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# MODEL EVALUATION
# ============================================================

print()
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print()
print(f"Test Loss     : {test_loss:.6f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# PREDICTIONS
# ============================================================

print()
print("=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_true = []
y_pred = []
y_confidence = []
file_paths = []

# ------------------------------------------------------------
# Get file paths directly from the dataset directory
# ------------------------------------------------------------

for class_index, class_name in enumerate(class_names):

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    files = sorted(
        [
            os.path.join(
                class_dir,
                file
            )
            for file in os.listdir(class_dir)
            if file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ]
    )

    for file in files:

        file_paths.append(file)
        y_true.append(class_index)


# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    confidences = np.max(
        predictions,
        axis=1
    )

    y_pred.extend(
        predicted_classes.tolist()
    )

    y_confidence.extend(
        confidences.tolist()
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_confidence = np.array(y_confidence)


# ============================================================
# CHECK LENGTHS
# ============================================================

if not (
    len(y_true)
    == len(y_pred)
    == len(file_paths)
):

    print("ERROR: Prediction/file count mismatch.")

    print(
        f"True labels : {len(y_true)}"
    )

    print(
        f"Predictions : {len(y_pred)}"
    )

    print(
        f"Files       : {len(file_paths)}"
    )

    exit()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "V3 CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(10, 10)
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues"
)

plt.title(
    "V3 Sign Language Model - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    cm_path,
    dpi=200
)

plt.close()


# ============================================================
# MISCLASSIFIED IMAGES
# ============================================================

print()
print("=" * 60)
print("MISCLASSIFIED IMAGES")
print("=" * 60)

misclassified_count = 0

for index in range(len(y_true)):

    actual = y_true[index]
    predicted = y_pred[index]

    if actual == predicted:
        continue

    misclassified_count += 1

    actual_name = class_names[actual]
    predicted_name = class_names[predicted]

    confidence = y_confidence[index]

    source_file = file_paths[index]

    destination_dir = os.path.join(
        MISCLASSIFIED_DIR,
        f"actual_{actual_name}_predicted_{predicted_name}"
    )

    os.makedirs(
        destination_dir,
        exist_ok=True
    )

    filename = os.path.basename(
        source_file
    )

    destination_file = os.path.join(
        destination_dir,
        filename
    )

    shutil.copy2(
        source_file,
        destination_file
    )

    print(
        f"{filename}: "
        f"Actual={actual_name}, "
        f"Predicted={predicted_name}, "
        f"Confidence={confidence * 100:.2f}%"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("V3 EVALUATION COMPLETE")
print("=" * 60)

print(
    f"Total test images : {len(y_true)}"
)

print(
    f"Correct           : "
    f"{np.sum(y_true == y_pred)}"
)

print(
    f"Incorrect         : "
    f"{misclassified_count}"
)

print(
    f"Accuracy          : "
    f"{np.mean(y_true == y_pred) * 100:.2f}%"
)

print()
print(f"Evaluation folder : {OUTPUT_DIR}")
print(f"Confusion matrix  : {cm_path}")
print(f"Classification    : {report_path}")
print(
    f"Misclassified     : "
    f"{MISCLASSIFIED_DIR}"
)

print("=" * 60)