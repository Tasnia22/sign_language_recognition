import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 64
BATCH_SIZE = 32

TEST_DIR = "dataset/test"
MODEL_PATH = "models/sign_language_model_v2.keras"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

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

print("\nClasses:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")


# ============================================================
# NORMALIZE IMAGES
# ============================================================

test_dataset = test_dataset.map(
    lambda images, labels: (
        tf.cast(images, tf.float32) / 255.0,
        labels
    )
)

test_dataset = test_dataset.prefetch(
    buffer_size=tf.data.AUTOTUNE
)


# ============================================================
# GET TRUE LABELS AND PREDICTIONS
# ============================================================

print("\nMaking predictions...")

true_labels = []
predicted_labels = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(labels.numpy())
    predicted_labels.extend(predicted)

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = np.mean(
    true_labels == predicted_labels
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Overall Accuracy: {accuracy * 100:.2f}%")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)

for i, class_name in enumerate(class_names):

    total = np.sum(true_labels == i)

    correct = np.sum(
        (true_labels == i) &
        (predicted_labels == i)
    )

    class_accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"Class {class_name:>2}: "
        f"{correct:>3}/{total:<3} "
        f"({class_accuracy * 100:.2f}%)"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

os.makedirs("models/evaluation", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(10, 10)
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Sign Language Recognition - Confusion Matrix")
plt.tight_layout()

confusion_path = (
    "models/evaluation/confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=200
)

plt.close()

print(
    f"\nConfusion matrix saved to:\n"
    f"{confusion_path}"
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)