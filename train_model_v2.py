import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 40
NUM_CLASSES = 10

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"
TEST_DIR = "dataset/test"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sign_language_model_v2.keras"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
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

print("\nLoading test dataset...")

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

# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses:")

for index, name in enumerate(class_names):

    print(
        f"{index}: {name}"
    )

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    
    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        height_factor=0.10,
        width_factor=0.10
    ),

    layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08
    ),

])

# ============================================================
# NORMALIZATION
# ============================================================

normalization_layer = layers.Rescaling(
    1.0 / 255
)

# ============================================================
# TRAINING PIPELINE
# ============================================================

train_dataset = train_dataset.map(
    lambda images, labels: (
        data_augmentation(
            normalization_layer(images),
            training=True
        ),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

# ============================================================
# VALIDATION PIPELINE
# ============================================================

validation_dataset = validation_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

# ============================================================
# TEST PIPELINE
# ============================================================

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

# ============================================================
# PREFETCH
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# ============================================================
# CNN MODEL
# ============================================================

model = models.Sequential([

    layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        )
    ),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CNN MODEL")
print("=" * 60)

model.summary()

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]

)

# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True

)

model_checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]

)

# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"\nModel saved to:\n"
    f"{MODEL_PATH}"
)