import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "dataset_v2/train"
VALIDATION_DIR = "dataset_v2/validation"
TEST_DIR = "dataset_v2/test"

MODEL_PATH = "models/sign_language_model_v3.keras"

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 40

# ============================================================
# CHECK DIRECTORIES
# ============================================================

for directory in [
    TRAIN_DIR,
    VALIDATION_DIR,
    TEST_DIR
]:
    if not os.path.exists(directory):
        print(f"ERROR: Directory not found: {directory}")
        exit()

os.makedirs("models", exist_ok=True)

# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 60)
print("LOADING V3 DATASET")
print("=" * 60)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()
print("Class names:")
print(train_dataset.class_names)

print()

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
# BUILD MODEL
# ============================================================

print("=" * 60)
print("BUILDING V3 MODEL")
print("=" * 60)

model = models.Sequential([

    # Input normalization
    layers.Rescaling(
        1.0 / 255,
        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        )
    ),

    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.5
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])

# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# CALLBACKS
# ============================================================

early_stopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("STARTING V3 TRAINING")
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
# FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 60)
print("EVALUATING V3 MODEL")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print()
print("=" * 60)
print("V3 TRAINING COMPLETE")
print("=" * 60)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)

print(
    f"Model saved   : {MODEL_PATH}"
)

print("=" * 60)