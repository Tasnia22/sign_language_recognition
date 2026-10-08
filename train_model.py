import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 10

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_language_model.keras")

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading training dataset...")

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

print("Loading testing dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DISPLAY CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nClasses:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")


# ============================================================
# NORMALIZE PIXEL VALUES
# ============================================================

normalization_layer = layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# ============================================================
# IMPROVE PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# ============================================================
# BUILD CNN MODEL
# ============================================================

model = models.Sequential([
    
    # Input
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

    # Block 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    # Output
    layers.Dense(NUM_CLASSES, activation="softmax")
])


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\nCNN Model:")
model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(test_dataset)

print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

print(f"\nModel saved to:")
print(MODEL_PATH)