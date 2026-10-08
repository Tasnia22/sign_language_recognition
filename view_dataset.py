import os
import random
import cv2
import matplotlib.pyplot as plt


DATASET_DIR = "dataset/test"

# Pick a class to inspect
CLASS_NAME = "3"

CLASS_DIR = os.path.join(
    DATASET_DIR,
    CLASS_NAME
)

images = [
    file for file in os.listdir(CLASS_DIR)
    if file.lower().endswith((".png", ".jpg", ".jpeg"))
]

random.shuffle(images)

# Show first 20 images
images = images[:20]

fig, axes = plt.subplots(
    4,
    5,
    figsize=(12, 10)
)

for ax, filename in zip(
    axes.flatten(),
    images
):

    image_path = os.path.join(
        CLASS_DIR,
        filename
    )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(filename)
    ax.axis("off")


plt.tight_layout()

plt.show()