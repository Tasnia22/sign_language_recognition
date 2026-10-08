import cv2
import os
import random


# ==========================================
# SETTINGS
# ==========================================

GESTURE = "1"

DATASET_PATH = os.path.join(
    "dataset_v2",
    "raw",
    GESTURE
)

SAMPLE_COUNT = 25

IMAGE_SIZE = 64


# ==========================================
# GET IMAGES
# ==========================================

images = [
    file
    for file in os.listdir(DATASET_PATH)
    if file.lower().endswith(".png")
]

if len(images) == 0:
    print("❌ No images found.")
    exit()


# Select random samples
sample_count = min(
    SAMPLE_COUNT,
    len(images)
)

selected = random.sample(
    images,
    sample_count
)


# ==========================================
# DISPLAY
# ==========================================

for index, filename in enumerate(selected):

    path = os.path.join(
        DATASET_PATH,
        filename
    )

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        continue


    # Enlarge for easier inspection
    display = cv2.resize(
        image,
        (
            IMAGE_SIZE * 5,
            IMAGE_SIZE * 5
        ),
        interpolation=cv2.INTER_NEAREST
    )


    cv2.putText(
        display,
        f"{index + 1}/{sample_count}  {filename}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        255,
        2
    )


    cv2.imshow(
        "Dataset V2 Inspection",
        display
    )


    key = cv2.waitKey(0) & 0xFF


    # ESC = exit
    if key == 27:
        break


cv2.destroyAllWindows()