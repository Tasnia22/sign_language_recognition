import os
import random
import cv2
import numpy as np

DATASET_DIR = "dataset_v2/raw"
CLASS_NAME = "3"

SAMPLE_COUNT = 25
GRID_COLS = 5
IMAGE_SIZE = 64

class_dir = os.path.join(DATASET_DIR, CLASS_NAME)

files = [
    f for f in os.listdir(class_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not files:
    print("No images found.")
    exit()

sample_count = min(SAMPLE_COUNT, len(files))

samples = random.sample(files, sample_count)

GRID_ROWS = int(np.ceil(sample_count / GRID_COLS))

grid = np.zeros(
    (
        GRID_ROWS * IMAGE_SIZE,
        GRID_COLS * IMAGE_SIZE
    ),
    dtype=np.uint8
)

for index, filename in enumerate(samples):

    path = os.path.join(
        class_dir,
        filename
    )

    image = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        continue

    row = index // GRID_COLS
    col = index % GRID_COLS

    y1 = row * IMAGE_SIZE
    y2 = y1 + IMAGE_SIZE

    x1 = col * IMAGE_SIZE
    x2 = x1 + IMAGE_SIZE

    grid[y1:y2, x1:x2] = image

# Scale up for easier inspection
display_size = GRID_COLS * 160

grid_large = cv2.resize(
    grid,
    (display_size, GRID_ROWS * 160),
    interpolation=cv2.INTER_NEAREST
)

cv2.imshow(
    f"V2 Dataset - Class {CLASS_NAME}",
    grid_large
)

print("=" * 50)
print(f"CLASS {CLASS_NAME} DATASET INSPECTION")
print("=" * 50)
print(f"Total images : {len(files)}")
print(f"Displayed    : {sample_count}")
print()
print("Press Q or ESC to close.")

while True:

    key = cv2.waitKey(0) & 0xFF

    if key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()