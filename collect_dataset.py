import cv2
import numpy as np
import os
import time
import random
import shutil


# ==========================================
# SETTINGS
# ==========================================

CAMERA_INDEX = 0

IMAGE_SIZE = 64

IMAGES_PER_CLASS = 500

BACKGROUND_FRAMES = 60

ROI_X1_RATIO = 0.55
ROI_Y1_RATIO = 0.20
ROI_X2_RATIO = 0.95
ROI_Y2_RATIO = 0.80


# ==========================================
# SELECT GESTURE
# ==========================================

gesture = input(
    "\nEnter gesture number (1-10): "
).strip()

if gesture not in [str(i) for i in range(1, 11)]:
    print("❌ Gesture must be between 1 and 10.")
    exit()


# ==========================================
# CREATE DIRECTORY
# ==========================================

save_directory = os.path.join(
    "dataset",
    "raw",
    gesture
)

os.makedirs(
    save_directory,
    exist_ok=True
)

os.makedirs(
    save_directory,
    exist_ok=True
)


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)

if not camera.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Camera opened successfully.")


# ==========================================
# BACKGROUND
# ==========================================

background = None

frame_count = 0

print()
print("======================================")
print("BACKGROUND CAPTURE")
print("======================================")
print()
print("Keep the ROI EMPTY.")
print("Do not place your hand inside.")
print()


# ==========================================
# CAPTURE BACKGROUND
# ==========================================

while frame_count < BACKGROUND_FRAMES:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read frame.")
        camera.release()
        exit()

    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape

    x1 = int(
        width * ROI_X1_RATIO
    )

    y1 = int(
        height * ROI_Y1_RATIO
    )

    x2 = int(
        width * ROI_X2_RATIO
    )

    y2 = int(
        height * ROI_Y2_RATIO
    )

    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    if background is None:

        background = gray.astype(
            np.float32
        )

    else:

        cv2.accumulateWeighted(
            gray,
            background,
            0.5
        )

    frame_count += 1

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Background: "
        f"{frame_count}/{BACKGROUND_FRAMES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Dataset Collection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        camera.release()
        cv2.destroyAllWindows()
        exit()


# ==========================================
# PREPARE FOR COLLECTION
# ==========================================

print()
print("======================================")
print(f"GESTURE: {gesture}")
print("======================================")
print()
print("Place your hand inside the ROI.")
print("Press SPACE to start collecting.")
print("Press ESC to cancel.")
print()


while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape

    x1 = int(
        width * ROI_X1_RATIO
    )

    y1 = int(
        height * ROI_Y1_RATIO
    )

    x2 = int(
        width * ROI_X2_RATIO
    )

    y2 = int(
        height * ROI_Y2_RATIO
    )

    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press SPACE to start",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Dataset Collection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == 32:
        break

    if key == 27:
        camera.release()
        cv2.destroyAllWindows()
        exit()


# ==========================================
# COLLECT IMAGES
# ==========================================

print()
print("📸 Collecting images...")
print()


image_count = 0

last_save_time = 0

SAVE_INTERVAL = 0.08


while image_count < IMAGES_PER_CLASS:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read frame.")
        break

    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape

    x1 = int(
        width * ROI_X1_RATIO
    )

    y1 = int(
        height * ROI_Y1_RATIO
    )

    x2 = int(
        width * ROI_X2_RATIO
    )

    y2 = int(
        height * ROI_Y2_RATIO
    )

    roi = frame[y1:y2, x1:x2]


    # ======================================
    # GRAYSCALE
    # ======================================

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )


    # ======================================
    # BACKGROUND DIFFERENCE
    # ======================================

    difference = cv2.absdiff(
        background.astype(
            np.uint8
        ),
        gray
    )


    # ======================================
    # THRESHOLD
    # ======================================

    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )


    # ======================================
    # MORPHOLOGICAL CLEANUP
    # ======================================

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_DILATE,
        kernel
    )


    # ======================================
    # CONTOURS
    # ======================================

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) > 0:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            largest_contour
        )


        # Ignore tiny objects
        if area > 1000:

            x, y, w, h = cv2.boundingRect(
                largest_contour
            )


            # ==================================
            # CROP HAND
            # ==================================

            hand = threshold[
                y:y + h,
                x:x + w
            ]


            if hand.size > 0:

                # Resize
                hand = cv2.resize(
                    hand,
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    )
                )


                # ==================================
                # SAVE IMAGE
                # ==================================

                current_time = time.time()

                if (
                    current_time -
                    last_save_time
                ) >= SAVE_INTERVAL:

                    filename = os.path.join(
                        save_directory,
                        f"{image_count + 1:04d}.png"
                    )

                    cv2.imwrite(
                        filename,
                        hand
                    )

                    image_count += 1

                    last_save_time = current_time


    # ======================================
    # DISPLAY
    # ======================================

    display = cv2.cvtColor(
        threshold,
        cv2.COLOR_GRAY2BGR
    )

    cv2.putText(
        display,
        f"Gesture: {gesture}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Images: "
        f"{image_count}/{IMAGES_PER_CLASS}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Hand Dataset",
        display
    )


    # ======================================
    # ESC
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        print("\nCollection cancelled.")
        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()
cv2.destroyAllWindows()


# ==========================================
# SPLIT DATASET
# ==========================================

print()
print("======================================")
print("CREATING TRAIN / TEST DATASET")
print("======================================")


raw_directory = os.path.join(
    "dataset",
    "raw",
    gesture
)

train_directory = os.path.join(
    "dataset",
    "train",
    gesture
)

test_directory = os.path.join(
    "dataset",
    "test",
    gesture
)


os.makedirs(
    train_directory,
    exist_ok=True
)

os.makedirs(
    test_directory,
    exist_ok=True
)


# Get all images
images = [
    file
    for file in os.listdir(raw_directory)
    if file.lower().endswith(".png")
]


# Shuffle images
random.shuffle(images)


# Calculate split
train_count = int(
    len(images) * 0.8
)


train_images = images[
    :train_count
]

test_images = images[
    train_count:
]


# Move training images
for image in train_images:

    source = os.path.join(
        raw_directory,
        image
    )

    destination = os.path.join(
        train_directory,
        image
    )

    shutil.move(
        source,
        destination
    )


# Move test images
for image in test_images:

    source = os.path.join(
        raw_directory,
        image
    )

    destination = os.path.join(
        test_directory,
        image
    )

    shutil.move(
        source,
        destination
    )


# Remove empty raw directory
try:
    os.rmdir(raw_directory)
except OSError:
    pass


print()
print(f"Total images : {len(images)}")
print(f"Training     : {len(train_images)}")
print(f"Testing      : {len(test_images)}")

print()
print("Dataset preparation complete!")