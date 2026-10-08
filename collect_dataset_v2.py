import cv2
import numpy as np
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================

GESTURE = "10"
GESTURE_NAME = "YO YO"

SAVE_DIR = f"dataset_v2/raw/{GESTURE}"
TARGET_IMAGES = 500

CAMERA_INDEX = 0

# ---------------- ROI ----------------
ROI_X = 400
ROI_Y = 74
ROI_WIDTH = 200
ROI_HEIGHT = 284

# ---------------- Background ----------------
BACKGROUND_FRAMES = 60

# ---------------- Threshold ----------------
THRESHOLD_VALUE = 25

# ---------------- Contour Quality ----------------
MIN_CONTOUR_AREA = 1500
MAX_CONTOUR_AREA_RATIO = 0.85

MIN_BBOX_WIDTH = 30
MIN_BBOX_HEIGHT = 30

MIN_FILL_RATIO = 0.20

# ---------------- Foreground Quality ----------------
# Reject almost completely black masks
MIN_FOREGROUND_RATIO = 0.05

# Minimum foreground ratio inside the detected hand bounding box
MIN_HAND_FOREGROUND_RATIO = 0.05

# ============================================================
# CREATE DIRECTORY
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("\n==============================================")
print("       SIGN LANGUAGE DATASET COLLECTOR V2")
print("==============================================")
print(f"Gesture       : {GESTURE} - {GESTURE_NAME}")
print(f"Target images : {TARGET_IMAGES}")
print(f"Save location : {SAVE_DIR}")
print()
print("ROI:")
print(f"X      : {ROI_X}")
print(f"Y      : {ROI_Y}")
print(f"Width  : {ROI_WIDTH}")
print(f"Height : {ROI_HEIGHT}")
print()
print("Press Q at any time to quit.")
print("==============================================\n")

# ============================================================
# ROI FUNCTION
# ============================================================

def get_roi(frame):
    x1 = ROI_X
    y1 = ROI_Y

    x2 = ROI_X + ROI_WIDTH
    y2 = ROI_Y + ROI_HEIGHT

    roi = frame[y1:y2, x1:x2]

    return roi, x1, y1, x2, y2


# ============================================================
# CAPTURE BACKGROUND
# ============================================================

print("Step 1: Move your hand OUTSIDE the ROI.")
print("Capturing background...")

background_frames = []

for i in range(BACKGROUND_FRAMES):

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    frame = cv2.flip(frame, 1)

    roi, x1, y1, x2, y2 = get_roi(frame)

    background_frames.append(
        cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    )

    display = frame.copy()

    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Capturing background: {i + 1}/{BACKGROUND_FRAMES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Dataset Collection", display)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

background = np.median(
    np.array(background_frames),
    axis=0
).astype(np.uint8)

print("Background captured successfully.\n")

# ============================================================
# WAIT BEFORE COLLECTION
# ============================================================

print("Step 2: Place your hand inside the ROI.")
print("Starting collection in 3 seconds...")

for countdown in [3, 2, 1]:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    roi, x1, y1, x2, y2 = get_roi(frame)

    display = frame.copy()

    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Starting in {countdown}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("Dataset Collection", display)

    if cv2.waitKey(1000) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# ============================================================
# COLLECTION
# ============================================================

saved_count = 0
rejected_count = 0

last_save_time = 0

SAVE_INTERVAL = 0.10

print("\nCollecting images...\n")

while saved_count < TARGET_IMAGES:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    roi, x1, y1, x2, y2 = get_roi(frame)

    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ========================================================
    # BACKGROUND DIFFERENCE
    # ========================================================

    background_uint8 = cv2.convertScaleAbs(background)

    diff = cv2.absdiff(
        gray,
        background_uint8
    )

    # ========================================================
    # THRESHOLD
    # ========================================================

    _, threshold = cv2.threshold(
        diff,
        THRESHOLD_VALUE,
        255,
        cv2.THRESH_BINARY
    )

    # ========================================================
    # MORPHOLOGICAL CLEANUP
    # ========================================================

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    threshold = cv2.dilate(
        threshold,
        kernel,
        iterations=2
    )

    # ========================================================
    # FOREGROUND QUALITY CHECK
    # ========================================================

    total_pixels = threshold.shape[0] * threshold.shape[1]

    foreground_pixels = cv2.countNonZero(threshold)

    foreground_ratio = (
        foreground_pixels / total_pixels
    )

    # Reject almost completely empty masks
    if foreground_ratio < MIN_FOREGROUND_RATIO:

        rejected_count += 1

        status = "REJECTED: EMPTY"

        display = frame.copy()

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            display,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.imshow("Dataset Collection", display)
        cv2.imshow("Hand Mask", threshold)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        continue

    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        rejected_count += 1

        continue

    # Largest contour
    contour = max(
        contours,
        key=cv2.contourArea
    )

    contour_area = cv2.contourArea(contour)

    # ========================================================
    # CONTOUR AREA CHECK
    # ========================================================

    if contour_area < MIN_CONTOUR_AREA:

        rejected_count += 1
        continue

    roi_area = ROI_WIDTH * ROI_HEIGHT

    if contour_area > roi_area * MAX_CONTOUR_AREA_RATIO:

        rejected_count += 1
        continue

    # ========================================================
    # BOUNDING BOX
    # ========================================================

    bx, by, bw, bh = cv2.boundingRect(contour)

    if bw < MIN_BBOX_WIDTH or bh < MIN_BBOX_HEIGHT:

        rejected_count += 1
        continue

    # ========================================================
    # FILL RATIO
    # ========================================================

    bbox_area = bw * bh

    fill_ratio = contour_area / bbox_area

    if fill_ratio < MIN_FILL_RATIO:

        rejected_count += 1
        continue

    # ========================================================
    # HAND MASK
    # ========================================================

    hand_mask = threshold[
        by:by + bh,
        bx:bx + bw
    ]

    hand_foreground_pixels = cv2.countNonZero(
        hand_mask
    )

    hand_area = bw * bh

    hand_foreground_ratio = (
        hand_foreground_pixels / hand_area
    )

    # Reject weak hand masks
    if hand_foreground_ratio < MIN_HAND_FOREGROUND_RATIO:

        rejected_count += 1
        continue

    # ========================================================
    # CROP HAND
    # ========================================================

    hand_crop = threshold[
        by:by + bh,
        bx:bx + bw
    ]

    # ========================================================
    # RESIZE
    # ========================================================

    hand_crop = cv2.resize(
        hand_crop,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    # ========================================================
    # SAVE
    # ========================================================

    current_time = time.time()

    if current_time - last_save_time >= SAVE_INTERVAL:

        filename = os.path.join(
            SAVE_DIR,
            f"{saved_count + 1:04d}.png"
        )

        cv2.imwrite(
            filename,
            hand_crop
        )

        saved_count += 1
        last_save_time = current_time

    # ========================================================
    # DISPLAY
    # ========================================================

    display = frame.copy()

    # ROI
    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Hand bounding box
    cv2.rectangle(
        display,
        (x1 + bx, y1 + by),
        (x1 + bx + bw, y1 + by + bh),
        (255, 0, 0),
        2
    )

    cv2.putText(
        display,
        f"Saved: {saved_count}/{TARGET_IMAGES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display,
        f"Rejected: {rejected_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        display,
        f"Foreground: {foreground_ratio:.2f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Dataset Collection",
        display
    )

    cv2.imshow(
        "Hand Mask",
        threshold
    )

    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\n==============================================")
print("          COLLECTION COMPLETE")
print("==============================================")
print(f"Gesture       : {GESTURE} - {GESTURE_NAME}")
print(f"Saved images  : {saved_count}")
print(f"Rejected      : {rejected_count}")
print(f"Location      : {SAVE_DIR}")
print("==============================================")