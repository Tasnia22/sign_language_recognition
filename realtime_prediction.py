import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque, Counter

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 64

MODEL_PATH = "models/sign_language_model.keras"

CONFIDENCE_THRESHOLD = 75.0

PREDICTION_HISTORY_SIZE = 10

CLASS_NAMES = [
    "1",
    "10",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

GESTURE_NAMES = {
    "1": "One Finger",
    "2": "Two Fingers",
    "3": "Three Fingers",
    "4": "Four Fingers",
    "5": "Open Palm",
    "6": "Thumbs Up",
    "7": "Call me",
    "8": "Nice",
    "9": "OK",
    "10": "Yo Yo"
}

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = deque(
    maxlen=PREDICTION_HISTORY_SIZE
)

# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("❌ Could not open camera.")

    exit()

print("Camera opened successfully!")

# ============================================================
# BACKGROUND CAPTURE
# ============================================================

print("\nKeep the ROI empty.")
print("Capturing background...")

background = None

for i in range(60):

    success, frame = camera.read()

    if not success:
        continue

    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape

    x1 = int(width * 0.55)
    y1 = int(height * 0.20)

    x2 = int(width * 0.95)
    y2 = int(height * 0.80)

    roi = frame[
        y1:y2,
        x1:x2
    ]

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
            "float"
        )

    else:

        cv2.accumulateWeighted(
            gray,
            background,
            0.5
        )

    # --------------------------------------------------------
    # Background capture screen
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Keep ROI empty",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Background: {i + 1}/60",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Sign Language Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:

        camera.release()

        cv2.destroyAllWindows()

        exit()

print("Background captured!")

# ============================================================
# FPS VARIABLES
# ============================================================

previous_time = time.time()

fps = 0

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:

        print(
            "❌ Could not read frame."
        )

        break

    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )

    height, width, _ = frame.shape

    # ========================================================
    # ROI
    # ========================================================

    x1 = int(width * 0.55)
    y1 = int(height * 0.20)

    x2 = int(width * 0.95)
    y2 = int(height * 0.80)

    roi = frame[
        y1:y2,
        x1:x2
    ]

    # ========================================================
    # SEGMENTATION
    # ========================================================

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )

    difference = cv2.absdiff(
        gray,
        cv2.convertScaleAbs(
            background
        )
    )

    _, threshold = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # Remove noise
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
    # DEFAULT DISPLAY VALUES
    # ========================================================

    prediction_text = "No hand detected"

    confidence_text = ""

    status_text = "WAITING"

    # ========================================================
    # CONTOUR DETECTION
    # ========================================================

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(
            largest_contour
        )

        if area > 1000:

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            hx, hy, hw, hh = cv2.boundingRect(
                largest_contour
            )

            cv2.drawContours(
                roi,
                [largest_contour],
                -1,
                (0, 0, 255),
                2
            )

            cv2.rectangle(
                roi,
                (hx, hy),
                (hx + hw, hy + hh),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # HAND CROP
            # ------------------------------------------------

            hand = threshold[
                hy:hy + hh,
                hx:hx + hw
            ]

            if hand.size > 0:

                hand = cv2.resize(
                    hand,
                    (
                        IMAGE_SIZE,
                        IMAGE_SIZE
                    )
                )

                hand = (
                    hand.astype(
                        np.float32
                    ) / 255.0
                )

                hand = np.expand_dims(
                    hand,
                    axis=-1
                )

                hand = np.expand_dims(
                    hand,
                    axis=0
                )

                # ------------------------------------------------
                # CNN PREDICTION
                # ------------------------------------------------

                predictions = model.predict(
                    hand,
                    verbose=0
                )

                current_index = np.argmax(
                    predictions[0]
                )

                current_confidence = (
                    predictions[0][
                        current_index
                    ] * 100
                )

                # ------------------------------------------------
                # SMOOTHING
                # ------------------------------------------------

                prediction_history.append(
                    current_index
                )

                most_common = Counter(
                    prediction_history
                ).most_common(1)[0]

                smoothed_index = (
                    most_common[0]
                )

                smoothed_class = (
                    CLASS_NAMES[
                        smoothed_index
                    ]
                )

                gesture_name = (
                    GESTURE_NAMES[
                        smoothed_class
                    ]
                )

                smoothed_confidence = (
                    predictions[0][
                        smoothed_index
                    ] * 100
                )

                # ------------------------------------------------
                # CONFIDENCE CHECK
                # ------------------------------------------------

                if (
                    smoothed_confidence
                    >= CONFIDENCE_THRESHOLD
                ):

                    prediction_text = (
                        gesture_name
                    )

                    confidence_text = (
                        f"{smoothed_confidence:.1f}%"
                    )

                    status_text = (
                        "RECOGNIZED"
                    )

                else:

                    prediction_text = (
                        "Unknown"
                    )

                    confidence_text = (
                        f"{smoothed_confidence:.1f}%"
                    )

                    status_text = (
                        "LOW CONFIDENCE"
                    )

    # ========================================================
    # FPS CALCULATION
    # ========================================================

    current_time = time.time()

    elapsed = (
        current_time -
        previous_time
    )

    if elapsed > 0:

        fps = 1 / elapsed

    previous_time = current_time

    # ========================================================
    # UI
    # ========================================================

    # Main title
    cv2.putText(
        frame,
        "SIGN LANGUAGE RECOGNITION",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Status
    cv2.putText(
        frame,
        f"Status: {status_text}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # Gesture label
    cv2.putText(
        frame,
        f"Gesture: {prediction_text}",
        (20, height - 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Confidence
    cv2.putText(
        frame,
        f"Confidence: {confidence_text}",
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (
            width - 140,
            40
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ROI border
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # ========================================================
    # DISPLAY WINDOWS
    # ========================================================

    cv2.imshow(
        "Sign Language Recognition",
        frame
    )

    cv2.imshow(
        "Hand Mask",
        threshold
    )

    # ========================================================
    # EXIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == 27:

        break

# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()