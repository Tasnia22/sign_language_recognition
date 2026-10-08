import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "../models/sign_language_model_v3.keras"

# ============================================================
# ROI
# ============================================================

ROI_X = 400
ROI_Y = 74
ROI_WIDTH = 200
ROI_HEIGHT = 284

# ============================================================
# BACKGROUND
# ============================================================

BACKGROUND_FRAMES = 60

# ============================================================
# THRESHOLD
# ============================================================

THRESHOLD_VALUE = 25

# ============================================================
# CONTOUR QUALITY
# ============================================================

MIN_CONTOUR_AREA = 1500
MAX_CONTOUR_AREA_RATIO = 0.85

MIN_BBOX_WIDTH = 30
MIN_BBOX_HEIGHT = 30

MIN_FILL_RATIO = 0.20

# ============================================================
# FOREGROUND QUALITY
# ============================================================

MIN_FOREGROUND_RATIO = 0.05
MIN_HAND_FOREGROUND_RATIO = 0.05

# ============================================================
# PREDICTION
# ============================================================

CONFIDENCE_THRESHOLD = 0.75

SMOOTHING_WINDOW = 8

STABILITY_THRESHOLD = 0.50

# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0

# ============================================================
# WINDOWS
# ============================================================

WINDOW_NAME = "Sign Language Recognition - V3"

MASK_WINDOW_NAME = "Hand Mask - V3"


# ============================================================
# CLASS ORDER
# ============================================================

# This is the exact order returned by:
# image_dataset_from_directory()
#
# ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']

CLASS_ORDER = [
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


# ============================================================
# GESTURE NAMES
# ============================================================

GESTURE_NAMES = {

    "1": "One Finger",

    "2": "Two Fingers",

    "3": "Three Fingers",

    "4": "Four Fingers",

    "5": "Open Palm",

    "6": "Fist",

    "7": "Thumbs Up",

    "8": "Thumbs Down",

    "9": "OK",

    "10": "I Love You"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("       SIGN LANGUAGE RECOGNITION - V3")
print("=" * 65)

print("\nLoading V3 model...")


if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model not found.")

    print(
        "Expected:"
    )

    print(
        os.path.abspath(MODEL_PATH)
    )

    exit()


model = tf.keras.models.load_model(
    MODEL_PATH
)


print(
    "Model loaded successfully."
)

print(
    "Input shape :",
    model.input_shape
)

print(
    "Output shape:",
    model.output_shape
)


# ============================================================
# CAMERA
# ============================================================

print("\nOpening camera...")


cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    print(
        "\nERROR: Could not open camera."
    )

    exit()


# Use the same general camera setup
# used during dataset collection.

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# ROI FUNCTION
# ============================================================

def get_roi(frame):

    x1 = ROI_X
    y1 = ROI_Y

    x2 = ROI_X + ROI_WIDTH
    y2 = ROI_Y + ROI_HEIGHT

    roi = frame[
        y1:y2,
        x1:x2
    ]

    return roi, x1, y1, x2, y2


# ============================================================
# BACKGROUND CALIBRATION
# ============================================================

print("\n" + "=" * 65)
print("BACKGROUND CALIBRATION")
print("=" * 65)

print(
    "\nKeep your hand OUTSIDE the ROI."
)

print(
    f"ROI: X={ROI_X}, Y={ROI_Y}, "
    f"W={ROI_WIDTH}, H={ROI_HEIGHT}"
)


background_frames = []


for i in range(BACKGROUND_FRAMES):

    ret, frame = cap.read()


    if not ret:

        print(
            "\nERROR: Could not read camera frame."
        )

        cap.release()

        cv2.destroyAllWindows()

        exit()


    # ========================================================
    # IMPORTANT:
    # Same flip used by collect_dataset_v2.py
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    roi, x1, y1, x2, y2 = get_roi(
        frame
    )


    # ========================================================
    # Same grayscale operation as collector
    # ========================================================

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    background_frames.append(
        gray
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    display = frame.copy()


    cv2.rectangle(
        display,

        (x1, y1),

        (x2, y2),

        (255, 255, 255),

        2
    )


    cv2.putText(
        display,

        "BACKGROUND CALIBRATION",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2
    )


    cv2.putText(
        display,

        f"Keep hand OUT - {i + 1}/{BACKGROUND_FRAMES}",

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    cv2.imshow(
        WINDOW_NAME,
        display
    )


    if cv2.waitKey(30) & 0xFF == ord("q"):

        cap.release()

        cv2.destroyAllWindows()

        exit()


# ============================================================
# CREATE BACKGROUND
# ============================================================

background = np.median(
    np.array(background_frames),
    axis=0
).astype(np.uint8)


print(
    "\nBackground captured successfully."
)


print(
    "\nStarting realtime recognition..."
)

print(
    "Press Q to quit."
)

print("=" * 65)


# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)


last_prediction = None

last_confidence = 0.0


# ============================================================
# FPS
# ============================================================

fps = 0.0

previous_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # ========================================================
    # CAMERA FRAME
    # ========================================================

    ret, frame = cap.read()


    if not ret:

        print(
            "\nERROR: Failed to read camera frame."
        )

        break


    # ========================================================
    # SAME FLIP AS DATASET COLLECTOR
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # ROI
    # ========================================================

    roi, x1, y1, x2, y2 = get_roi(
        frame
    )


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # IMPORTANT:
    # SAME GAUSSIAN BLUR AS collect_dataset_v2.py
    # ========================================================

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # ========================================================
    # BACKGROUND DIFFERENCE
    # ========================================================

    # Same operation as collector:
    #
    # background_uint8 = cv2.convertScaleAbs(background)
    #
    # diff = cv2.absdiff(gray, background_uint8)

    background_uint8 = cv2.convertScaleAbs(
        background
    )


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

    # IMPORTANT:
    # Exactly the same kernel and iterations
    # as collect_dataset_v2.py.

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

    total_pixels = (
        threshold.shape[0] *
        threshold.shape[1]
    )


    foreground_pixels = cv2.countNonZero(
        threshold
    )


    foreground_ratio = (
        foreground_pixels /
        total_pixels
    )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    hand_detected = False

    hand_mask = np.zeros_like(
        threshold
    )

    status = "No hand detected"

    current_prediction = None

    current_confidence = 0.0


    # ========================================================
    # EMPTY MASK CHECK
    # ========================================================

    if foreground_ratio < MIN_FOREGROUND_RATIO:

        status = "Empty mask"


    else:

        # ====================================================
        # FIND CONTOURS
        # ====================================================

        contours, _ = cv2.findContours(
            threshold,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE
        )


        if contours:

            # =================================================
            # LARGEST CONTOUR
            # =================================================

            contour = max(
                contours,

                key=cv2.contourArea
            )


            contour_area = cv2.contourArea(
                contour
            )


            # =================================================
            # CONTOUR AREA CHECK
            # =================================================

            roi_area = (
                ROI_WIDTH *
                ROI_HEIGHT
            )


            if contour_area < MIN_CONTOUR_AREA:

                status = "Contour too small"


            elif (
                contour_area >
                roi_area *
                MAX_CONTOUR_AREA_RATIO
            ):

                status = "Contour too large"


            else:

                # =============================================
                # BOUNDING BOX
                # =============================================

                bx, by, bw, bh = cv2.boundingRect(
                    contour
                )


                # =============================================
                # BOUNDING BOX SIZE
                # =============================================

                if (
                    bw < MIN_BBOX_WIDTH
                    or
                    bh < MIN_BBOX_HEIGHT
                ):

                    status = "Hand too small"


                else:

                    # =========================================
                    # FILL RATIO
                    # =========================================

                    bbox_area = (
                        bw *
                        bh
                    )


                    fill_ratio = (
                        contour_area /
                        bbox_area
                    )


                    if fill_ratio < MIN_FILL_RATIO:

                        status = "Poor hand shape"


                    else:

                        # =====================================
                        # HAND REGION
                        # =====================================

                        hand_crop = threshold[
                            by:by + bh,
                            bx:bx + bw
                        ]


                        # =====================================
                        # HAND FOREGROUND RATIO
                        # =====================================

                        hand_foreground_pixels = (
                            cv2.countNonZero(
                                hand_crop
                            )
                        )


                        hand_area = (
                            bw *
                            bh
                        )


                        hand_foreground_ratio = (
                            hand_foreground_pixels /
                            hand_area
                        )


                        if (
                            hand_foreground_ratio
                            <
                            MIN_HAND_FOREGROUND_RATIO
                        ):

                            status = (
                                "Weak hand mask"
                            )


                        else:

                            # =================================
                            # HAND DETECTED
                            # =================================

                            hand_detected = True

                            status = (
                                "Hand detected"
                            )


                            # =================================
                            # IMPORTANT:
                            #
                            # EXACTLY THE SAME OPERATION
                            # AS collect_dataset_v2.py
                            # =================================

                            hand_crop = threshold[
                                by:by + bh,
                                bx:bx + bw
                            ]


                            # =================================
                            # RESIZE TO 64×64
                            # =================================

                            hand_image = cv2.resize(
                                hand_crop,

                                (64, 64),

                                interpolation=cv2.INTER_AREA
                            )


                            # =================================
                            # DEBUG MASK
                            # =================================

                            # Display the exact image that
                            # will be fed to the model.

                            hand_mask = hand_image.copy()


                            # =================================
                            # NORMALIZATION
                            # =================================

                            hand_image = (
                                hand_image.astype(
                                    np.float32
                                )
                                /
                                255.0
                            )


                            # =================================
                            # ADD CHANNEL
                            # =================================

                            hand_image = np.expand_dims(
                                hand_image,

                                axis=-1
                            )


                            # =================================
                            # ADD BATCH
                            # =================================

                            hand_image = np.expand_dims(
                                hand_image,

                                axis=0
                            )


                            # =================================
                            # MODEL PREDICTION
                            # =================================

                            predictions = model.predict(
                                hand_image,

                                verbose=0
                            )[0]


                            # =================================
                            # BEST CLASS
                            # =================================

                            predicted_index = int(
                                np.argmax(
                                    predictions
                                )
                            )


                            current_confidence = float(
                                predictions[
                                    predicted_index
                                ]
                            )


                            predicted_class = (
                                CLASS_ORDER[
                                    predicted_index
                                ]
                            )


                            current_prediction = (
                                predicted_class
                            )


                            current_gesture = (
                                GESTURE_NAMES[
                                    predicted_class
                                ]
                            )


                            # =================================
                            # TERMINAL OUTPUT
                            # =================================

                            print(
                                (
                                    f"\rPrediction: "
                                    f"{current_gesture:<15} "
                                    f"Confidence: "
                                    f"{current_confidence * 100:6.2f}%"
                                ),

                                end=""
                            )


                            # =================================
                            # CONFIDENCE FILTER
                            # =================================

                            if (
                                current_confidence
                                >=
                                CONFIDENCE_THRESHOLD
                            ):

                                prediction_history.append(
                                    predicted_class
                                )


                                # =================================
                                # STABILITY
                                # =================================

                                counts = {}


                                for prediction in (
                                    prediction_history
                                ):

                                    counts[
                                        prediction
                                    ] = (
                                        counts.get(
                                            prediction,
                                            0
                                        ) + 1
                                    )


                                stable_prediction = max(
                                    counts,

                                    key=counts.get
                                )


                                stability = (
                                    counts[
                                        stable_prediction
                                    ]
                                    /
                                    len(
                                        prediction_history
                                    )
                                )


                                if (
                                    stability
                                    >=
                                    STABILITY_THRESHOLD
                                ):

                                    last_prediction = (
                                        stable_prediction
                                    )

                                    last_confidence = (
                                        current_confidence
                                    )

                                    status = (
                                        "Recognized: "
                                        +
                                        GESTURE_NAMES[
                                            stable_prediction
                                        ]
                                    )


                            else:

                                status = (
                                    "Low confidence"
                                )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.time()

    elapsed = (
        current_time -
        previous_time
    )


    if elapsed > 0:

        current_fps = (
            1.0 /
            elapsed
        )


        if fps == 0:

            fps = current_fps

        else:

            fps = (
                0.9 * fps +
                0.1 * current_fps
            )


    previous_time = current_time


    # ========================================================
    # MAIN DISPLAY
    # ========================================================

    display = frame.copy()


    # ========================================================
    # ROI
    # ========================================================

    cv2.rectangle(
        display,

        (x1, y1),

        (x2, y2),

        (255, 255, 255),

        2
    )


    # ========================================================
    # HAND BOUNDING BOX
    # ========================================================

    if hand_detected:

        cv2.rectangle(
            display,

            (
                x1 + bx,
                y1 + by
            ),

            (
                x1 + bx + bw,
                y1 + by + bh
            ),

            (255, 0, 0),

            2
        )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        display,

        "SIGN LANGUAGE RECOGNITION - V3",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (255, 255, 255),

        2
    )


    # ========================================================
    # STATUS
    # ========================================================

    cv2.putText(
        display,

        f"Status: {status}",

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )


    # ========================================================
    # CURRENT PREDICTION
    # ========================================================

    if current_prediction is not None:

        cv2.putText(
            display,

            (
                "Model: "
                +
                GESTURE_NAMES[
                    current_prediction
                ]
            ),

            (20, 115),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (255, 255, 255),

            2
        )


        cv2.putText(
            display,

            (
                "Confidence: "
                +
                f"{current_confidence * 100:.1f}%"
            ),

            (20, 150),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )


    # ========================================================
    # STABLE PREDICTION
    # ========================================================

    if last_prediction is not None:

        cv2.putText(
            display,

            (
                "Stable: "
                +
                GESTURE_NAMES[
                    last_prediction
                ]
            ),

            (20, 190),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 255, 255),

            2
        )


        cv2.putText(
            display,

            (
                "Stable Confidence: "
                +
                f"{last_confidence * 100:.1f}%"
            ),

            (20, 225),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )


    # ========================================================
    # FOREGROUND
    # ========================================================

    cv2.putText(
        display,

        f"Foreground: {foreground_ratio * 100:.1f}%",

        (20, 260),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    # ========================================================
    # FPS
    # ========================================================

    cv2.putText(
        display,

        f"FPS: {fps:.1f}",

        (20, 295),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    # ========================================================
    # INSTRUCTIONS
    # ========================================================

    cv2.putText(
        display,

        "Press Q to quit",

        (20, display.shape[0] - 25),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    # ========================================================
    # SHOW WINDOWS
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display
    )


    cv2.imshow(
        MASK_WINDOW_NAME,
        hand_mask
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print(
    "\n\nRealtime prediction stopped."
)

print(
    "Goodbye!"
)