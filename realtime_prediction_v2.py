print(">>> REALTIME PREDICTION V2 STARTED <<<")

import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque, Counter


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

IMAGE_SIZE = 64

MODEL_PATH = "models/sign_language_model_v2.keras"

CONFIDENCE_THRESHOLD = 75.0

PREDICTION_HISTORY_SIZE = 10

BACKGROUND_FRAMES = 60


# ============================================================
# CALIBRATED ROI
# ============================================================

ROI_X = 400
ROI_Y = 74

ROI_WIDTH = 200
ROI_HEIGHT = 284


# ============================================================
# CONTOUR QUALITY SETTINGS
# Same philosophy as collect_dataset_v2.py
# ============================================================

MIN_CONTOUR_AREA = 1500

MAX_CONTOUR_AREA_RATIO = 0.85

MIN_BBOX_WIDTH = 30

MIN_BBOX_HEIGHT = 30

MIN_FILL_RATIO = 0.20


# ============================================================
# GESTURE CLASS ORDER
#
# IMPORTANT:
# This follows the class order used by the existing model.
# ============================================================

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


# ============================================================
# GESTURE NAMES
# ============================================================

GESTURE_NAMES = {

    "1": "One Finger",

    "2": "Two Fingers",

    "3": "Three Fingers",

    "4": "Four Fingers",

    "5": "Open Palm",

    "6": "Six",

    "7": "Call me",

    "8": "Nice",

    "9": "OK",

    "10": "Yo Yo"
}


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("============================================================")
print("SIGN LANGUAGE RECOGNITION - V2")
print("============================================================")
print()

print("Loading V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("✅ Model loaded successfully.")

print(
    f"Input shape  : {model.input_shape}"
)

print(
    f"Output shape : {model.output_shape}"
)


# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = deque(
    maxlen=PREDICTION_HISTORY_SIZE
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)

if not camera.isOpened():

    print()
    print("❌ Could not open camera.")

    exit()


print("✅ Camera opened successfully.")


# ============================================================
# BACKGROUND CAPTURE
# ============================================================

print()
print("============================================================")
print("BACKGROUND CAPTURE")
print("============================================================")
print()
print("Keep the ROI completely EMPTY.")
print("Do not place your hand inside.")
print()


background = None

frame_count = 0


while frame_count < BACKGROUND_FRAMES:

    success, frame = camera.read()

    if not success:

        print("❌ Could not read frame.")

        camera.release()

        cv2.destroyAllWindows()

        exit()


    # Mirror camera
    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    x1 = ROI_X
    y1 = ROI_Y

    x2 = ROI_X + ROI_WIDTH
    y2 = ROI_Y + ROI_HEIGHT


    roi = frame[
        y1:y2,
        x1:x2
    ]


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )


    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # DISPLAY
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
        f"Background: "
        f"{frame_count}/{BACKGROUND_FRAMES}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Sign Language Recognition V2",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        camera.release()

        cv2.destroyAllWindows()

        exit()


print()
print("✅ Background captured successfully.")
print()
print("Place your hand inside the ROI.")
print("Press ESC to exit.")
print()


# ============================================================
# FPS
# ============================================================

previous_time = time.time()

fps = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()


    if not success:

        print("❌ Could not read frame.")

        break


    # ========================================================
    # MIRROR
    # ========================================================

    frame = cv2.flip(
        frame,
        1
    )


    height, width, _ = frame.shape


    # ========================================================
    # ROI
    # ========================================================

    x1 = ROI_X
    y1 = ROI_Y

    x2 = ROI_X + ROI_WIDTH
    y2 = ROI_Y + ROI_HEIGHT


    roi = frame[
        y1:y2,
        x1:x2
    ]


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # BLUR
    # ========================================================

    gray = cv2.GaussianBlur(
        gray,
        (7, 7),
        0
    )


    # ========================================================
    # BACKGROUND DIFFERENCE
    # ========================================================

    difference = cv2.absdiff(
        background.astype(
            np.uint8
        ),
        gray
    )


    # ========================================================
    # THRESHOLD
    # ========================================================

    _, threshold = cv2.threshold(
        difference,
        25,
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
        kernel
    )


    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_DILATE,
        kernel
    )


    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    prediction_text = "No hand detected"

    confidence_text = ""

    status_text = "WAITING"

    quality_message = ""

    detected_contour = None


    # ========================================================
    # CONTOUR DETECTION
    # ========================================================

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if len(contours) > 0:

        # ----------------------------------------------------
        # Find largest contour
        # ----------------------------------------------------

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )


        contour_area = cv2.contourArea(
            largest_contour
        )


        roi_area = (
            threshold.shape[0]
            *
            threshold.shape[1]
        )


        area_ratio = (
            contour_area /
            roi_area
        )


        # ====================================================
        # QUALITY CHECK 1
        # AREA
        # ====================================================

        if contour_area < MIN_CONTOUR_AREA:

            quality_message = "TOO SMALL"

            status_text = "REJECTED"


        elif area_ratio > MAX_CONTOUR_AREA_RATIO:

            quality_message = "TOO LARGE"

            status_text = "REJECTED"


        else:

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            hx, hy, hw, hh = cv2.boundingRect(
                largest_contour
            )


            # =================================================
            # QUALITY CHECK 2
            # BOUNDING BOX
            # =================================================

            if (
                hw < MIN_BBOX_WIDTH
                or
                hh < MIN_BBOX_HEIGHT
            ):

                quality_message = "BAD SIZE"

                status_text = "REJECTED"


            else:

                # =================================================
                # QUALITY CHECK 3
                # FILL RATIO
                # =================================================

                bbox_area = (
                    hw *
                    hh
                )


                fill_ratio = (
                    contour_area /
                    bbox_area
                )


                if fill_ratio < MIN_FILL_RATIO:

                    quality_message = "BAD SHAPE"

                    status_text = "REJECTED"


                else:

                    # =================================================
                    # VALID HAND
                    # =================================================

                    detected_contour = (
                        largest_contour
                    )


                    status_text = "GOOD HAND"


                    # =================================================
                    # BOUNDING BOX
                    # =================================================

                    cv2.rectangle(
                        roi,
                        (hx, hy),
                        (
                            hx + hw,
                            hy + hh
                        ),
                        (0, 255, 0),
                        2
                    )


                    # =================================================
                    # HAND CROP
                    # =================================================

                    hand = threshold[
                        hy:hy + hh,
                        hx:hx + hw
                    ]


                    if hand.size > 0:

                        # =================================================
                        # RESIZE
                        # =================================================

                        hand = cv2.resize(
                            hand,
                            (
                                IMAGE_SIZE,
                                IMAGE_SIZE
                            )
                        )


                        # =================================================
                        # NORMALIZE
                        # =================================================

                        hand = (
                            hand.astype(
                                np.float32
                            )
                            /
                            255.0
                        )


                        # =================================================
                        # ADD CHANNEL
                        # =================================================

                        hand = np.expand_dims(
                            hand,
                            axis=-1
                        )


                        # =================================================
                        # ADD BATCH
                        # =================================================

                        hand = np.expand_dims(
                            hand,
                            axis=0
                        )


                        # =================================================
                        # MODEL PREDICTION
                        # =================================================

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
                            ]
                            *
                            100
                        )


                        # =================================================
                        # HISTORY
                        # =================================================

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
                            ]
                            *
                            100
                        )


                        # =================================================
                        # CONFIDENCE
                        # =================================================

                        if (
                            smoothed_confidence
                            >=
                            CONFIDENCE_THRESHOLD
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
    # DRAW CONTOUR
    # ========================================================

    if detected_contour is not None:

        cv2.drawContours(
            roi,
            [detected_contour],
            -1,
            (0, 0, 255),
            2
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

        fps = (
            1 /
            elapsed
        )


    previous_time = current_time


    # ========================================================
    # MAIN UI
    # ========================================================

    cv2.putText(
        frame,
        "SIGN LANGUAGE RECOGNITION V2",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Status: {status_text}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # ========================================================
    # QUALITY MESSAGE
    # ========================================================

    if quality_message != "":

        cv2.putText(
            frame,
            f"Quality: {quality_message}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2
        )


    # ========================================================
    # GESTURE
    # ========================================================

    cv2.putText(
        frame,
        f"Gesture: {prediction_text}",
        (20, height - 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    cv2.putText(
        frame,
        f"Confidence: {confidence_text}",
        (20, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # FPS
    # ========================================================

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


    # ========================================================
    # ROI BORDER
    # ========================================================

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Sign Language Recognition V2",
        frame
    )


    cv2.imshow(
        "Hand Mask V2",
        threshold
    )


    # ========================================================
    # EXIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()


print()
print("============================================================")
print("REALTIME RECOGNITION STOPPED")
print("============================================================")