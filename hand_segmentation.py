import cv2
import numpy as np

# ---------------------------------------
# Camera setup
# ---------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Camera opened successfully.")


# ---------------------------------------
# Background variables
# ---------------------------------------

background = None

# Number of frames used to build background
background_frames = 60

print()
print("Keep the ROI empty.")
print(f"Capturing background for {background_frames} frames...")


# ---------------------------------------
# Main loop
# ---------------------------------------

frame_count = 0

while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read frame.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # ---------------------------------------
    # Define ROI
    # ---------------------------------------

    x1 = int(width * 0.55)
    y1 = int(height * 0.20)

    x2 = int(width * 0.95)
    y2 = int(height * 0.80)

    roi = frame[y1:y2, x1:x2]

    # ---------------------------------------
    # Capture background
    # ---------------------------------------

    if frame_count < background_frames:

        # Convert ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if background is None:
            background = gray.astype(np.float32)
        else:
            # Accumulate background
            cv2.accumulateWeighted(
                gray,
                background,
                0.5
            )

        frame_count += 1

        cv2.putText(
            frame,
            f"Capturing background: {frame_count}/{background_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    else:

        # ---------------------------------------
        # Current ROI
        # ---------------------------------------

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(
            gray,
            (7, 7),
            0
        )

        # ---------------------------------------
        # Background difference
        # ---------------------------------------

        difference = cv2.absdiff(
            background.astype(np.uint8),
            gray
        )

        # ---------------------------------------
        # Threshold
        # ---------------------------------------

        _, threshold = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )

        # ---------------------------------------
        # Remove small noise
        # ---------------------------------------

        kernel = np.ones((3, 3), np.uint8)

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

        # ---------------------------------------
        # Display threshold
        # ---------------------------------------

        cv2.imshow(
            "Hand Mask",
            threshold
        )

        # ---------------------------------------
        # Draw ROI on original frame
        # ---------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Place your hand inside",
            (x1 - 20, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # ---------------------------------------
    # Display camera
    # ---------------------------------------

    cv2.imshow(
        "Camera",
        frame
    )

    # ---------------------------------------
    # ESC to exit
    # ---------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ---------------------------------------
# Cleanup
# ---------------------------------------

camera.release()
cv2.destroyAllWindows()