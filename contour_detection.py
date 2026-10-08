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
background_frames = 60
frame_count = 0

print()
print("Keep the ROI empty.")
print(f"Capturing background for {background_frames} frames...")


while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape


    # ---------------------------------------
    # ROI
    # ---------------------------------------

    x1 = int(width * 0.55)
    y1 = int(height * 0.20)

    x2 = int(width * 0.95)
    y2 = int(height * 0.80)

    roi = frame[y1:y2, x1:x2]


    # ---------------------------------------
    # Background capture
    # ---------------------------------------

    if frame_count < background_frames:

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

        cv2.putText(
            frame,
            f"Capturing background: "
            f"{frame_count}/{background_frames}",
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

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

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
        # Noise removal
        # ---------------------------------------

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


        # ---------------------------------------
        # Find contours
        # ---------------------------------------

        contours, hierarchy = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )


        # ---------------------------------------
        # Find largest contour
        # ---------------------------------------

        if len(contours) > 0:

            largest_contour = max(
                contours,
                key=cv2.contourArea
            )

            area = cv2.contourArea(
                largest_contour
            )


            # ---------------------------------------
            # Ignore tiny contours
            # ---------------------------------------

            if area > 1000:

                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(
                    largest_contour
                )

                # Draw bounding box
                cv2.rectangle(
                    roi,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # Draw contour
                cv2.drawContours(
                    roi,
                    [largest_contour],
                    -1,
                    (0, 0, 255),
                    2
                )

                # Display area
                cv2.putText(
                    roi,
                    f"Area: {int(area)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )


        # ---------------------------------------
        # Display mask
        # ---------------------------------------

        cv2.imshow(
            "Hand Mask",
            threshold
        )


        # ---------------------------------------
        # Draw ROI
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
        "Contour Detection",
        frame
    )


    # ---------------------------------------
    # ESC
    # ---------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ---------------------------------------
# Cleanup
# ---------------------------------------

camera.release()
cv2.destroyAllWindows()