import cv2

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera opened.")
print()
print("Move your hand into the normal position.")
print("When ready, press SPACE to select the ROI.")
print("ESC to exit.")

# Let the camera stabilize
for _ in range(30):
    ret, frame = cap.read()
    if not ret:
        continue

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    display = frame.copy()

    cv2.putText(
        display,
        "Press SPACE to select Hand ROI",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("ROI Calibration", display)

    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # SPACE
        # Let user draw the ROI
        roi = cv2.selectROI(
            "Select Hand ROI",
            frame,
            fromCenter=False,
            showCrosshair=True
        )

        x, y, w, h = roi

        if w == 0 or h == 0:
            print("No ROI selected.")
            continue

        print()
        print("===================================")
        print("SELECTED ROI")
        print("===================================")
        print(f"x      = {x}")
        print(f"y      = {y}")
        print(f"width  = {w}")
        print(f"height = {h}")
        print()
        print("You can use these values in your collector.")
        print("===================================")

        # Show selected ROI
        selected = frame.copy()

        cv2.rectangle(
            selected,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            selected,
            "Selected ROI - Press ENTER to accept",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Selected ROI", selected)

        while True:
            k = cv2.waitKey(0) & 0xFF

            if k == 13:  # ENTER
                print("ROI accepted.")
                break

            elif k == 27:  # ESC
                print("ROI cancelled.")
                break

        cv2.destroyWindow("Select Hand ROI")
        cv2.destroyWindow("Selected ROI")

        if k == 13:
            break

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()