import cv2


# ==========================================
# SETTINGS
# ==========================================

CAMERA_INDEX = 0

# Your previously calibrated ROI
ROI_X = 37
ROI_Y = 74
ROI_WIDTH = 174
ROI_HEIGHT = 284


# ==========================================
# CAMERA
# ==========================================

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Camera opened successfully.")
print("Press ESC to exit.")


# ==========================================
# CAMERA LOOP
# ==========================================

while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read frame.")
        break

    # Mirror the camera
    frame = cv2.flip(frame, 1)


    # ======================================
    # ROI
    # ======================================

    x1 = ROI_X
    y1 = ROI_Y

    x2 = ROI_X + ROI_WIDTH
    y2 = ROI_Y + ROI_HEIGHT


    # Draw ROI
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )


    # ======================================
    # TEXT
    # ======================================

    cv2.putText(
        frame,
        "Place your hand inside ROI",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        f"ROI: {ROI_WIDTH} x {ROI_HEIGHT}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "ROI Test",
        frame
    )


    # ======================================
    # EXIT
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ==========================================
# CLEANUP
# ==========================================

camera.release()
cv2.destroyAllWindows()

print("ROI test finished.")