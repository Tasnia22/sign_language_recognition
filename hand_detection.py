import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Camera opened!")
print("Place your hand inside the rectangle.")
print("Press ESC to exit.")

while True:
    success, frame = camera.read()

    if not success:
        print("❌ Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    height, width, _ = frame.shape

    # ROI coordinates
    x1 = int(width * 0.55)
    y1 = int(height * 0.20)

    x2 = int(width * 0.95)
    y2 = int(height * 0.80)

    # Draw ROI rectangle
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Add instruction
    cv2.putText(
        frame,
        "Place your hand inside",
        (x1 - 20, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Detection - ROI", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()