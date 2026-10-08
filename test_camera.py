# import cv2

# # Open the default camera
# camera = cv2.VideoCapture(0)

# if not camera.isOpened():
#     print("❌ Could not open camera.")
#     exit()

# print("✅ Camera opened successfully!")
# print("Press ESC to exit.")

# while True:
#     success, frame = camera.read()

#     if not success:
#         print("❌ Could not read frame.")
#         break

#     # Mirror the camera image
#     frame = cv2.flip(frame, 1)

#     # Display the frame
#     cv2.imshow("Camera Test", frame)

#     # Wait for a key press
#     key = cv2.waitKey(1) & 0xFF

#     # ESC key
#     if key == 27:
#         break

# camera.release()
# cv2.destroyAllWindows()

import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Could not open camera.")
    exit()

print("✅ Camera opened successfully.")

while True:

    success, frame = camera.read()

    if not success:
        print("❌ Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

camera.release()
cv2.destroyAllWindows()

print("Camera test finished.")