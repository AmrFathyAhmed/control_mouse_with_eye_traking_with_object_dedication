import cv2
import pyautogui
from eye_tracker import EyeTracker
from object_detector import ObjectDetector
from constants import *

# Prevent mouse from going off screen
pyautogui.FAILSAFE = True

# Initialize webcam
webcam_cap = cv2.VideoCapture(0)

# Initialize phone camera feed (replace with your phone's IP and port)
phone_ip = "192.168.1.3"  # Replace with your phone's IP address
phone_port = "8080"  # Replace with your phone's port
phone_url = f"http://{phone_ip}:{phone_port}/video"  # URL for the phone camera feed

# Print the phone URL for debugging
print(f"Connecting to phone camera at: {phone_url}")

# Initialize phone camera feed
phone_cap = cv2.VideoCapture(phone_url)

# Check if the phone camera feed is opened successfully
if not phone_cap.isOpened():
    print(f"Failed to connect to phone camera at: {phone_url}")
    print("Please check the IP address, port, and network connection.")
    exit()

# Initialize EyeTracker and ObjectDetector
eye_tracker = EyeTracker()
object_detector = ObjectDetector("yolov8n.pt")

# Main loop
while True:
    # Capture frame from phone camera
    ret_phone, phone_frame = phone_cap.read()
    if not ret_phone:
        print("Failed to capture frame from phone camera.")  # Debug
        break

    # Capture frame from webcam
    ret_webcam, webcam_frame = webcam_cap.read()
    if not ret_webcam:
        print("Failed to capture frame from webcam.")  # Debug
        break

    # Flip the webcam frame for a mirror-like experience
    webcam_frame = cv2.flip(webcam_frame, 1)

    # Resize phone frame to match screen dimensions
    phone_frame = cv2.resize(phone_frame, (screen_width, screen_height))

    # Resize webcam frame to a smaller size (e.g., 20% of screen width and height)
    webcam_width = int(screen_width* 0.2)  # 20% of screen width
    webcam_height = int(screen_height * 0.2)  # 20% of screen height
    webcam_frame = cv2.resize(webcam_frame, (webcam_width, webcam_height))

    # Run YOLOv8 inference ONLY on the phone frame (not the webcam frame)
    annotated_phone_frame = object_detector.detect(phone_frame)

    # Overlay the webcam feed in the top-left corner of the annotated phone frame
    annotated_phone_frame[0:webcam_height, 0:webcam_width] = webcam_frame

    # Process the webcam frame with MediaPipe (for eye-tracking)
    eye_tracker.process_frame(webcam_frame)

    # Display instructions for calibration
    if eye_tracker.calibrated_center is None:
        cv2.putText(
            annotated_phone_frame, "Look at the center and press 'C' to calibrate",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    else:
        cv2.putText(
            annotated_phone_frame, "Gaze tracking active. Press 'Q' to quit.",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

    # Display the annotated frame
    cv2.imshow("Phone Camera with YOLOv8 Detections and Webcam Overlay", annotated_phone_frame)

    # Wait for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c') and eye_tracker.calibrated_center is None:
        # Calibrate center based on current gaze
        eye_tracker.calibrate_blink_threshold()

# Release resources
webcam_cap.release()
phone_cap.release()
cv2.destroyAllWindows()