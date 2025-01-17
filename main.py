import cv2
import numpy as np
from eye_tracker import EyeTracker
import pyautogui
from object_detector import ObjectDetector

# Prevent mouse from going off screen
pyautogui.FAILSAFE = True

# Initialize webcam
webcam_cap = cv2.VideoCapture(0)

# Initialize EyeTracker
eye_tracker = EyeTracker()

# Initialize phone camera feed (replace with your phone's IP and port)
phone_ip = "192.168.1.3"  # Replace with your phone's IP address
phone_port = "8080"  # Replace with your phone's port
phone_url = f"http://{phone_ip}:{phone_port}/video"
print(f"Connecting to phone camera at: {phone_url}")

# Initialize phone camera feed
phone_cap = cv2.VideoCapture(phone_url)
if not phone_cap.isOpened():
    print(f"Failed to connect to phone camera at: {phone_url}")
    print("Please check the IP address, port, and network connection.")
    exit()

# Initialize Object Detector (ensure the model path is correct)
object_detector = ObjectDetector("yolov8n.pt")

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

def main():
    while True:
        # Read frames from webcam and phone camera
        ret_webcam, webcam_frame = webcam_cap.read()
        ret_phone, phone_frame = phone_cap.read()

        if not ret_webcam:
            print("Failed to capture frame from webcam.")  # Debug
            break
        if not ret_phone:
            print("Failed to capture frame from phone camera.")  # Debug
            break

        # Flip the webcam frame for a mirror-like experience
        webcam_frame = cv2.flip(webcam_frame, 1)

        # Resize phone frame to screen dimensions
        phone_frame = cv2.resize(phone_frame, (screen_width, screen_height))

        # Perform object detection on the phone frame
        annotated_phone_frame = object_detector.detect(phone_frame)

        # Resize webcam frame for the top-left corner overlay 
        webcam_width = 320
        webcam_height = 240
        small_webcam_frame = cv2.resize(webcam_frame, (webcam_width, webcam_height))

        # Overlay the webcam feed in the top-left corner of the annotated phone frame
        annotated_phone_frame[0:webcam_height, 0:webcam_width] = small_webcam_frame

        # Process the webcam frame with EyeTracker
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

        # Display the annotated phone frame with webcam overlay
        cv2.imshow("Phone Camera with YOLOv8 Detections and Webcam Overlay", annotated_phone_frame)

        # Wait for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') and eye_tracker.calibrated_center is None:
            # Calibrate center based on current gaze
            rgb_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB)
            results = eye_tracker.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                eye_tracker.calibrate_blink_threshold(landmarks, webcam_frame.shape[1], webcam_frame.shape[0])

    # Release resources
    webcam_cap.release()
    phone_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()