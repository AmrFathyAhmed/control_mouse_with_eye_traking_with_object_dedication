import numpy as np
import pyautogui
import mediapipe as mp
import time
from utils import get_eye_landmarks, get_iris_landmarks, calculate_ear
from constants import *
import cv2
class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.calibrated_center = None
        self.recent_gaze_points = []
        self.smoothing_factor = 15  # Number of points for smoothing
        self.blink_start_time = None
        self.blink_count = 0
        self.blink_timeout = 0.8  # Time (in seconds) for detecting double blinks
        self.is_blinking = False
        self.blink_threshold = None  # Will be set dynamically during calibration
        self.calibration_state = "WAITING"  # Calibration state: WAITING, OPEN_EYES, CLOSED_EYES
        self.ear_open = None  # Store EAR when eyes are open
        self.ear_closed = None  # Store EAR when eyes are closed

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Blink detection (only if blink_threshold is set)
            if self.blink_threshold is not None:
                self.detect_blink(landmarks, frame.shape[1], frame.shape[0])

            # Get eye landmarks
            left_eye, right_eye = get_eye_landmarks(landmarks, frame.shape[1], frame.shape[0])

            # Calculate the refined pupil position for each eye
            iris_left_x, iris_left_y = get_iris_landmarks(left_eye)
            iris_right_x, iris_right_y = get_iris_landmarks(right_eye)

            avg_pupil_x = (iris_left_x + iris_right_x) / 2
            avg_pupil_y = (iris_left_y + iris_right_y) / 2

            # Store gaze points for smoothing
            self.recent_gaze_points.append((avg_pupil_x, avg_pupil_y))
            if len(self.recent_gaze_points) > self.smoothing_factor:
                self.recent_gaze_points.pop(0)

            # Calculate smoothed gaze point
            smoothed_x = np.mean([point[0] for point in self.recent_gaze_points])
            smoothed_y = np.mean([point[1] for point in self.recent_gaze_points])

            # Map offsets to screen coordinates after calibration
            if self.calibrated_center is not None:
                offset_x = smoothed_x - self.calibrated_center[0]
                offset_y = smoothed_y - self.calibrated_center[1]

                # Constrain offsets to a reasonable range
                offset_x = max(-100, min(offset_x, 100))
                offset_y = max(-100, min(offset_y, 100))

                # Map to screen coordinates, avoiding corners
                x = np.interp(offset_x, [-100, 100], [100, screen_width - 100])
                y = np.interp(offset_y, [-100, 100], [100, screen_height - 100])

                # Ensure the mouse stays within screen bounds
                x = max(100, min(x, screen_width - 100))  # Keep mouse 100 pixels away from edges
                y = max(100, min(y, screen_height - 100))  # Keep mouse 100 pixels away from edges

                # Move the mouse
                pyautogui.moveTo(x, y, duration=0.1)

            # Draw a marker at the gaze position on the webcam frame
            gaze_x = int(smoothed_x)
            gaze_y = int(smoothed_y)
            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 255, 0), -1)  # Green circle at gaze position

    def calibrate_blink_threshold(self, landmarks, width, height):
        """
        Calibrate the blink threshold based on the user's eye aspect ratio (EAR) when eyes are open and closed.
        """
        if self.calibration_state == "WAITING":
            # Start calibration
            self.calibration_state = "OPEN_EYES"
            print("Calibration started. Please keep your eyes open.")

        elif self.calibration_state == "OPEN_EYES":
            # Calculate EAR when eyes are open
            right_eye_landmarks = [33, 160, 158, 133, 153, 144]
            right_eye_coords = [(landmarks[i].x * width, landmarks[i].y * height) for i in right_eye_landmarks]

            if len(right_eye_coords) == 6:
                self.ear_open = calculate_ear(right_eye_coords)  # EAR when eyes are open
                print(f"EAR (Open): {self.ear_open}")

                # Move to the next calibration state
                self.calibration_state = "CLOSED_EYES"
                print("Please close your eyes.")

        elif self.calibration_state == "CLOSED_EYES":
            # Calculate EAR when eyes are closed
            right_eye_landmarks = [33, 160, 158, 133, 153, 144]
            right_eye_coords = [(landmarks[i].x * width, landmarks[i].y * height) for i in right_eye_landmarks]

            if len(right_eye_coords) == 6:
                self.ear_closed = calculate_ear(right_eye_coords)
                print(f"EAR (Closed): {self.ear_closed}")

                # Set the blink threshold to a value between open and closed EAR
                self.blink_threshold = (self.ear_open + self.ear_closed) / 2
                print(f"Calibrated blink threshold: {self.blink_threshold}")

                # Calibrate gaze center
                left_eye, right_eye = get_eye_landmarks(landmarks, width, height)
                iris_left_x, iris_left_y = get_iris_landmarks(left_eye)
                iris_right_x, iris_right_y = get_iris_landmarks(right_eye)

                avg_pupil_x = (iris_left_x + iris_right_x) / 2
                avg_pupil_y = (iris_left_y + iris_right_y) / 2

                self.calibrated_center = (avg_pupil_x, avg_pupil_y)
                print(f"Calibrated gaze center: {self.calibrated_center}")

                # Reset calibration state
                self.calibration_state = "WAITING"

    def detect_blink(self, landmarks, width, height):
        """
        Detects blink and double blink using EAR method.
        """
        right_eye_landmarks = [33, 160, 158, 133, 153, 144]
        right_eye_coords = [(landmarks[i].x * width, landmarks[i].y * height) for i in right_eye_landmarks]

        if len(right_eye_coords) == 6:
            ear = calculate_ear(right_eye_coords)
            current_time = time.time()

            # Detect blink (EAR-based)
            if ear < self.blink_threshold:  # Eye is closed (blink detected)
                if self.blink_start_time is None:
                    self.blink_start_time = current_time  # Start tracking the blink time
                    print("Blink detected!")
            else:  # Eye is open
                if self.blink_start_time:
                    blink_duration = current_time - self.blink_start_time
                    if 0.08 <= blink_duration <= 0.35:  # Valid blink duration range
                        if self.blink_count == 0:  # First blink
                            self.blink_count = 1
                            self.last_action_time = current_time
                            print(f"First blink detected! Count: {self.blink_count}, Time: {current_time}")
                        elif self.blink_count == 1 and (current_time - self.last_action_time) <= self.blink_timeout:  # Second blink within timeout
                            print(f"Double blink detected! Time: {current_time}")
                            self.blink_count = 0  # Reset blink count after detecting double blink
                            self.last_action_time = current_time

                            # Detect if pointer lands in a box (interaction logic)
                            screen_x, screen_y = pyautogui.position()
                            print(f"Mouse Position: ({screen_x}, {screen_y})")
                    else:
                        self.blink_start_time = None  # Reset for invalid blink duration

            # Reset blink count if too much time passes (e.g., 1.5 seconds of inactivity)
            if self.blink_count > 0 and (time.time() - self.last_action_time) > 1.5:
                self.blink_count = 0
                print("Blink count reset due to inactivity.")