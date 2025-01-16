import cv2
import mediapipe as mp
import numpy as np
import time
from constants import *
import pyautogui
from utils import calculate_ear, get_eye_landmarks, draw_eye_markers, get_iris_landmarks

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.calibrated_center = None
        self.recent_gaze_points = []
        self.smoothing_factor = 15
        self.blink_start_time = None
        self.blink_count = 0
        self.blink_timeout = 0.8
        self.is_blinking = False
        self.blink_threshold = None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Calibrate blink threshold if not already set
            if self.blink_threshold is None:
                self.calibrate_blink_threshold(landmarks, frame.shape[1], frame.shape[0])

            # Blink detection
            self.detect_blink(landmarks, frame.shape[1], frame.shape[0])

            # Get eye landmarks
            left_eye, right_eye = get_eye_landmarks(landmarks, frame.shape[1], frame.shape[0])

            # Draw markers on the eyes
            draw_eye_markers(frame, left_eye)
            draw_eye_markers(frame, right_eye)

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

                x = np.interp(offset_x, [-50, 50], [0, screen_width])
                y = np.interp(offset_y, [-50, 50], [0, screen_height])

                # Move the mouse
                pyautogui.moveTo(x, y, duration=0.1)

    def calibrate_blink_threshold(self, landmarks=None, width=None, height=None):
        if landmarks is not None and width is not None and height is not None:
            right_eye_landmarks = [33, 160, 158, 133, 153, 144]
            right_eye_coords = [(landmarks[i].x * width, landmarks[i].y * height) for i in right_eye_landmarks]

            if len(right_eye_coords) == 6:
                ear_open = calculate_ear(right_eye_coords)  # EAR when eyes are open
                print(f"EAR (Open): {ear_open}")

                # Ask the user to blink and calculate EAR when eyes are closed
                input("Please close your eyes and press Enter to continue...")
                ear_closed = calculate_ear(right_eye_coords)
                print(f"EAR (Closed): {ear_closed}")

                # Set the blink threshold to a value between open and closed EAR
                self.blink_threshold = (ear_open + ear_closed) / 2
                print(f"Calibrated blink threshold: {self.blink_threshold}")

    def detect_blink(self, landmarks, width, height):
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