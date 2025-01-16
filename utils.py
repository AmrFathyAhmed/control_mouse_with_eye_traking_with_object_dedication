import cv2
import numpy as np
import mediapipe as mp
def draw_boxes(canvas, box_positions, box_colors, box_width, box_height):
    """Draw boxes on the canvas with their respective colors."""
    for i, (x, y) in enumerate(box_positions):
        color = box_colors[i]  # Use the color from the box_colors list
        cv2.rectangle(canvas, (x, y), (x + box_width, y + box_height), color, -1)
        cv2.putText(canvas, f"Box {i + 1}", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def get_eye_landmarks(landmarks, width, height):
    """Get the coordinates of the left and right eyes."""
    left_eye = [landmarks[i] for i in [362, 385, 386, 387, 388, 263, 466, 249]]
    right_eye = [landmarks[i] for i in [33, 160, 161, 159, 158, 133, 173, 7]]
    left_coords = [(int(p.x * width), int(p.y * height)) for p in left_eye]
    right_coords = [(int(p.x * width), int(p.y * height)) for p in right_eye]
    return left_coords, right_coords

def draw_eye_markers(frame, eye_coords):
    """Draw refined markers on the iris area."""
    for i, (x, y) in enumerate(eye_coords):
        color = (0, 255, 0) if i in [1, 2, 5, 6] else (255, 0, 0)  # Highlight iris points in green
        cv2.circle(frame, (x, y), 3, color, -1)  # Green dots for iris points

def get_iris_landmarks(eye_coords):
    """
    Focus only on the central part of the eye (iris area) for higher accuracy.
    Use the middle points of the landmarks to approximate gaze.
    """
    iris_coords = [eye_coords[1], eye_coords[2], eye_coords[5], eye_coords[6]]
    avg_x = sum(x for x, y in iris_coords) / len(iris_coords)
    avg_y = sum(y for x, y in iris_coords) / len(iris_coords)
    return avg_x, avg_y
def calculate_ear(eye_coords):
    A = np.linalg.norm(np.array(eye_coords[1]) - np.array(eye_coords[5]))  # Vertical distance
    B = np.linalg.norm(np.array(eye_coords[2]) - np.array(eye_coords[4]))  # Vertical distance
    C = np.linalg.norm(np.array(eye_coords[0]) - np.array(eye_coords[3]))  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear