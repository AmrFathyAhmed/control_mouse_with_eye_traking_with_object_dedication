# Screen dimensions
screen_width = 1920
screen_height = 1080

# Webcam dimensions
webcam_width = int(screen_width * 0.2)
webcam_height = int(screen_height * 0.2)

# Box dimensions and positions
box_width = int(screen_width * 0.15)
box_height = int(screen_height * 0.1)
box_positions = [
    (screen_width // 2 - box_width // 2, screen_height // 8),
    (screen_width // 8, screen_height // 2 - box_height // 2),
    (screen_width - screen_width // 8 - box_width, screen_height // 2 - box_height // 2),
    (screen_width // 2 - box_width // 2, screen_height - screen_height // 4),
    (screen_width // 2 - box_width // 2, screen_height // 2 - box_height // 2),
]

# Blink detection thresholds
blink_threshold = 0.2
right_eye_landmarks = [362, 385, 387, 263, 373, 380]
