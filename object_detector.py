from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)
        annotated_frame = results[0].plot()  # Get the annotated frame with bounding boxes and labels
        return annotated_frame
    