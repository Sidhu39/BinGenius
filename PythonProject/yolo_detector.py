import cv2
import numpy as np
from ultralytics import YOLO


class WasteDetector:
    def __init__(self, model_path="best.pt"):
        """Initialize YOLO model for waste detection"""
        print(f"Loading YOLO model: {model_path}...")
        self.model = YOLO(model_path)
        print("✓ Model loaded successfully")

    def detect(self, frame):
        """
        Run detection on a single frame

        Args:
            frame: OpenCV image (numpy array)

        Returns:
            List of detections with format:
            [
                {
                    'name': 'bottle',
                    'confidence': 0.92,
                    'bbox': [x1, y1, x2, y2]
                },
                ...
            ]
        """
        if frame is None:
            return []

        try:
            detections = []
            results = self.model(frame, stream=False, verbose=False)

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]

                    detections.append({
                        'name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id
                    })

            return detections
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def detect_and_draw(self, frame):
        """
        Detect objects and draw bounding boxes on frame

        Args:
            frame: OpenCV image

        Returns:
            Annotated frame, list of detections
        """
        detections = self.detect(frame)
        annotated_frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            name = det['name']
            conf = det['confidence']

            # Choose color based on confidence
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{name} {conf:.2%}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame, detections
