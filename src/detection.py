from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
from ultralytics import YOLO


@dataclass
class Detection:
    """Represents one YOLO object detection."""

    class_id: int
    class_name: str
    confidence: float

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        """Return the center point of the bounding box."""

        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return the bounding box."""

        return (
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        )


class YOLODetector:
    """
    YOLOv8-based object detector.

    This module is based on the YOLO inference logic
    implemented in the supplied surveillance notebook.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = 0.65,
        target_classes: Optional[List[str]] = None,
    ) -> None:

        self.model_path = Path(model_path)

        self.confidence_threshold = confidence_threshold

        self.target_classes = set(
            target_classes or []
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

        self.class_names: Dict[int, str] = dict(
            self.model.names
        )

        print(
            "Model class mapping:",
            self.class_names,
        )

    def detect(
        self,
        frame,
    ) -> List[Detection]:
        """
        Run YOLO inference on a single frame.

        Returns only detections that meet the configured
        confidence threshold.
        """

        results = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        detections: List[Detection] = []

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            class_name = self.class_names.get(
                class_id,
                str(class_id),
            )

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            x1, y1, x2, y2 = map(
                int,
                coordinates,
            )

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections

    def detect_targets(
        self,
        frame,
    ) -> List[Detection]:
        """
        Detect only configured suspicious/target classes.
        """

        detections = self.detect(frame)

        if not self.target_classes:
            return detections

        return [
            detection
            for detection in detections
            if detection.class_name in self.target_classes
        ]

    def annotate(
        self,
        frame,
        detections: List[Detection],
    ):
        """Draw detection bounding boxes on a frame."""

        annotated = frame.copy()

        for detection in detections:

            cv2.rectangle(
                annotated,
                (
                    detection.x1,
                    detection.y1,
                ),
                (
                    detection.x2,
                    detection.y2,
                ),
                (0, 255, 0),
                2,
            )

            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )

            cv2.putText(
                annotated,
                label,
                (
                    detection.x1,
                    max(detection.y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def predict_and_annotate(self, frame):
        """
        Convenience method that performs detection
        and returns the annotated frame.
        """

        detections = self.detect(frame)

        annotated = self.annotate(
            frame,
            detections,
        )

        return annotated, detections
