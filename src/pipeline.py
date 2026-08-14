import time
from typing import Optional

import cv2

from .alerts import AlertManager
from .behavior import (
    BehaviorAnalyzer,
    TemporalFrameBuffer,
)
from .detection import YOLODetector
from .fusion import FusionEngine
from .logger import EventLogger


class SurveillancePipeline:
    """
    End-to-end surveillance processing pipeline.

    Video
        ↓
    YOLOv8
        ↓
    Behavior Analysis
        ↓
    Feature Fusion / Decision
        ↓
    Alert + Logging
    """

    def __init__(
        self,
        detector: YOLODetector,
        behavior_analyzer: BehaviorAnalyzer,
        fusion_engine: FusionEngine,
        alert_manager: AlertManager,
        logger: EventLogger,
        frame_buffer_size: int = 16,
    ) -> None:

        self.detector = detector

        self.behavior_analyzer = (
            behavior_analyzer
        )

        self.fusion_engine = fusion_engine

        self.alert_manager = (
            alert_manager
        )

        self.logger = logger

        self.frame_buffer = (
            TemporalFrameBuffer(
                max_length=frame_buffer_size
            )
        )

        self.total_frames = 0

        self.start_time = time.time()

    def process_frame(self, frame):
        """
        Process one video frame.

        Returns:
            annotated_frame,
            decision
        """

        self.total_frames += 1

        self.frame_buffer.add(
            frame
        )

        detections = (
            self.detector.detect(frame)
        )

        annotated = (
            self.detector.annotate(
                frame,
                detections,
            )
        )

        if self.frame_buffer.ready():

            behavior = (
                self.behavior_analyzer.analyze(
                    self.frame_buffer.get()
                )
            )

        else:

            behavior = self.behavior_analyzer.analyze(
                self.frame_buffer.get()
            )

        decision = (
            self.fusion_engine.evaluate(
                detections,
                behavior,
            )
        )

        if decision.suspicious:

            evidence_file = (
                self.alert_manager.trigger(
                    level=decision.level,
                    reason=decision.reason,
                    frame=annotated,
                )
            )

            if evidence_file is not None:

                self.logger.log(
                    decision,
                    evidence_file,
                )

        self._draw_status(
            annotated,
            decision,
        )

        return annotated, decision

    @staticmethod
    def _draw_status(
        frame,
        decision,
    ) -> None:

        text = (
            f"{decision.level}: "
            f"{decision.reason}"
        )

        cv2.putText(
            frame,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255)
            if decision.suspicious
            else (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def run(
        self,
        source=0,
        window_name=(
            "Deep Learning Anomaly Detection "
            "- Surveillance System"
        ),
    ) -> None:

        capture = cv2.VideoCapture(
            source
        )

        if not capture.isOpened():

            raise RuntimeError(
                f"Unable to open video source: "
                f"{source}"
            )

        previous_time = time.time()

        try:

            while True:

                success, frame = (
                    capture.read()
                )

                if not success:
                    break

                now = time.time()

                fps = 1.0 / max(
                    now - previous_time,
                    1e-9,
                )

                previous_time = now

                annotated, decision = (
                    self.process_frame(frame)
                )

                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    window_name,
                    annotated,
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):
                    break

        finally:

            capture.release()

            cv2.destroyAllWindows()
