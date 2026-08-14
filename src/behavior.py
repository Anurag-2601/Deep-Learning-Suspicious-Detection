from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class BehaviorResult:
    """Result produced by the behavior-analysis module."""

    label: str
    confidence: float
    is_suspicious: bool
    backend: str = "unknown"


class BehaviorAnalyzer:
    """
    Base interface for temporal behavior analysis.

    A behavior model must receive a sequence of frames,
    rather than a single image.
    """

    def analyze(
        self,
        frames,
    ) -> BehaviorResult:
        raise NotImplementedError


class SlowFastBehaviorAnalyzer(BehaviorAnalyzer):
    """
    SlowFast behavior-analysis interface.

    The project currently does not contain a trained
    SlowFast checkpoint, so this class deliberately does
    not pretend to perform real SlowFast inference.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
    ) -> None:

        self.model_path = model_path

        if model_path is None:
            self.available = False
        else:
            self.available = False

            # Model loading will be implemented once
            # the project SlowFast checkpoint and its
            # preprocessing specification are available.

    def analyze(
        self,
        frames,
    ) -> BehaviorResult:

        if not self.available:
            return BehaviorResult(
                label="unknown",
                confidence=0.0,
                is_suspicious=False,
                backend="slowfast-unavailable",
            )

        raise NotImplementedError(
            "SlowFast inference has not yet been "
            "implemented for the project checkpoint."
        )


class MotionHeuristicAnalyzer(BehaviorAnalyzer):
    """
    Temporary development fallback.

    This is NOT the final SlowFast model.

    It only estimates whether significant motion is
    present in the frame sequence. It should be replaced
    by the trained behavior model before final evaluation.
    """

    def __init__(
        self,
        frame_count: int = 16,
        motion_threshold: float = 12.0,
    ) -> None:

        self.frame_count = frame_count

        self.motion_threshold = (
            motion_threshold
        )

    def analyze(
        self,
        frames,
    ) -> BehaviorResult:

        if len(frames) < 2:
            return BehaviorResult(
                label="insufficient_frames",
                confidence=0.0,
                is_suspicious=False,
                backend="motion-heuristic",
            )

        motion_scores = []

        previous = None

        for frame in frames:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY,
            )

            gray = cv2.resize(
                gray,
                (320, 240),
            )

            if previous is not None:

                difference = cv2.absdiff(
                    previous,
                    gray,
                )

                motion_score = float(
                    np.mean(difference)
                )

                motion_scores.append(
                    motion_score
                )

            previous = gray

        if not motion_scores:
            return BehaviorResult(
                label="unknown",
                confidence=0.0,
                is_suspicious=False,
                backend="motion-heuristic",
            )

        average_motion = float(
            np.mean(motion_scores)
        )

        normalized_confidence = min(
            average_motion / max(
                self.motion_threshold,
                1e-6,
            ),
            1.0,
        )

        if average_motion >= self.motion_threshold:

            return BehaviorResult(
                label="high_motion",
                confidence=normalized_confidence,
                is_suspicious=True,
                backend="motion-heuristic",
            )

        return BehaviorResult(
            label="normal_motion",
            confidence=1.0 - normalized_confidence,
            is_suspicious=False,
            backend="motion-heuristic",
        )


class TemporalFrameBuffer:
    """Maintains a rolling sequence of video frames."""

    def __init__(
        self,
        max_length: int = 16,
    ) -> None:

        self.frames = deque(
            maxlen=max_length
        )

    def add(self, frame) -> None:
        """Add a frame to the temporal buffer."""

        self.frames.append(
            frame.copy()
        )

    def ready(self) -> bool:
        """Return True when enough frames are available."""

        return (
            len(self.frames)
            == self.frames.maxlen
        )

    def get(self):
        """Return the current frame sequence."""

        return list(self.frames)

    def clear(self) -> None:
        """Clear the temporal buffer."""

        self.frames.clear()
