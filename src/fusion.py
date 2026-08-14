from dataclasses import dataclass
from typing import List

from .behavior import BehaviorResult
from .detection import Detection


@dataclass
class FusionDecision:
    """Final decision produced by the fusion engine."""

    level: str
    suspicious: bool
    reason: str

    objects: List[str]
    behavior: str
    behavior_confidence: float


class FusionEngine:
    """
    Combines object detections and behavior analysis.

    Decision rules are based on the project's proposed
    suspicious-activity logic.
    """

    WEAPON_CLASSES = {
        "gun",
        "knife",
        "crowbar",
        "scissor",
    }

    def __init__(
        self,
        running_persistence_frames: int = 10,
    ) -> None:

        self.running_persistence_frames = (
            running_persistence_frames
        )

        self.running_counter = 0

    def evaluate(
        self,
        detections: List[Detection],
        behavior: BehaviorResult,
    ) -> FusionDecision:

        object_names = [
            detection.class_name
            for detection in detections
        ]

        object_set = set(object_names)

        has_person = (
            "person" in object_set
        )

        detected_weapons = (
            object_set
            & self.WEAPON_CLASSES
        )

        # --------------------------------------------------
        # Rule 1:
        # Person + weapon = HIGH RISK
        # --------------------------------------------------

        if has_person and detected_weapons:

            weapon_names = ", ".join(
                sorted(detected_weapons)
            )

            return FusionDecision(
                level="HIGH",
                suspicious=True,
                reason=(
                    f"Person with weapon detected: "
                    f"{weapon_names}"
                ),
                objects=object_names,
                behavior=behavior.label,
                behavior_confidence=behavior.confidence,
            )

        # --------------------------------------------------
        # Rule 2:
        # Person + suspicious behavior
        # --------------------------------------------------

        if (
            has_person
            and behavior.is_suspicious
        ):

            return FusionDecision(
                level="HIGH",
                suspicious=True,
                reason=(
                    f"Suspicious behavior detected: "
                    f"{behavior.label}"
                ),
                objects=object_names,
                behavior=behavior.label,
                behavior_confidence=behavior.confidence,
            )

        # --------------------------------------------------
        # Rule 3:
        # Running persistence
        # --------------------------------------------------

        if (
            has_person
            and behavior.label == "running"
        ):

            self.running_counter += 1

            if (
                self.running_counter
                >= self.running_persistence_frames
            ):

                return FusionDecision(
                    level="MEDIUM",
                    suspicious=True,
                    reason=(
                        "Persistent running behavior"
                    ),
                    objects=object_names,
                    behavior=behavior.label,
                    behavior_confidence=behavior.confidence,
                )

        else:

            self.running_counter = 0

        # --------------------------------------------------
        # Rule 4:
        # Weapon without person
        # --------------------------------------------------

        if detected_weapons:

            weapon_names = ", ".join(
                sorted(detected_weapons)
            )

            return FusionDecision(
                level="MEDIUM",
                suspicious=True,
                reason=(
                    f"Weapon/object detected: "
                    f"{weapon_names}"
                ),
                objects=object_names,
                behavior=behavior.label,
                behavior_confidence=behavior.confidence,
            )

        # --------------------------------------------------
        # Rule 5:
        # Normal
        # --------------------------------------------------

        return FusionDecision(
            level="NORMAL",
            suspicious=False,
            reason="No suspicious activity detected",
            objects=object_names,
            behavior=behavior.label,
            behavior_confidence=behavior.confidence,
        )
