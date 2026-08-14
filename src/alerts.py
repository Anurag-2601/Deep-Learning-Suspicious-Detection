import time
from pathlib import Path
from typing import Optional

import cv2


class AlertManager:
    """Handles suspicious-event alerts and evidence images."""

    def __init__(
        self,
        cooldown_seconds: float = 5.0,
        evidence_directory: Optional[Path] = None,
        sound_enabled: bool = True,
    ) -> None:

        self.cooldown_seconds = (
            cooldown_seconds
        )

        self.evidence_directory = (
            Path(evidence_directory)
            if evidence_directory
            else Path("results/evidence")
        )

        self.sound_enabled = sound_enabled

        self.last_alert_time = 0.0

        self.alert_count = 0

        self.evidence_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def can_alert(self) -> bool:
        """Check whether the alert cooldown has expired."""

        return (
            time.time()
            - self.last_alert_time
            >= self.cooldown_seconds
        )

    def trigger(
        self,
        level: str,
        reason: str,
        frame=None,
    ) -> Optional[str]:
        """
        Trigger an alert.

        Returns the saved evidence-image path when
        an image is provided.
        """

        if not self.can_alert():
            return None

        now = time.time()

        self.last_alert_time = now

        self.alert_count += 1

        print(
            f"[ALERT] {level}: {reason}"
        )

        image_path = None

        if frame is not None:

            image_name = (
                f"alert_{int(now * 1000)}.jpg"
            )

            image_path = (
                self.evidence_directory
                / image_name
            )

            cv2.imwrite(
                str(image_path),
                frame,
            )

        self._sound_alert()

        return (
            str(image_path)
            if image_path is not None
            else None
        )

    def _sound_alert(self) -> None:
        """Generate a platform-appropriate alert sound."""

        if not self.sound_enabled:
            return

        try:

            import winsound

            winsound.Beep(
                1000,
                800,
            )

        except (
            ImportError,
            RuntimeError,
        ):

            print(
                "\a",
                end="",
                flush=True,
            )
