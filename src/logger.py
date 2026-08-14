import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from .fusion import FusionDecision


class EventLogger:
    """CSV logger for integrated surveillance events."""

    FIELDNAMES = [
        "Timestamp",
        "Decision",
        "Risk_Level",
        "Reason",
        "Detected_Objects",
        "Behavior",
        "Behavior_Confidence",
        "Image_File",
    ]

    def __init__(
        self,
        log_file: Path,
    ) -> None:

        self.log_file = Path(
            log_file
        )

        self.log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_file()

    def _initialize_file(self) -> None:

        if self.log_file.exists():
            return

        with self.log_file.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )

            writer.writeheader()

    def log(
        self,
        decision: FusionDecision,
        image_file: Optional[str] = None,
    ) -> None:

        with self.log_file.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.FIELDNAMES,
            )

            writer.writerow(
                {
                    "Timestamp": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "Decision": (
                        "SUSPICIOUS"
                        if decision.suspicious
                        else "NORMAL"
                    ),
                    "Risk_Level": decision.level,
                    "Reason": decision.reason,
                    "Detected_Objects": ", ".join(
                        decision.objects
                    ),
                    "Behavior": decision.behavior,
                    "Behavior_Confidence": (
                        f"{decision.behavior_confidence:.4f}"
                    ),
                    "Image_File": image_file or "",
                }
            )
