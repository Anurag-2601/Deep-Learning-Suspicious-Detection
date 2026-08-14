from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"
EVIDENCE_DIR = RESULTS_DIR / "evidence"

DEFAULT_MODEL_PATH = WEIGHTS_DIR / "best.pt"
DEFAULT_LOG_PATH = RESULTS_DIR / "detection_log.csv"


@dataclass
class DetectionConfig:
    """Configuration for YOLOv8 object detection."""

    model_path: Path = DEFAULT_MODEL_PATH

    confidence_threshold: float = 0.65

    target_classes: List[str] = field(
        default_factory=lambda: [
            "gun",
            "knife",
            "crowbar",
            "scissor",
        ]
    )


@dataclass
class AlertConfig:
    """Configuration for alert generation."""

    cooldown_seconds: float = 5.0

    save_evidence: bool = True

    evidence_directory: Path = EVIDENCE_DIR

    sound_enabled: bool = True

    popup_enabled: bool = True


@dataclass
class LoggingConfig:
    """Configuration for event logging."""

    log_file: Path = DEFAULT_LOG_PATH


@dataclass
class ApplicationConfig:
    """Complete surveillance application configuration."""

    detection: DetectionConfig = field(
        default_factory=DetectionConfig
    )

    alert: AlertConfig = field(
        default_factory=AlertConfig
    )

    logging: LoggingConfig = field(
        default_factory=LoggingConfig
    )

    camera_source: int = 0

    display_window_name: str = (
        "Deep Learning Anomaly Detection - Surveillance System"
    )


def create_directories(config: ApplicationConfig) -> None:
    """Create directories required by the application."""

    config.detection.model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.alert.evidence_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    config.logging.log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
