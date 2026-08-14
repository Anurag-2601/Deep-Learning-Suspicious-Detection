import argparse

from .alerts import AlertManager
from .behavior import (
    MotionHeuristicAnalyzer,
    SlowFastBehaviorAnalyzer,
)
from .config import (
    ApplicationConfig,
    create_directories,
)
from .detection import YOLODetector
from .fusion import FusionEngine
from .logger import EventLogger
from .pipeline import SurveillancePipeline


def parse_source(value: str):
    """
    Convert a numeric camera source to int.

    Example:
        "0" -> 0

    File paths remain strings.
    """

    try:
        return int(value)
    except ValueError:
        return value


def build_pipeline(
    config: ApplicationConfig,
    use_motion_fallback: bool = True,
) -> SurveillancePipeline:

    detector = YOLODetector(
        model_path=(
            config.detection.model_path
        ),
        confidence_threshold=(
            config.detection.confidence_threshold
        ),
        target_classes=(
            config.detection.target_classes
        ),
    )

    slowfast = (
        SlowFastBehaviorAnalyzer()
    )

    if use_motion_fallback:

        behavior_analyzer = (
            MotionHeuristicAnalyzer()
        )

        print(
            "[INFO] Using development "
            "motion heuristic."
        )

        print(
            "[INFO] Replace this with the "
            "trained SlowFast model before "
            "final evaluation."
        )

    else:

        behavior_analyzer = slowfast

    fusion_engine = FusionEngine()

    alert_manager = AlertManager(
        cooldown_seconds=(
            config.alert.cooldown_seconds
        ),
        evidence_directory=(
            config.alert.evidence_directory
        ),
        sound_enabled=(
            config.alert.sound_enabled
        ),
    )

    logger = EventLogger(
        config.logging.log_file
    )

    return SurveillancePipeline(
        detector=detector,
        behavior_analyzer=behavior_analyzer,
        fusion_engine=fusion_engine,
        alert_manager=alert_manager,
        logger=logger,
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Deep Learning Based Suspicious "
            "Detection in Surveillance System"
        )
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Camera index or video file path. "
            "Default: 0"
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to YOLOv8 model. "
            "Default: weights/best.pt"
        ),
    )

    args = parser.parse_args()

    config = ApplicationConfig()

    if args.model is not None:

        config.detection.model_path = (
            args.model
        )

    config.camera_source = (
        parse_source(args.source)
    )

    create_directories(config)

    pipeline = build_pipeline(
        config,
        use_motion_fallback=True,
    )

    pipeline.run(
        source=config.camera_source,
        window_name=(
            config.display_window_name
        ),
    )


if __name__ == "__main__":
    main()
