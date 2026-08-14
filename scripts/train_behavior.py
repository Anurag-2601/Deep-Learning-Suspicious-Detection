"""
Train the temporal behavior-analysis model.

This script is the training entry point for the behavior-analysis
branch of the surveillance system.

Expected project flow:

    Video clips
        ↓
    Temporal preprocessing
        ↓
    SlowFast behavior model
        ↓
    Behavior classification
        ↓
    Saved behavior checkpoint

IMPORTANT:
The project currently does not include a trained SlowFast checkpoint
or a finalized behavior-training dataset specification. Therefore,
this script validates the training configuration and provides the
entry point for the eventual SlowFast training implementation.

It should NOT be used to claim that the behavior model has already
been trained.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


DEFAULT_CLASSES = [
    "standing",
    "walking",
    "running",
    "fighting",
    "falling",
    "loitering",
    "suspicious_object_handling",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Train the behavior-analysis model for "
            "the surveillance system."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Path to the behavior dataset directory."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("weights/behavior"),
        help=(
            "Directory in which behavior-model "
            "checkpoints will be saved."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )

    parser.add_argument(
        "--clip-length",
        type=int,
        default=16,
        help=(
            "Number of consecutive video frames "
            "used as one temporal clip."
        ),
    )

    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help=(
            "Behavior classes used by the model."
        ),
    )

    return parser.parse_args()


def validate_arguments(
    data_directory: Path,
    output_directory: Path,
    epochs: int,
    batch_size: int,
    clip_length: int,
    classes: List[str],
) -> None:
    """Validate the training configuration."""

    if not data_directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{data_directory}"
        )

    if not data_directory.is_dir():
        raise NotADirectoryError(
            f"Dataset path is not a directory: "
            f"{data_directory}"
        )

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if clip_length <= 0:
        raise ValueError(
            "clip_length must be greater than zero."
        )

    if not classes:
        raise ValueError(
            "At least one behavior class is required."
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def print_training_configuration(
    data_directory: Path,
    output_directory: Path,
    epochs: int,
    batch_size: int,
    clip_length: int,
    classes: List[str],
) -> None:
    """Display the selected training configuration."""

    print()
    print("=" * 60)
    print("BEHAVIOR MODEL TRAINING CONFIGURATION")
    print("=" * 60)

    print(f"Dataset       : {data_directory}")
    print(f"Output        : {output_directory}")
    print(f"Epochs        : {epochs}")
    print(f"Batch size    : {batch_size}")
    print(f"Clip length   : {clip_length}")
    print()

    print("Behavior classes:")

    for index, class_name in enumerate(
        classes
    ):
        print(
            f"  {index}: {class_name}"
        )

    print("=" * 60)
    print()


def train_slowfast(
    data_directory: Path,
    output_directory: Path,
    epochs: int,
    batch_size: int,
    clip_length: int,
    classes: List[str],
) -> None:
    """
    Train the SlowFast model.

    This function is intentionally not implemented yet because
    the supplied project materials do not contain enough information
    to faithfully reproduce the behavior-training pipeline.

    Before implementing it, we need to establish:

    1. The exact behavior dataset.
    2. Video/clip directory structure.
    3. Annotation format.
    4. Number of frames per clip.
    5. SlowFast model configuration.
    6. Input preprocessing.
    7. Training/validation split.
    8. Number of classes.
    9. Checkpoint format.

    Once those are fixed, the implementation belongs here.
    """

    raise NotImplementedError(
        "SlowFast training has not yet been implemented. "
        "The behavior dataset and training specification "
        "must be finalized first."
    )


def main() -> None:
    """Main training entry point."""

    args = parse_args()

    validate_arguments(
        data_directory=args.data,
        output_directory=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        clip_length=args.clip_length,
        classes=args.classes,
    )

    print_training_configuration(
        data_directory=args.data,
        output_directory=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        clip_length=args.clip_length,
        classes=args.classes,
    )

    train_slowfast(
        data_directory=args.data,
        output_directory=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        clip_length=args.clip_length,
        classes=args.classes,
    )


if __name__ == "__main__":
    main()
