"""
Prepare the behavior dataset for temporal training.

The behavior dataset is organized into:
    data/train/
    data/val/
    data/test/

The project uses the following behavior classes:
    standing
    walking
    running
    falling
    fighting
    loitering
    suspicious_object_handling

This script reads COCO-style annotations and creates temporal
clip manifests for the SlowFast behavior-training pipeline.

Output:
    data/behavior_prepared/
        train_manifest.json
        val_manifest.json
        test_manifest.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


BEHAVIOR_CLASSES = [
    "standing",
    "walking",
    "running",
    "falling",
    "fighting",
    "loitering",
    "suspicious_object_handling",
]


@dataclass
class FrameRecord:
    """Information associated with one dataset frame."""

    image_id: int
    file_name: str
    category_ids: List[int]
    track_ids: List[int]


@dataclass
class ClipRecord:
    """A temporal sequence used by the behavior model."""

    clip_id: str
    split: str
    label: str
    frame_paths: List[str]
    frame_ids: List[int]
    track_id: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare temporal behavior clips."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing train, val and test.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/behavior_prepared"),
        help="Directory for generated manifests.",
    )

    parser.add_argument(
        "--clip-length",
        type=int,
        default=16,
        help="Number of frames in each temporal clip.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Sliding-window stride.",
    )

    return parser.parse_args()


def find_annotation_file(
    split_directory: Path,
) -> Path:
    """
    Find the COCO annotation JSON associated with a dataset split.
    """

    candidates = [
        split_directory / "_annotations.coco.json",
        split_directory / "annotations.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    json_files = list(
        split_directory.glob("*.json")
    )

    if len(json_files) == 1:
        return json_files[0]

    raise FileNotFoundError(
        f"No unambiguous COCO annotation file found in "
        f"{split_directory}"
    )


def load_coco(
    annotation_file: Path,
) -> dict:
    """Load and validate the COCO annotation file."""

    with annotation_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        coco = json.load(file)

    required = {
        "images",
        "annotations",
        "categories",
    }

    missing = required - set(coco)

    if missing:
        raise ValueError(
            f"Invalid COCO annotation file. "
            f"Missing fields: {sorted(missing)}"
        )

    return coco


def normalize_class_name(
    name: str,
) -> str:
    """Normalize annotation class names."""

    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_category_map(
    coco: dict,
) -> Dict[int, str]:
    """Create category_id -> normalized class name mapping."""

    category_map: Dict[int, str] = {}

    for category in coco["categories"]:

        category_id = int(
            category["id"]
        )

        name = normalize_class_name(
            category["name"]
        )

        category_map[
            category_id
        ] = name

    return category_map


def build_annotations_by_image(
    coco: dict,
) -> Dict[int, list]:
    """Group annotations by image ID."""

    annotations_by_image: Dict[int, list] = {}

    for annotation in coco["annotations"]:

        image_id = int(
            annotation["image_id"]
        )

        annotations_by_image.setdefault(
            image_id,
            [],
        ).append(annotation)

    return annotations_by_image


def build_frame_records(
    coco: dict,
    category_map: Dict[int, str],
) -> List[FrameRecord]:
    """Convert COCO records into frame-level records."""

    annotations_by_image = (
        build_annotations_by_image(coco)
    )

    records: List[FrameRecord] = []

    for image in coco["images"]:

        image_id = int(
            image["id"]
        )

        annotations = (
            annotations_by_image.get(
                image_id,
                [],
            )
        )

        category_ids: List[int] = []
        track_ids: List[int] = []

        for annotation in annotations:

            category_id = int(
                annotation["category_id"]
            )

            if category_id in category_map:
                category_ids.append(
                    category_id
                )

            if "track_id" in annotation:

                try:
                    track_ids.append(
                        int(
                            annotation["track_id"]
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        records.append(
            FrameRecord(
                image_id=image_id,
                file_name=image["file_name"],
                category_ids=category_ids,
                track_ids=track_ids,
            )
        )

    return records


def get_behavior_label(
    record: FrameRecord,
    category_map: Dict[int, str],
) -> Optional[str]:
    """
    Get the behavior class associated with a frame.

    Frames without one of the project's behavior classes
    are excluded from temporal behavior clips.
    """

    labels = []

    for category_id in record.category_ids:

        class_name = category_map.get(
            category_id
        )

        if class_name in BEHAVIOR_CLASSES:
            labels.append(
                class_name
            )

    if not labels:
        return None

    # Keep the selection deterministic when multiple
    # behavior annotations occur in a frame.
    return sorted(labels)[0]


def resolve_frame_path(
    split_directory: Path,
    file_name: str,
) -> Path:
    """Resolve a frame filename inside a dataset split."""

    direct_path = (
        split_directory / file_name
    )

    if direct_path.exists():
        return direct_path

    filename = Path(file_name).name

    matches = list(
        split_directory.rglob(
            filename
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"Frame not found: {file_name}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files found for: {file_name}"
        )

    return matches[0]


def sort_frames(
    records: List[FrameRecord],
) -> List[FrameRecord]:
    """
    Provide deterministic frame ordering.

    Image IDs are used as the ordering key because the exact
    temporal metadata structure of the exported dataset is
    not assumed here.
    """

    return sorted(
        records,
        key=lambda record: record.image_id,
    )


def generate_clips(
    records: List[FrameRecord],
    category_map: Dict[int, str],
    split_directory: Path,
    split_name: str,
    clip_length: int,
    stride: int,
) -> List[ClipRecord]:
    """
    Generate fixed-length temporal clips.

    A clip is retained only when all frames in the window
    have the same behavior label.
    """

    records = sort_frames(records)

    labeled_frames = []

    for record in records:

        label = get_behavior_label(
            record,
            category_map,
        )

        if label is None:
            continue

        frame_path = resolve_frame_path(
            split_directory,
            record.file_name,
        )

        labeled_frames.append(
            (
                record,
                label,
                frame_path,
            )
        )

    clips: List[ClipRecord] = []

    clip_index = 0

    if len(labeled_frames) < clip_length:
        return clips

    for start in range(
        0,
        len(labeled_frames)
        - clip_length
        + 1,
        stride,
    ):

        window = labeled_frames[
            start:start + clip_length
        ]

        labels = [
            item[1]
            for item in window
        ]

        if len(set(labels)) != 1:
            continue

        label = labels[0]

        frame_records = [
            item[0]
            for item in window
        ]

        frame_paths = [
            str(item[2])
            for item in window
        ]

        track_ids = []

        for record in frame_records:
            track_ids.extend(
                record.track_ids
            )

        track_id = (
            track_ids[0]
            if track_ids
            else None
        )

        clips.append(
            ClipRecord(
                clip_id=(
                    f"{split_name}_"
                    f"{clip_index:06d}"
                ),
                split=split_name,
                label=label,
                frame_paths=frame_paths,
                frame_ids=[
                    record.image_id
                    for record in frame_records
                ],
                track_id=track_id,
            )
        )

        clip_index += 1

    return clips


def save_manifest(
    clips: List[ClipRecord],
    output_file: Path,
) -> None:
    """Save clip metadata as JSON."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = [
        asdict(clip)
        for clip in clips
    ]

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
        )


def process_split(
    data_root: Path,
    output_directory: Path,
    split_name: str,
    clip_length: int,
    stride: int,
) -> int:
    """Prepare one dataset split."""

    split_directory = (
        data_root / split_name
    )

    if not split_directory.exists():
        raise FileNotFoundError(
            f"Dataset split does not exist: "
            f"{split_directory}"
        )

    annotation_file = (
        find_annotation_file(
            split_directory
        )
    )

    coco = load_coco(
        annotation_file
    )

    category_map = build_category_map(
        coco
    )

    records = build_frame_records(
        coco,
        category_map,
    )

    clips = generate_clips(
        records=records,
        category_map=category_map,
        split_directory=split_directory,
        split_name=split_name,
        clip_length=clip_length,
        stride=stride,
    )

    output_file = (
        output_directory
        / f"{split_name}_manifest.json"
    )

    save_manifest(
        clips,
        output_file,
    )

    print(
        f"[INFO] {split_name}: "
        f"{len(records)} frames -> "
        f"{len(clips)} clips"
    )

    return len(clips)


def main() -> None:
    args = parse_args()

    if args.clip_length <= 0:
        raise ValueError(
            "clip-length must be greater than zero."
        )

    if args.stride <= 0:
        raise ValueError(
            "stride must be greater than zero."
        )

    total_clips = 0

    for split in (
        "train",
        "val",
        "test",
    ):

        total_clips += process_split(
            data_root=args.data_root,
            output_directory=args.output,
            split_name=split,
            clip_length=args.clip_length,
            stride=args.stride,
        )

    print(
        f"[INFO] Total generated clips: "
        f"{total_clips}"
    )


if __name__ == "__main__":
    main()
