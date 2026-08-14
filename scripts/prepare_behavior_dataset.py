"""
Prepare the surveillance behavior dataset for temporal training.

Expected input:

data/
├── train/
├── val/
└── test/

Each split should contain RGB frames and a COCO-style annotation JSON.

The project documentation describes seven behavior classes:

    standing
    walking
    running
    falling
    fighting
    loitering
    suspicious_object_handling

The output is a manifest describing temporal clips.

Output:

data/behavior_prepared/
├── train_manifest.json
├── val_manifest.json
└── test_manifest.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    """Metadata for one annotated frame."""

    image_id: int
    file_name: str
    category_ids: List[int]
    track_ids: List[int]


@dataclass
class ClipRecord:
    """Metadata for one temporal behavior clip."""

    clip_id: str
    split: str
    label: str
    frame_paths: List[str]
    frame_ids: List[int]
    track_id: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare COCO-style surveillance frames "
            "for temporal behavior training."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing train/val/test.",
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
        help="Number of consecutive frames per clip.",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Sliding-window stride between clips.",
    )

    return parser.parse_args()


def find_annotation_file(
    split_directory: Path,
) -> Path:
    """
    Locate a COCO annotation JSON file.

    Supports common Roboflow-style names such as:
        _annotations.coco.json
        annotations.json
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
        f"Could not identify a COCO annotation file in "
        f"{split_directory}"
    )


def load_coco(
    annotation_file: Path,
) -> dict:
    with annotation_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    required_keys = {
        "images",
        "annotations",
        "categories",
    }

    missing = required_keys - set(data.keys())

    if missing:
        raise ValueError(
            f"Invalid COCO annotation file "
            f"{annotation_file}. Missing: {missing}"
        )

    return data


def build_category_map(
    coco: dict,
) -> Dict[int, str]:
    """Build category_id -> class_name mapping."""

    category_map = {}

    for category in coco["categories"]:

        category_id = int(
            category["id"]
        )

        category_name = (
            category["name"]
            .strip()
            .lower()
            .replace(" ", "_")
        )

        category_map[
            category_id
        ] = category_name

    return category_map


def build_frame_records(
    coco: dict,
    category_map: Dict[int, str],
) -> List[FrameRecord]:
    """
    Convert COCO image/annotation entries into frame records.
    """

    annotations_by_image = defaultdict(list)

    for annotation in coco["annotations"]:

        annotations_by_image[
            int(annotation["image_id"])
        ].append(annotation)

    records = []

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

        category_ids = []
        track_ids = []

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
                        int(annotation["track_id"])
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


def choose_behavior_label(
    record: FrameRecord,
    category_map: Dict[int, str],
) -> Optional[str]:
    """
    Determine the behavior label for a frame.

    The category names come from the project's documented
    behavior taxonomy.
    """

    labels = []

    for category_id in record.category_ids:

        category_name = category_map.get(
            category_id
        )

        if category_name in BEHAVIOR_CLASSES:

            labels.append(
                category_name
            )

    if not labels:
        return None

    # Deterministic selection if multiple behavior
    # annotations exist on the same frame.
    return sorted(labels)[0]


def resolve_frame_path(
    split_directory: Path,
    file_name: str,
) -> Path:
    """
    Resolve an image filename within the split directory.
    """

    direct = (
        split_directory
        / file_name
    )

    if direct.exists():
        return direct

    matches = list(
        split_directory.rglob(
            Path(file_name).name
        )
    )

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Frame not found: {file_name}"
        )

    raise RuntimeError(
        f"Multiple frames found for: {file_name}"
    )


def sort_records(
    records: List[FrameRecord],
) -> List[FrameRecord]:
    """
    Sort frames deterministically.

    COCO image IDs are used when no explicit frame
    ordering is available.
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
    Generate temporal clips using a sliding window.

    A clip receives a behavior label when the frames in
    the window provide a consistent behavior label.
    """

    records = sort_records(records)

    labeled_records = []

    for record in records:

        label = choose_behavior_label(
            record,
            category_map,
        )

        if label is None:
            continue

        frame_path = resolve_frame_path(
            split_directory,
            record.file_name,
        )

        labeled_records.append(
            (
                record,
                label,
                frame_path,
            )
        )

    clips = []

    clip_counter = 0

    for start in range(
        0,
        len(labeled_records)
        - clip_length
        + 1,
        stride,
    ):

        window = labeled_records[
            start : start + clip_length
        ]

        labels = [
            item[1]
            for item in window
        ]

        # Require temporal consistency.
        if len(set(labels)) != 1:
            continue

        label = labels[0]

        records_in_window = [
            item[0]
            for item in window
        ]

        paths_in_window = [
            str(item[2])
            for item in window
        ]

        track_ids = []

        for record in records_in_window:
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
                    f"{clip_counter:06d}"
                ),
                split=split_name,
                label=label,
                frame_paths=paths_in_window,
                frame_ids=[
                    record.image_id
                    for record in records_in_window
                ],
                track_id=track_id,
            )
        )

        clip_counter += 1

    return clips


def save_manifest(
    clips: List[ClipRecord],
    output_file: Path,
) -> None:

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

    split_directory = (
        data_root / split_name
    )

    if not split_directory.exists():
        raise FileNotFoundError(
            f"Missing dataset split: "
            f"{split_directory}"
        )

    annotation_file = (
        find_annotation_file(
            split_directory
        )
    )

    print(
        f"[INFO] {split_name}: "
        f"{annotation_file}"
    )

    coco = load_coco(
        annotation_file
    )

    category_map = build_category_map(
        coco
    )

    print(
        f"[INFO] Categories: "
        f"{category_map}"
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
        f"{len(records)} frames → "
        f"{len(clips)} clips"
    )

    print(
        f"[INFO] Manifest: "
        f"{output_file}"
    )

    return len(clips)


def main() -> None:

    args = parse_args()

    if args.clip_length <= 0:
        raise ValueError(
            "--clip-length must be > 0"
        )

    if args.stride <= 0:
        raise ValueError(
            "--stride must be > 0"
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

    print()
    print(
        f"[INFO] Total generated clips: "
        f"{total_clips}"
    )


if __name__ == "__main__":
    main()
