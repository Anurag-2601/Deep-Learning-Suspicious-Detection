
"""
Unit tests for the detection/behavior fusion component.
"""

import pytest

from src.fusion import (
    FusionDecision,
    fuse_predictions,
)


def test_high_confidence_dangerous_object_and_behavior():
    """
    A dangerous object combined with suspicious behavior
    should produce a suspicious decision.
    """

    result = fuse_predictions(
        object_class="knife",
        object_confidence=0.90,
        behavior_class="fighting",
        behavior_confidence=0.90,
    )

    assert isinstance(
        result,
        FusionDecision,
    )

    assert result.is_suspicious is True


def test_safe_behavior_with_no_dangerous_object():
    """
    A normal behavior with no dangerous object should
    not produce a suspicious decision.
    """

    result = fuse_predictions(
        object_class=None,
        object_confidence=0.0,
        behavior_class="walking",
        behavior_confidence=0.90,
    )

    assert result.is_suspicious is False


def test_low_confidence_detection():
    """
    Low-confidence object detection should not automatically
    produce a high-confidence suspicious decision.
    """

    result = fuse_predictions(
        object_class="knife",
        object_confidence=0.20,
        behavior_class="walking",
        behavior_confidence=0.80,
    )

    assert result.is_suspicious is False


def test_suspicious_object_handling():
    """
    Suspicious-object handling combined with a dangerous
    object should produce a suspicious decision.
    """

    result = fuse_predictions(
        object_class="knife",
        object_confidence=0.85,
        behavior_class="suspicious_object_handling",
        behavior_confidence=0.90,
    )

    assert result.is_suspicious is True


def test_falling_behavior_without_weapon():
    """
    Falling alone should not necessarily be classified as
    a dangerous-object event.
    """

    result = fuse_predictions(
        object_class=None,
        object_confidence=0.0,
        behavior_class="falling",
        behavior_confidence=0.90,
    )

    assert result.is_suspicious is False
