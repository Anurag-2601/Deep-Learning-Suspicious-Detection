# Detection Performance Metrics

## Session Performance

| Metric | Value |
|---|---:|
| Session Duration | 85.7 sec |
| Total Frames Processed | 131 |
| Average FPS | 2.63 |
| Average Detection Confidence | 75.90% |
| Total Alerts Fired | 5 |
| Wrong-Class Alerts (FP) | 1 |

## Performance Evaluation

| Metric | Value |
|---|---:|
| Accuracy | 3.82% |
| Precision | 83.33% |
| Recall | 3.85% |
| F1 Score | 7.35% |
| True Positives (TP) | 5 |
| False Positives - Low Confidence (FP1) | 0 |
| False Positives - Wrong Class (FP2) | 1 |
| False Negatives (FN) | 125 |

## Per-Class Breakdown

| Class | TP | FP | Wrong-Class | Precision | Average Confidence |
|---|---:|---:|---:|---:|---:|
| Knife | 3 | 0 | 0 | 100.0% | 81.4% |
| Scissor | 2 | 0 | 0 | 100.0% | 84.2% |
| Gun | 0 | 1 | 1 | 0.0% | 70.2% |

## Wrong-Class Detection

One wrong-class false positive was recorded:

- Ground-truth/expected object: Knife
- Detected class: Gun

## Interpretation

The detection session achieved high precision (83.33%) but very low recall
(3.85%). The low recall is primarily associated with the large number of
false negatives (125) relative to the five true positives.

The average detection confidence was 75.90%, indicating that the model
generally produced reasonably confident detections when it detected an
object.

The system processed 131 frames at an average rate of 2.63 FPS during
the evaluated session.

These results represent the reported object-detection/session evaluation.
They should not be interpreted as behavior-analysis or SlowFast metrics.
