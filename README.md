# Deep Learning Surveillance and Anomaly Detection

A modular deep-learning surveillance system for detecting potentially dangerous objects and analyzing suspicious activity in surveillance scenes.

The project is designed around a multi-stage pipeline that combines object detection, behavior analysis, prediction fusion, and alert generation.

---

## Overview

The system is organized as a modular surveillance pipeline combining
object detection, behavior analysis, prediction fusion, alert
generation, and event logging.

```mermaid
flowchart TD

    A[Video / Images] --> B[Object Detection]

    B --> C[Knife]
    B --> D[Scissor]
    B --> E[Gun]

    B --> F[Behavior Analysis]

    F --> G[Temporal Activity]
    F --> H[Behavior Cues]

    B --> I[Prediction Fusion]
    F --> I

    I --> J[Suspicion / Alert Decision]

    J --> K[Logging & Evidence]             
````

The repository separates detection, behavior analysis, fusion, alert generation, logging, training, and evaluation into independent components.

---

## Key Features

* Object detection for potentially dangerous objects
* Modular behavior-analysis component
* Detection and behavior prediction fusion
* Suspicious-event decision logic
* Alert generation
* Detection and alert logging
* Configurable thresholds and runtime parameters
* Training and dataset-preparation scripts
* Unit testing for prediction fusion
* Evaluation results and visual evidence
* Wrong-class false-positive analysis
* Model-weight storage
* Reproducible project configuration

---

## Detected Object Classes

The detection evaluation includes the following primary object categories:

| Class   |
| ------- |
| Knife   |
| Scissor |
| Gun     |

---

## Repository Structure

```text
deep-learning-surveillance/
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── train/
│   │   ├── files.md
│   │   ├── images.png
│   │   └── labels.png
│   │
│   ├── valid/
│   │   ├── file.md
│   │   ├── images.png
│   │   └── labels.png
│   │
│   └── test/
│       ├── file.md
│       ├── images.png
│       └── labels.png
│
├── notebooks/
│   └── surveillance_wrongclass_fp_fixed.ipynb
│
├── results/
│   ├── README.md
│   ├── detection_metrics.md
│   └── visualizations/
│       ├── 1. performance_report.jpg
│       ├── 2. knife_detection.jpg
│       ├── 3. person_scissor_detection.jpg
│       └── 4. scissor_detection.jpg
│
├── scripts/
│   ├── prepare_behavior_dataset.py
│   └── train_behavior.py
│
├── src/
│   ├── __init__.py
│   ├── alerts.py
│   ├── behavior.py
│   ├── config.py
│   ├── detection.py
│   ├── fusion.py
│   ├── logger.py
│   ├── main.py
│   └── pipeline.py
│
├── tests/
│   ├── __init__.py
│   └── test_fusion.py
│
├── weights/
│   ├── .gitkeep
│   ├── README.md
│   └── best.pt
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Project Components

## `src/`

The `src/` directory contains the main application components.

### `detection.py`

Contains the object-detection component of the surveillance pipeline.

The detector is intended to identify objects such as:

* Knife
* Scissor
* Gun

and provide detection confidence and object information.

### `behavior.py`

Contains the behavior-analysis component.

This module provides the interface for analyzing temporal activity and behavioral information from surveillance data.

### `fusion.py`

Combines information from the object-detection and behavior-analysis components.

The objective of the fusion stage is to combine multiple prediction sources before making a suspicious-event decision.

### `alerts.py`

Contains suspicious-event and alert-generation logic.

The alert component uses detection/fusion information and configured thresholds to determine when an event should be reported.

### `logger.py`

Provides logging functionality for detection and alert events.

Logged information can include detection class, confidence, timestamp, and associated evidence.

### `pipeline.py`

Coordinates the different components into the surveillance-processing pipeline.

### `config.py`

Loads and manages project configuration.

### `main.py`

Provides the main application entry point.

---

# Configuration

Runtime and model parameters are centralized in:

```text
configs/default.yaml
```

Configuration includes parameters related to:

* Dataset paths
* Detection thresholds
* Behavior thresholds
* Fusion parameters
* Alert settings
* Logging settings
* Training parameters
* Runtime options

Keeping configuration separate from application code makes the system easier to modify and reproduce.

---

# Dataset

The project uses separate training, validation, and testing splits.

The repository organizes the dataset as:

```text
data/
├── train/
├── valid/
└── test/
```

### Dataset Splits

| Directory | Purpose         |
| --------- | --------------- |
| `train/`  | Training data   |
| `valid/`  | Validation data |
| `test/`   | Testing data    |

The dataset is maintained separately from the source-code repository where necessary because of dataset size and storage constraints.

The dataset was organized using the following external Google Drive locations:

* **Training:** [Google Drive Training Dataset](https://drive.google.com/drive/folders/1orkSXst-S-8CLoGBCzE0PO_S2ijIYvO5?usp=drive_link)
* **Validation:** [Google Drive Validation Dataset](https://drive.google.com/drive/folders/16CPbdVPhcd37DFIrOnlhk4IgRK0o3kdU?usp=drive_link)
* **Testing:** [Google Drive Testing Dataset](https://drive.google.com/drive/folders/1rHM3QIR8rDiLtGj4BAkaoDZAmVghTVtz?usp=drive_link)

> **Note:** The repository uses `valid/` as the validation directory. Configuration paths and scripts should use the same naming convention.

---

# Model Weights

Model checkpoints are stored in:

```text
weights/
```

The repository currently contains:

```text
weights/
├── best.pt
├── .gitkeep
└── README.md
```

`best.pt` is the trained object-detection checkpoint associated with the project.

Model checkpoints can be large, so model-weight distribution should take GitHub file-size limitations into account.

---

# Training and Dataset Preparation

The repository contains two training/data-preparation scripts:

```text
scripts/
├── prepare_behavior_dataset.py
└── train_behavior.py
```

## Dataset Preparation

`prepare_behavior_dataset.py` prepares the data required by the behavior-analysis workflow.

## Behavior Training

`train_behavior.py` contains the training workflow for the behavior-analysis component.

Training parameters are controlled through the project configuration where applicable.

---

# Evaluation

Evaluation results are stored under:

```text
results/
```

The documented surveillance detection session produced the following results.

## Session Performance

| Metric                       |   Result |
| ---------------------------- | -------: |
| Session Duration             | 85.7 sec |
| Total Frames Processed       |      131 |
| Average FPS                  |     2.63 |
| Average Detection Confidence |   75.90% |
| Total Alerts                 |        5 |
| Wrong-Class Alerts           |        1 |

## Detection Metrics

| Metric                           | Result |
| -------------------------------- | -----: |
| Accuracy                         |  3.82% |
| Precision                        | 83.33% |
| Recall                           |  3.85% |
| F1 Score                         |  7.35% |
| True Positives (TP)              |      5 |
| False Positives - Low Confidence |      0 |
| False Positives - Wrong Class    |      1 |
| False Negatives (proxy)          |    125 |

### Evaluation Note

The reported false-negative value of `125` is explicitly described in the evaluation report as a proxy based on frames without detections.

Therefore, it should not be interpreted as a conventional object-level false-negative count obtained through ground-truth bounding-box matching.

The reported precision, recall, and F1 values should be interpreted within the evaluation procedure used for this experiment.

---

# Per-Class Results

| Class   | TP | FP | Wrong-Class FP | Precision | Average Confidence |
| ------- | -: | -: | -------------: | --------: | -----------------: |
| Knife   |  3 |  0 |              0 |    100.0% |              81.4% |
| Scissor |  2 |  0 |              0 |    100.0% |              84.2% |
| Gun     |  0 |  1 |              1 |      0.0% |              70.2% |

The evaluation identified one wrong-class false positive:

```text
Knife → Gun
```

This class-confusion case is one of the documented failure cases of the current evaluation.

---

# Visual Results

Representative detection outputs are included in:

```text
results/visualizations/
```

## Knife Detection

The detector identified a knife with a confidence of approximately `0.82`.

![Knife Detection](results/visualizations/2.%20knife_detection.jpg)

---

## Scissor Detection

The detector identified a scissor with a confidence of approximately `0.88`.

![Scissor Detection](results/visualizations/4.%20scissor_detection.jpg)

---

## Person and Scissor Detection

The system produced simultaneous detections of a person and a scissor in the same surveillance frame.

![Person and Scissor Detection](results/visualizations/3.%20person_scissor_detection.jpg)

---

## Performance Report

The complete session performance report is included below.

![Performance Report](results/visualizations/1.%20performance_report.jpg)

Additional evaluation information is available in:

```text
results/README.md
results/detection_metrics.md
```

---

# Wrong-Class False Positive Analysis

A dedicated notebook is included for the analysis of the observed wrong-class false positive:

```text
notebooks/
└── surveillance_wrongclass_fp_fixed.ipynb
```

The documented error was:

```text
Expected / actual object: Knife
Predicted object: Gun
```

The notebook preserves the analysis associated with this class-confusion case.

---

# Testing

Unit tests are located in:

```text
tests/
```

Current tests include the prediction-fusion component:

```text
tests/
└── test_fusion.py
```

The tests are intended to verify expected fusion behavior for combinations of object detections and behavioral predictions.

---

# Reproducibility

Important project parameters are centralized in:

```text
configs/default.yaml
```

The intended workflow is:

```text
1. Prepare dataset
        ↓
2. Configure parameters
        ↓
3. Train / load models
        ↓
4. Run object detection
        ↓
5. Analyze behavior
        ↓
6. Fuse predictions
        ↓
7. Generate alerts
        ↓
8. Log results
        ↓
9. Evaluate performance
```

---

# Limitations

The current evaluation highlights several limitations.

## 1. Low Recall

The reported recall is:

```text
3.85%
```

This indicates that the evaluation procedure resulted in many missed detections relative to the reported positive cases.

## 2. Wrong-Class Detection

One wrong-class false positive was observed:

```text
Knife → Gun
```

Class confusion remains an area for improvement.

## 3. Runtime Performance

The evaluated session processed:

```text
131 frames
2.63 FPS
```

This indicates that inference speed requires further optimization for real-time surveillance applications.

## 4. Evaluation Methodology

The reported false-negative value is a proxy based on frames without detections rather than a conventional object-level false-negative calculation.

For a rigorous object-detection benchmark, future evaluation should use ground-truth bounding-box matching and standard object-detection metrics.

---

# Future Improvements

Potential improvements include:

* Improve object-detection recall
* Reduce wrong-class predictions
* Increase inference FPS
* Improve dataset diversity
* Perform systematic hyperparameter tuning
* Improve temporal behavior modeling
* Improve object/behavior prediction fusion
* Introduce temporal smoothing for unstable detections
* Use rigorous ground-truth evaluation
* Evaluate the complete pipeline on longer surveillance sequences
* Add automated experiment tracking
* Expand unit and integration test coverage

---

# Technology Stack

The project is implemented as a Python-based deep-learning and computer-vision pipeline.

Core technologies include:

* Python
* PyTorch
* Computer vision
* Object detection
* Temporal behavior analysis
* YAML-based configuration
* OpenCV-based image/video processing
* Automated testing

Required Python packages are listed in:

```text
requirements.txt
```

---

# Project Status

| Component                   | Status   |
| --------------------------- | -------- |
| Repository structure        | Complete |
| Dataset organization        | Complete |
| Object detection component  | Included |
| Behavior-analysis component | Included |
| Prediction fusion           | Included |
| Alert system                | Included |
| Logging                     | Included |
| Configuration               | Included |
| Training scripts            | Included |
| Evaluation results          | Included |
| Visual results              | Included |
| Wrong-class analysis        | Included |
| Fusion tests                | Included |
| Object-detection checkpoint | Included |

The quantitative results documented in this repository correspond to the current evaluation session and should not be interpreted as a state-of-the-art benchmark.

---

# Ethical and Safety Considerations

This project is intended for research and educational purposes in computer vision and surveillance analytics.

Automated surveillance systems can produce false positives, false negatives, and class-confusion errors. Detection results should therefore not be treated as definitive evidence of harmful intent or activity.

Any real-world deployment should include appropriate human review, privacy safeguards, security controls, and evaluation for the intended operational environment.  

---

# Acknowledgements

This repository documents the development and evaluation of a deep-learning-based surveillance and anomaly-detection pipeline, including object detection, behavioral analysis, prediction fusion, alert generation, logging, and evaluation.

```

**Note:** I kept the metrics and terminology from the performance report you provided rather than silently changing them. In particular, the `125` false negatives are labeled as a **proxy**, exactly because your report describes them that way.
```

