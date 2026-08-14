# Model Weights

This directory is reserved for trained model checkpoints used by the
surveillance system.

Expected model files include:

- `best.pt` — trained YOLO object-detection model
- `behavior_best.pth` — trained temporal behavior-analysis model

Model weights are not included in the Git repository by default because
trained checkpoint files can be large.

Place the required model weights in this directory when running the
inference, evaluation, or training pipeline.

## Expected Structure

```text
weights/
├── best.pt
├── .gitkeep
└── README.md
