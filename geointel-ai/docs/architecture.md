# Architecture

## Pipeline

```text
Satellite Images
  -> Image Preprocessing
  -> Change Detection
  -> Object / Region Analysis
  -> Plain-English Explanation
  -> Alert Generation
  -> Analyst Review Dashboard
  -> CSV / PDF Report Export
```

## Modules

- `app.py`: Streamlit dashboard and page navigation.
- `preprocessing/`: image loading, RGB conversion, resizing, normalization, and tiling.
- `inference/change_detection.py`: classical and deep learning inference.
- `inference/object_detection.py`: region metrics and bounding boxes.
- `inference/change_explainer.py`: rule-based plain-English explanation.
- `inference/alert_generation.py`: alert level and recommendation logic.
- `models/`: U-Net and Siamese U-Net implementations.
- `database/`: SQLite schema and CRUD helpers.
- `dashboard/`: UI cards, visualization, CSV and PDF reporting.
- `training/`: LEVIR-CD training and evaluation scripts.
- `tests/`: automated pytest coverage.

## Data Flow

1. The user uploads two images or selects the built-in demo pair.
2. Images are converted to RGB and resized to a common size.
3. The selected mode produces a binary change mask.
4. Region analysis calculates changed pixels, percentage, bounding boxes, and risk score.
5. The explanation module classifies changed regions using visual rules and produces simple English.
6. Alert generation assigns Low, Medium, High, or Critical status.
7. The analyst reviews the alert, chooses a status, and enters remarks.
8. SQLite stores the alert and review.
9. The history page exports CSV and PDF reports.
