# GeoIntel AI

AI-Based Geospatial Intelligence System Using Satellite Imagery and Deep Learning.

GeoIntel AI is a free, open-source final-year project for satellite image change detection, alert generation, analyst review, and CSV/PDF report export. It works locally with no paid APIs and no cloud services.

## Features

- Upload before and after satellite images in PNG, JPG, JPEG, TIFF formats.
- Built-in demo image pair for instant testing.
- Image preprocessing: RGB conversion, safe resizing, normalization utilities, and tiling support.
- Fast Classical Demo Mode using image difference, thresholding, morphology, and contour detection.
- Deep Learning Mode using a PyTorch Siamese U-Net when trained weights are available.
- Changed region analysis with pixel count, area percentage, bounding boxes, region count, and confidence score.
- Rule-based alert levels: Low, Medium, High, Critical.
- Plain-English AI explanations for detected changes.
- Analyst review workflow with status and remarks saved in SQLite.
- Alert history table with CSV and PDF export.
- Training and evaluation scripts for LEVIR-CD style datasets.
- Pytest tests for preprocessing, change detection, alerts, database, and reports.

## Architecture

```text
Satellite Images
    |
    v
Image Preprocessing
    |
    v
Change Detection
    |-- Fast Classical Demo Mode
    |-- Siamese U-Net Deep Learning Mode
    |
    v
Object / Region Analysis
    |
    v
Plain-English Change Explanation
    |
    v
Alert Generation
    |
    v
Analyst Review Dashboard
    |
    v
CSV / PDF Report Export
```

## Installation

```bash
cd geointel-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
cd geointel-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Demo Without Training

1. Start the app.
2. Go to `Upload & Analyze`.
3. Keep `Use built-in demo satellite pair` checked.
4. Select `Fast Classical Demo Mode`.
5. Click `Run Change Detection`.
6. Review the mask, overlay, alert card, AI explanation, and analyst review form.
7. Save the review and export reports from `Alert History`.

Classical mode is mandatory and works immediately without model weights.

## Fine-Tune with LEVIR-CD

Download LEVIR-CD or prepare a compatible dataset in this structure:

```text
dataset/
  train/
    A/
    B/
    label/
  val/
    A/
    B/
    label/
  test/
    A/
    B/
    label/
```

Each folder must contain matching filenames. For example, `train/A/001.png`, `train/B/001.png`, and `train/label/001.png`.

Fine-tune Siamese U-Net:

```bash
python training/train_siamese_unet.py \
  --data_dir data/LEVIR-CD \
  --epochs 30 \
  --batch_size 4 \
  --image_size 256 \
  --lr 0.0001 \
  --save_path models/weights/siamese_unet_best.pth \
  --device auto
```

Use `--device cuda` on a GPU machine, or keep `--device auto` for local, Google Colab, or Kaggle. The script loads image pairs from `A` and `B`, loads binary masks from `label`, resizes all inputs, normalizes RGB images to `0-1`, trains with combined BCE + Dice loss, validates after every epoch, and saves the best checkpoint by validation IoU.

Evaluate:

```bash
python training/evaluate.py \
  --data_dir data/LEVIR-CD \
  --weights models/weights/siamese_unet_best.pth \
  --batch_size 4 \
  --image_size 256 \
  --device auto \
  --output_dir data/outputs/predicted_masks
```

Evaluation saves predicted binary masks and reports:

- IoU: overlap between predicted and true changed pixels.
- Precision: how many predicted changed pixels are correct.
- Recall: how many true changed pixels were found.
- F1-score: balance between precision and recall.
- Accuracy: overall pixel-level correctness.
- Dice score: segmentation overlap score commonly used for masks.

After training, keep the checkpoint at `models/weights/siamese_unet_best.pth`, then select `Deep Learning Mode` in the dashboard. If weights are missing, the app shows a warning and falls back to classical mode.

## Tests

```bash
pytest tests/
```

## Project Structure

```text
geointel-ai/
  app.py
  config.py
  models/
  preprocessing/
  inference/
  dashboard/
  database/
  training/
  tests/
  docs/
  data/
```

## Screenshots

Add screenshots here after running the app:

- Home dashboard metrics
- Upload and analysis view
- Change mask and overlay
- AI explanation section
- Alert history and export buttons

## Limitations

- Classical mode detects visual pixel changes and may respond to shadows, seasonal vegetation, or lighting differences.
- Rule-based explanations are approximate and require analyst verification.
- Deep learning accuracy depends on dataset quality and training time.
- Area is reported as pixel percentage, not real-world square meters unless georeferencing is added.

## Future Enhancements

- GeoTIFF georeferencing and map coordinate output.
- Real area calculation using spatial resolution metadata.
- Transformer-based change detection models.
- Better land-use classification with trained classifiers.
- Interactive map viewer with Folium or Leaflet.
- Role-based authentication for analyst teams.
- Export reports with embedded images and maps.
