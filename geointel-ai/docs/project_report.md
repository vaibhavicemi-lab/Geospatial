# GeoIntel AI Project Report

## 1. Abstract

GeoIntel AI is an AI-based geospatial intelligence system that compares before and after satellite images to detect land-use changes. The system performs preprocessing, change detection, alert generation, analyst review, and report export. It includes a fast OpenCV-based demo mode and a PyTorch Siamese U-Net model for deep learning change detection.

## 2. Introduction

Satellite imagery is widely used for urban monitoring, disaster response, agriculture, defense, and environmental observation. Manual comparison of images is slow and error-prone. GeoIntel AI assists analysts by automatically highlighting changed regions and generating structured alerts.

## 3. Problem Statement

The problem is to detect meaningful changes between two satellite images captured at different times and present the results in a form that an analyst can review, store, and export.

## 4. Objectives

- Accept before and after satellite images.
- Preprocess images safely.
- Detect changed areas using classical and deep learning methods.
- Analyze changed regions and assign alert levels.
- Explain changes in simple English.
- Support analyst review and report export.

## 5. Existing System

Traditional systems often require manual image interpretation or expensive commercial GIS platforms. Many academic projects stop at mask generation and do not include alert review, persistent storage, and reporting.

## 6. Proposed System

The proposed system is a local Streamlit application with OpenCV, PyTorch, SQLite, and report generation. It is free, beginner-friendly, and suitable for a final-year project demonstration.

## 7. System Architecture

```text
Input Images -> Preprocessing -> Change Detection -> Region Analysis
             -> Explanation -> Alert Generation -> Review Dashboard -> Reports
```

## 8. Methodology

Images are converted to RGB and resized to a common shape. In classical mode, the grayscale absolute difference is blurred, thresholded, cleaned with morphology, and converted to contours. In deep learning mode, before and after images are processed by a Siamese U-Net and converted into a probability mask.

## 9. Algorithms Used

- Absolute image differencing.
- Gaussian blur.
- Binary thresholding.
- Morphological opening and closing.
- Contour detection.
- Connected region analysis.
- Siamese U-Net segmentation.
- Rule-based alert classification.

## 10. Model Description

The Siamese U-Net uses a shared encoder for before and after images. Feature maps are subtracted using absolute difference. The decoder upsamples these difference features to produce a one-channel sigmoid change probability mask. The model can be fine-tuned using the provided PyTorch training script.

## 11. Dataset Description

The recommended dataset is LEVIR-CD, a building change detection dataset with before images, after images, and binary labels. The expected format is:

```text
dataset/
  train/A
  train/B
  train/label
  val/A
  val/B
  val/label
  test/A
  test/B
  test/label
```

Folder `A` contains before images, folder `B` contains after images, and folder `label` contains binary change masks. Filenames should match across the three folders.

## 12. Implementation Details

The backend is Python. The dashboard is built with Streamlit. OpenCV handles classical detection. PyTorch handles deep learning. SQLite stores alerts and analyst reviews. ReportLab generates PDF reports, and Pandas generates CSV reports.

Fine-tuning command:

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

The training script resizes images, normalizes RGB values, converts labels into binary tensors, optimizes combined BCE + Dice loss, validates after each epoch, and saves the best checkpoint using validation IoU by default.

Evaluation command:

```bash
python training/evaluate.py \
  --data_dir data/LEVIR-CD \
  --weights models/weights/siamese_unet_best.pth \
  --output_dir data/outputs/predicted_masks
```

Evaluation metrics include IoU, precision, recall, F1-score, accuracy, and Dice score. Predicted masks are saved as PNG files.

## 13. Result Analysis

The system reports changed pixels, changed area percentage, number of changed regions, bounding boxes, alert level, risk score, and plain-English explanation. Classical mode is suitable for demonstration, while fine-tuned deep learning mode can improve robustness. The dashboard loads trained weights from `models/weights/siamese_unet_best.pth`; if the file is missing, it warns the user and falls back to classical change detection.

## 14. Advantages

- Runs fully on a local machine.
- No paid cloud or API dependency.
- Works immediately using classical mode.
- Includes full analyst workflow.
- Beginner-friendly training scripts and documentation.

## 15. Limitations

- Classical mode can be affected by lighting, shadows, clouds, and seasonal changes.
- Plain-English explanations are rule-based and not guaranteed.
- Pixel percentage is not equal to real-world area without geospatial metadata.
- Deep learning performance depends on trained weights.
- Accuracy depends on image alignment, spatial resolution, cloud cover, seasonal variation, quality of labels, and whether the trained dataset matches the uploaded imagery.

## 16. Future Scope

- Add GeoTIFF coordinate support.
- Add map-based visualization.
- Add trained land-use classification.
- Add transformer models.
- Add image thumbnails to PDF reports.
- Add multi-user review workflow.

## 17. Conclusion

GeoIntel AI demonstrates an end-to-end geospatial intelligence workflow. It combines image processing, deep learning, alert generation, explainability, database storage, and reporting in a free local project suitable for academic submission.
