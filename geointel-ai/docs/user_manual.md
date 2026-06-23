# User Manual

## Install

```bash
cd geointel-ai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows activation:

```bash
.venv\Scripts\activate
```

## Run

```bash
streamlit run app.py
```

## Upload Images

1. Open the Streamlit URL.
2. Select `Upload & Analyze`.
3. Either keep the demo image pair enabled or upload a before image and an after image.
4. Supported formats are PNG, JPG, JPEG, TIF, and TIFF.

## Run Analysis

1. Select `Fast Classical Demo Mode` for immediate local detection.
2. Select `Deep Learning Mode` only after training or adding weights at `data/outputs/siamese_unet_best.pth`.
3. Click `Run Change Detection`.
4. Review the binary mask, red overlay, metrics, alert card, and AI explanation.

## Review Alerts

1. In the analyst review section, select a status:
   - Pending
   - Accepted
   - Rejected
   - Needs Further Review
2. Enter remarks.
3. Click `Save Alert Review`.

## Export Reports

1. Open `Alert History`.
2. Review saved records.
3. Click `Download CSV Report` or `Download PDF Report`.

## Train Model

Arrange LEVIR-CD as:

```text
train/A train/B train/label
val/A val/B val/label
test/A test/B test/label
```

Run:

```bash
python training/train_siamese_unet.py --data_dir /path/to/LEVIR-CD --epochs 30 --batch_size 4 --output data/outputs/siamese_unet_best.pth
```

Evaluate:

```bash
python training/evaluate.py --data_dir /path/to/LEVIR-CD --weights data/outputs/siamese_unet_best.pth
```
