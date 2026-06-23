from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = DATA_DIR / "outputs"
DB_PATH = OUTPUTS_DIR / "geointel_alerts.db"
MODEL_WEIGHTS_DIR = BASE_DIR / "models" / "weights"
MODEL_WEIGHTS = MODEL_WEIGHTS_DIR / "siamese_unet_best.pth"

SUPPORTED_IMAGE_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]
DEFAULT_IMAGE_SIZE = (256, 256)
CLASSICAL_THRESHOLD = 32
MIN_REGION_AREA = 80


def ensure_directories() -> None:
    for directory in [RAW_DIR, PROCESSED_DIR, SAMPLES_DIR, OUTPUTS_DIR, MODEL_WEIGHTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
