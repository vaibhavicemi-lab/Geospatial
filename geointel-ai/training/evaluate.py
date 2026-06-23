import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.siamese_unet import SiameseUNet
from training.train_siamese_unet import LEVIRChangeDataset, average_metrics, build_device, compute_metrics, move_batch_to_device


def load_checkpoint(model: SiameseUNet, weights_path: str | Path, device: torch.device) -> dict:
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
    model.load_state_dict(checkpoint)
    return {}


@torch.no_grad()
def evaluate(model: SiameseUNet, loader: DataLoader, device: torch.device, output_dir: Path, threshold: float = 0.5):
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = []

    for batch in loader:
        before, after, mask = move_batch_to_device(batch, device)
        pred = model(before, after)
        metrics.append(compute_metrics(pred, mask, threshold=threshold))

        pred_np = pred.squeeze(1).detach().cpu().numpy()
        sample_ids = batch["sample_id"]
        for item, sample_id in zip(pred_np, sample_ids):
            binary = (item >= threshold).astype(np.uint8) * 255
            cv2.imwrite(str(output_dir / f"{sample_id}_pred.png"), binary)

    return average_metrics(metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Siamese U-Net on the LEVIR-CD test split.")
    parser.add_argument("--data_dir", required=True, help="Dataset root containing train/val/test folders.")
    parser.add_argument("--weights", default="models/weights/siamese_unet_best.pth")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--output_dir", default="data/outputs/predicted_masks")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = build_device(args.device)
    print(f"Using device: {device}")

    dataset = LEVIRChangeDataset(args.data_dir, "test", args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SiameseUNet().to(device)
    checkpoint = load_checkpoint(model, args.weights, device)
    if checkpoint:
        print(
            "Loaded checkpoint: "
            f"epoch={checkpoint.get('epoch', 'unknown')} "
            f"image_size={checkpoint.get('image_size', args.image_size)} "
            f"best_{checkpoint.get('best_metric', 'score')}={checkpoint.get('best_score', 'unknown')}"
        )

    metrics = evaluate(model, loader, device, Path(args.output_dir), args.threshold)
    print(f"Saved predicted masks to {args.output_dir}")
    print("Test metrics:")
    for key in ["iou", "precision", "recall", "f1", "accuracy", "dice"]:
        print(f"{key}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
