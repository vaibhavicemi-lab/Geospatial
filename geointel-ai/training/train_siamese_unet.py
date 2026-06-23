import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.siamese_unet import SiameseUNet


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class LEVIRChangeDataset(Dataset):
    """LEVIR-CD style dataset: split/A, split/B, split/label."""

    def __init__(self, root: str | Path, split: str = "train", image_size: int = 256):
        self.root = Path(root)
        self.split_dir = self.root / split
        self.a_dir = self.split_dir / "A"
        self.b_dir = self.split_dir / "B"
        self.label_dir = self.split_dir / "label"
        self.image_size = int(image_size)
        self.samples = self._index_samples()

    def _index_samples(self) -> List[Tuple[Path, Path, Path, str]]:
        for directory in [self.a_dir, self.b_dir, self.label_dir]:
            if not directory.exists():
                raise FileNotFoundError(f"Required dataset folder is missing: {directory}")

        samples = []
        before_files = sorted([p for p in self.a_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        for before_path in before_files:
            after_path = self._matching_file(self.b_dir, before_path.stem)
            label_path = self._matching_file(self.label_dir, before_path.stem)
            if after_path and label_path:
                samples.append((before_path, after_path, label_path, before_path.stem))

        if not samples:
            raise FileNotFoundError(
                f"No matched A/B/label samples found under {self.split_dir}. "
                "Make sure filenames match across A, B, and label folders."
            )
        return samples

    @staticmethod
    def _matching_file(directory: Path, stem: str) -> Path | None:
        for extension in IMAGE_EXTENSIONS:
            candidate = directory / f"{stem}{extension}"
            if candidate.exists():
                return candidate
        matches = list(directory.glob(f"{stem}.*"))
        return matches[0] if matches else None

    def __len__(self) -> int:
        return len(self.samples)

    def _read_rgb_tensor(self, path: Path) -> torch.Tensor:
        image = np.array(Image.open(path).convert("RGB"))
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)

    def _read_mask_tensor(self, path: Path) -> torch.Tensor:
        mask = np.array(Image.open(path).convert("L"))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        return torch.from_numpy(mask).unsqueeze(0)

    def __getitem__(self, index: int):
        before_path, after_path, label_path, sample_id = self.samples[index]
        return {
            "before": self._read_rgb_tensor(before_path),
            "after": self._read_rgb_tensor(after_path),
            "mask": self._read_mask_tensor(label_path),
            "sample_id": sample_id,
        }


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    return ((2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)).mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 1.0 - dice_score(pred, target)


def bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy(pred, target)


def combined_bce_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return bce_loss(pred, target) + dice_loss(pred, target)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred_bin = (pred >= threshold).float()
    target_bin = (target >= 0.5).float()

    tp = (pred_bin * target_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - target_bin)).sum().item()
    fp = (pred_bin * (1 - target_bin)).sum().item()
    fn = ((1 - pred_bin) * target_bin).sum().item()
    eps = 1e-7

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "dice": float(dice),
    }


def move_batch_to_device(batch: Dict, device: torch.device):
    before = batch["before"].to(device, non_blocking=True)
    after = batch["after"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    return before, after, mask


def average_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "dice": 0.0}
    return {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]}


def train_one_epoch(model: SiameseUNet, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    metrics = []
    for batch in loader:
        before, after, mask = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(before, after)
        loss = combined_bce_dice_loss(pred, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        metrics.append(compute_metrics(pred.detach(), mask))
    return total_loss / max(len(loader), 1), average_metrics(metrics)


@torch.no_grad()
def validate(model: SiameseUNet, loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    metrics = []
    for batch in loader:
        before, after, mask = move_batch_to_device(batch, device)
        pred = model(before, after)
        total_loss += combined_bce_dice_loss(pred, mask).item()
        metrics.append(compute_metrics(pred, mask))
    return total_loss / max(len(loader), 1), average_metrics(metrics)


def build_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available. Use --device cpu or --device auto.")
    return device


def print_epoch(epoch: int, epochs: int, train_loss: float, train_metrics: Dict[str, float], val_loss: float, val_metrics: Dict[str, float]) -> None:
    print(
        f"Epoch {epoch:03d}/{epochs:03d} | "
        f"train_loss={train_loss:.4f} train_iou={train_metrics['iou']:.4f} train_dice={train_metrics['dice']:.4f} | "
        f"val_loss={val_loss:.4f} val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
        f"val_recall={val_metrics['recall']:.4f} val_f1={val_metrics['f1']:.4f} val_dice={val_metrics['dice']:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Siamese U-Net on a LEVIR-CD style change detection dataset.")
    parser.add_argument("--data_dir", required=True, help="Dataset root containing train/val/test folders.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", default="models/weights/siamese_unet_best.pth")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_metric", choices=["iou", "dice"], default="iou")
    return parser.parse_args()


def main():
    args = parse_args()
    device = build_device(args.device)
    print(f"Using device: {device}")

    train_dataset = LEVIRChangeDataset(args.data_dir, "train", args.image_size)
    val_dataset = LEVIRChangeDataset(args.data_dir, "val", args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SiameseUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, device)
        print_epoch(epoch, args.epochs, train_loss, train_metrics, val_loss, val_metrics)

        score = val_metrics[args.save_metric]
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "best_metric": args.save_metric,
                    "best_score": best_score,
                },
                save_path,
            )
            print(f"Saved best model to {save_path} using val_{args.save_metric}={best_score:.4f}")

    print(f"Training complete. Best val_{args.save_metric}: {best_score:.4f}")


if __name__ == "__main__":
    main()
