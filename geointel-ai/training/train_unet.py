import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.unet import UNet
from training.train_siamese_unet import LEVIRChangeDataset, compute_metrics, dice_loss


def main():
    parser = argparse.ArgumentParser(description="Train concatenated-input U-Net baseline on LEVIR-CD style dataset.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output", default="data/outputs/unet_best.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(LEVIRChangeDataset(args.data_dir, "train", args.image_size), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(LEVIRChangeDataset(args.data_dir, "val", args.image_size), batch_size=args.batch_size)
    model = UNet(in_channels=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCELoss()
    best_f1 = -1.0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            before, after, label = batch["before"].to(device), batch["after"].to(device), batch["mask"].to(device)
            x = torch.cat([before, after], dim=1)
            optimizer.zero_grad()
            pred = model(x)
            loss = bce(pred, label) + dice_loss(pred, label)
            loss.backward()
            optimizer.step()
            total += loss.item()
        model.eval()
        metrics = []
        with torch.no_grad():
            for batch in val_loader:
                before, after, label = batch["before"].to(device), batch["after"].to(device), batch["mask"].to(device)
                pred = model(torch.cat([before, after], dim=1))
                metrics.append(compute_metrics(pred, label))
        f1 = sum(m["f1"] for m in metrics) / max(len(metrics), 1)
        print(f"Epoch {epoch}/{args.epochs} loss={total / max(len(train_loader), 1):.4f} val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), output)


if __name__ == "__main__":
    main()
