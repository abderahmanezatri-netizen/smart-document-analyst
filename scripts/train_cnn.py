from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ImageFolderLite(Dataset):
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, root: Path, classes: list[str] | None = None, transform=None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        self.classes = classes or sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for c in self.classes:
            class_dir = self.root / c
            if not class_dir.exists():
                continue

            for p in sorted(class_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in self.IMAGE_EXTS:
                    self.samples.append((p, self.class_to_idx[c]))

        if not self.samples:
            raise FileNotFoundError(f"No supported image files found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)


def build_model(num_classes: int):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--out", default="models/document_cnn.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data = Path(args.data_dir)

    train_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomRotation(3),
            transforms.RandomResizedCrop(224, scale=(0.90, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_ds = ImageFolderLite(data / "train", transform=train_tfms)
    val_ds = ImageFolderLite(data / "val", classes=train_ds.classes, transform=eval_tfms)

    Path("models").mkdir(exist_ok=True)
    Path("outputs/training").mkdir(parents=True, exist_ok=True)
    Path("models/classes.json").write_text(json.dumps(train_ds.classes, indent=2), encoding="utf-8")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = build_model(len(train_ds.classes)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        running = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            running += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()

        train_acc = correct / total

        model.eval()
        vtot = 0
        vcorr = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x).argmax(1)
                vcorr += (pred == y).sum().item()
                vtot += y.numel()

        val_acc = vcorr / vtot

        row = {
            "epoch": epoch,
            "train_loss": running / total,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }

        history.append(row)
        print(row)

        if val_acc >= best:
            best = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_ds.classes,
                    "val_acc": val_acc,
                    "architecture": "resnet18_transfer_learning",
                    "image_size": 224,
                },
                args.out,
            )

    Path("outputs/training/history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved best model to {args.out}; best val_acc={best:.4f}")


if __name__ == "__main__":
    main()