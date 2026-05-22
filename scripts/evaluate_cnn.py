from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class ImageFolderLite(Dataset):
    IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(self, root: Path, classes: list[str], transform=None):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.root}")

        self.classes = classes
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
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/real")
    ap.add_argument("--model-path", default="models/document_cnn.pt")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device)
    classes = checkpoint["classes"]

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

    ds = ImageFolderLite(Path(args.data_dir) / "test", classes=classes, transform=eval_tfms)
    loader = DataLoader(ds, batch_size=args.batch_size)

    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Test accuracy: {acc:.4f}")
    print(report)

    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "test_accuracy": acc,
                "classes": classes,
                "num_test_samples": len(ds),
                "architecture": "resnet18_transfer_learning",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    print(f"Saved evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()