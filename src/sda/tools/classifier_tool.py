from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Type

import torch
from crewai.tools import BaseTool
from PIL import Image
from pydantic import BaseModel, Field
from torch import nn
from torchvision import models, transforms

from sda.utils.logging import log_action

class ClassificationInput(BaseModel):
    document_path: str = Field(..., description="Path to a scanned document image to classify.")


class CNNDocumentClassifierTool(BaseTool):
    name: str = "cnn_document_classifier"
    description: str = (
        "Classifies a document using a trained PyTorch CNN model. "
        "Returns predicted class, confidence, class probabilities, and human-review flag."
    )
    args_schema: Type[BaseModel] = ClassificationInput

    def __init__(
        self,
        model_path: str = "models/document_cnn.pt",
        classes_path: str = "models/classes.json",
        confidence_threshold: float = 0.60,
    ):
        super().__init__()
        object.__setattr__(self, "model_path", Path(model_path))
        object.__setattr__(self, "classes_path", Path(classes_path))
        object.__setattr__(self, "confidence_threshold", confidence_threshold)

        checkpoint = torch.load(self.model_path, map_location="cpu")
        object.__setattr__(self, "checkpoint", checkpoint)

        classes = checkpoint.get("classes")
        if classes is None and self.classes_path.exists():
            classes = json.loads(self.classes_path.read_text(encoding="utf-8"))

        if not classes:
            raise ValueError("No classes found in model checkpoint or classes.json")

        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "model", self._load_model())
        object.__setattr__(self, "transform", self._build_transform())

    def _build_resnet18(self, num_classes: int):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def _load_model(self):
        architecture = self.checkpoint.get("architecture", "custom_document_cnn")

        if architecture == "resnet18_transfer_learning":
            model = self._build_resnet18(len(self.classes))
        else:
            from sda.models.cnn import DocumentCNN

            model = DocumentCNN(len(self.classes))

        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.eval()
        return model

    def _build_transform(self):
        architecture = self.checkpoint.get("architecture", "custom_document_cnn")

        if architecture == "resnet18_transfer_learning":
            return transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

        return None

    def _load_image_tensor(self, document_path: Path):
        architecture = self.checkpoint.get("architecture", "custom_document_cnn")

        if architecture == "resnet18_transfer_learning":
            image = Image.open(document_path).convert("RGB")
            return self.transform(image).unsqueeze(0)

        import numpy as np

        image = Image.open(document_path).convert("L").resize((128, 128))
        arr = np.array(image, dtype=np.float32) / 255.0
        return torch.tensor(arr).unsqueeze(0).unsqueeze(0)

    def _run(self, document_path: str) -> str:
        try:
            path = Path(document_path)

            if not path.exists():
                raise FileNotFoundError(f"Document not found: {path}")

            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                return json.dumps(
                    {
                        "error": "The CNN classifier expects an image file for the real dataset model.",
                        "document_path": str(path),
                        "supported_extensions": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
                        "needs_human_review": True,
                    },
                    indent=2,
                )

            x = self._load_image_tensor(path)

            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0)

            predicted_idx = int(torch.argmax(probs).item())
            predicted_class = self.classes[predicted_idx]
            confidence = float(probs[predicted_idx].item())

            class_probabilities = {
                class_name: float(probs[i].item())
                for i, class_name in enumerate(self.classes)
            }

            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "needs_human_review": confidence < self.confidence_threshold,
                "model_architecture": self.checkpoint.get("architecture", "custom_document_cnn"),
            }


            log_action(
                "Document Classification Agent",
                "cnn_classification_tool",
                "success",
                    {
                        "document_path": str(path),
                        "predicted_class": predicted_class,
                        "confidence": confidence,
                        "model_architecture": result["model_architecture"],
                    },
)
            return json.dumps(result, indent=2)

        except Exception as exc:
            log_action(
                "Document Classification Agent",
                "cnn_classification_tool",
                "error",
                {
                    "document_path": document_path,
                    "error": str(exc),
                },
)
            return json.dumps(
                {
                    "error": str(exc),
                    "needs_human_review": True,
                },
                indent=2,
            )