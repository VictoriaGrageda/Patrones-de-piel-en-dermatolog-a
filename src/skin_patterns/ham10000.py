from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import MODELS_DIR, REPORTS_DIR, SUPPORTED_EXTENSIONS


HAM10000_LABELS = {
    "akiec": "Actinic keratoses / Bowen disease",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    metadata: Path
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"


class Ham10000Dataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_index: dict[str, int],
        image_size: int,
        augment: bool = False,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.label_to_index = label_to_index
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(row["image_path"])
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (self.image_size, self.image_size), method=Image.Resampling.BILINEAR)

        if self.augment:
            image = self._augment(image)

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        label = torch.tensor(self.label_to_index[row["dx"]], dtype=torch.long)
        return tensor, label

    @staticmethod
    def _augment(image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.2:
            image = ImageOps.flip(image)
        if random.random() < 0.5:
            image = image.rotate(random.uniform(-20, 20), resample=Image.Resampling.BILINEAR)
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        return image


class SmallDermCnn(nn.Module):
    """CNN propia: arquitectura local, pesos aleatorios, sin backbone preentrenado."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def index_images(data_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images[path.stem] = path
    return images


def load_ham10000_frame(data_dir: Path, metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"No se encontro metadata: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required = {"image_id", "dx"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en HAM10000_metadata.csv: {sorted(missing)}")

    images = index_images(data_dir)
    frame = metadata.loc[:, ["image_id", "dx"]].copy()
    frame["image_path"] = frame["image_id"].map(images)
    frame = frame.dropna(subset=["image_path"]).reset_index(drop=True)
    frame = frame[frame["dx"].isin(HAM10000_LABELS)].reset_index(drop=True)

    if frame.empty:
        raise ValueError("No se encontraron imagenes HAM10000 que coincidan con la metadata.")
    return frame


def split_frame(frame: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=config.val_size + config.test_size,
        stratify=frame["dx"],
        random_state=config.seed,
    )
    relative_test_size = config.test_size / (config.val_size + config.test_size)
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=relative_test_size,
        stratify=temp_frame["dx"],
        random_state=config.seed,
    )
    return train_frame, val_frame, test_frame


def build_loaders(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    label_to_index: dict[str, int],
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = Ham10000Dataset(train_frame, label_to_index, config.image_size, augment=True)
    val_dataset = Ham10000Dataset(val_frame, label_to_index, config.image_size)
    test_dataset = Ham10000Dataset(test_frame, label_to_index, config.image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    return train_loader, val_loader, test_loader


def class_weights(train_frame: pd.DataFrame, label_to_index: dict[str, int], device: torch.device) -> torch.Tensor:
    counts = train_frame["dx"].value_counts()
    total = float(len(train_frame))
    weights = [total / (len(label_to_index) * counts[label]) for label in label_to_index]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        loss_sum += float(loss.item()) * labels.size(0)
        predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    accuracy = float(np.mean(np.asarray(predictions) == np.asarray(targets)))
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    return {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def evaluate_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    index_to_label: dict[int, str],
) -> tuple[dict[str, object], np.ndarray]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    labels = [index_to_label[index] for index in range(len(index_to_label))]
    label_indexes = list(range(len(index_to_label)))
    report = classification_report(
        targets,
        predictions,
        labels=label_indexes,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=label_indexes)
    return report, matrix


def train_ham10000(config: TrainingConfig) -> dict[str, object]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    frame = load_ham10000_frame(config.data_dir, config.metadata)
    labels = sorted(frame["dx"].unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    train_frame, val_frame, test_frame = split_frame(frame, config)
    train_loader, val_loader, test_loader = build_loaders(train_frame, val_frame, test_frame, label_to_index, config)

    model = SmallDermCnn(num_classes=len(label_to_index)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_frame, label_to_index, device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1 = -1.0
    history: list[dict[str, float | int]] = []
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{config.epochs} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f} val_f1={row['val_macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_report, test_matrix = evaluate_predictions(model, test_loader, device, index_to_label)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "ham10000_cnn_from_scratch.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_to_index": label_to_index,
            "index_to_label": index_to_label,
            "class_names": HAM10000_LABELS,
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            "note": "Modelo CNN entrenado desde cero; no usa backbones ni pesos preentrenados.",
        },
        model_path,
    )

    pd.DataFrame(history).to_csv(REPORTS_DIR / "ham10000_training_history.csv", index=False)
    pd.DataFrame(test_matrix, index=labels, columns=labels).to_csv(REPORTS_DIR / "ham10000_confusion_matrix.csv")
    with (REPORTS_DIR / "ham10000_test_report.json").open("w", encoding="utf-8") as file:
        json.dump(test_report, file, indent=2)

    return {
        "model_path": str(model_path),
        "history": history,
        "test_report": test_report,
        "labels": labels,
        "samples": {
            "train": len(train_frame),
            "val": len(val_frame),
            "test": len(test_frame),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena una CNN propia desde cero con HAM10000, sin modelos preentrenados."
    )
    parser.add_argument("--data-dir", default="data/raw/HAM10000", help="Carpeta raiz del dataset HAM10000.")
    parser.add_argument(
        "--metadata",
        default="data/raw/HAM10000/HAM10000_metadata.csv",
        help="Ruta al archivo HAM10000_metadata.csv.",
    )
    parser.add_argument("--image-size", type=int, default=128, help="Tamano cuadrado de entrada.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano de lote.")
    parser.add_argument("--epochs", type=int, default=20, help="Cantidad de epocas.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Tasa de aprendizaje.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Regularizacion AdamW.")
    parser.add_argument("--num-workers", type=int, default=0, help="Procesos para cargar imagenes.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda o cuda:0.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla reproducible.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        data_dir=Path(args.data_dir),
        metadata=Path(args.metadata),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )
    result = train_ham10000(config)
    print("\nEntrenamiento finalizado")
    print(f"Modelo: {result['model_path']}")
    print(f"Muestras: {result['samples']}")
