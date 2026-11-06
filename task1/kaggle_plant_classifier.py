"""
Kaggle notebook script for competition 2026task1.

Expected setup:
  * Attach the 2026task1 competition data.
  * Enable a GPU and Internet (needed once for ImageNet weights).
  * Run all cells or Save Version -> Save & Run All.

Outputs:
  /kaggle/working/submission.csv
  /kaggle/working/class_to_idx.json
  /kaggle/working/validation_scores.csv
"""

from __future__ import annotations

import gc
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


SEED = 2026
N_FOLDS = int(os.environ.get("PLANT_N_FOLDS", "5"))
EPOCHS = int(os.environ.get("PLANT_EPOCHS", "10"))
IMAGE_SIZE = int(os.environ.get("PLANT_IMAGE_SIZE", "256"))
BATCH_SIZE = int(os.environ.get("PLANT_BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("PLANT_NUM_WORKERS", "2"))
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

INPUT_ROOT = Path(os.environ.get("PLANT_INPUT_ROOT", "/kaggle/input"))
WORK_ROOT = Path(os.environ.get("PLANT_WORK_ROOT", "/kaggle/working"))
PRETRAINED_PATH = os.environ.get("PLANT_PRETRAINED_PATH")
SUBMISSION_PATH = WORK_ROOT / "submission.csv"

ImageFile.LOAD_TRUNCATED_IMAGES = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def image_files(directory: Path, recursive: bool = False) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_inputs(root: Path) -> tuple[Path, Path, Path]:
    if not root.exists():
        raise FileNotFoundError(
            "/kaggle/input does not exist. Attach the 2026task1 competition data first."
        )

    sample_candidates = sorted(root.rglob("*.csv"))
    sample_candidates = [
        path for path in sample_candidates
        if "submission" in path.name.lower()
    ]
    if not sample_candidates:
        raise FileNotFoundError("Could not find the sample submission CSV.")
    sample_path = max(sample_candidates, key=lambda p: p.stat().st_size)

    train_candidates: list[tuple[int, Path]] = []
    for directory in root.rglob("*"):
        if not directory.is_dir() or directory.name.lower() != "train":
            continue
        class_dirs = [child for child in directory.iterdir() if child.is_dir()]
        class_counts = [len(image_files(child, recursive=True)) for child in class_dirs]
        nonempty_classes = sum(count > 0 for count in class_counts)
        total_images = sum(class_counts)
        if nonempty_classes >= 2 and total_images >= 20:
            train_candidates.append((total_images, directory))
    if not train_candidates:
        raise FileNotFoundError("Could not locate train/<class>/*.png directories.")
    train_dir = max(train_candidates, key=lambda item: item[0])[1]

    test_candidates: list[tuple[int, Path]] = []
    for directory in root.rglob("*"):
        if not directory.is_dir() or directory.name.lower() != "test":
            continue
        count = len(image_files(directory, recursive=True))
        if count > 0:
            test_candidates.append((count, directory))
    if not test_candidates:
        raise FileNotFoundError("Could not locate the test image directory.")
    test_dir = max(test_candidates, key=lambda item: item[0])[1]

    return train_dir, test_dir, sample_path


class PlantDataset(Dataset):
    def __init__(
        self,
        paths: list[str] | list[Path],
        labels: list[int] | np.ndarray | None,
        transform,
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
        image = self.transform(image)
        if self.labels is None:
            return image, self.paths[index].name
        return image, int(self.labels[index])


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE, scale=(0.65, 1.0), ratio=(0.8, 1.25)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomRotation(25),
            transforms.ColorJitter(
                brightness=0.20, contrast=0.20, saturation=0.20, hue=0.03
            ),
            transforms.ToTensor(),
            normalization,
            transforms.RandomErasing(
                p=0.20, scale=(0.02, 0.12), ratio=(0.5, 2.0), value="random"
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(288),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, eval_transform


def make_model(num_classes: int) -> nn.Module:
    if PRETRAINED_PATH:
        weights_path = Path(PRETRAINED_PATH)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
        model = models.efficientnet_b0(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
    else:
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception as exc:
            raise RuntimeError(
                "Could not load ImageNet pretrained weights. In the Kaggle notebook "
                "Settings panel, turn Internet on and rerun."
            ) from exc
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


@torch.inference_mode()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        losses.append(float(loss.item()))
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())

    score = f1_score(targets, predictions, average="micro")
    return float(np.mean(losses)), float(score)


@torch.inference_mode()
def predict_with_tta(model, loader, device, num_classes: int) -> np.ndarray:
    model.eval()
    all_probabilities: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    for images, _names in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits_original = model(images)
            logits_flipped = model(torch.flip(images, dims=(3,)))
            probabilities = (
                logits_original.softmax(dim=1) + logits_flipped.softmax(dim=1)
            ) / 2.0
        all_probabilities.append(probabilities.float().cpu().numpy())

    if not all_probabilities:
        return np.empty((0, num_classes), dtype=np.float32)
    return np.concatenate(all_probabilities, axis=0)


def train_fold(
    fold: int,
    train_paths: list[str],
    labels: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    test_loader: DataLoader,
    train_transform,
    eval_transform,
    num_classes: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    train_dataset = PlantDataset(
        [train_paths[index] for index in train_indices],
        labels[train_indices],
        train_transform,
    )
    valid_dataset = PlantDataset(
        [train_paths[index] for index in valid_indices],
        labels[valid_indices],
        eval_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    model = make_model(num_classes).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.10)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE / 50
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for images, batch_labels in train_loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, batch_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        scheduler.step()
        valid_loss, valid_score = evaluate(model, valid_loader, device)
        print(
            f"fold={fold + 1}/{N_FOLDS} epoch={epoch:02d}/{EPOCHS} "
            f"train_loss={np.mean(train_losses):.4f} "
            f"valid_loss={valid_loss:.4f} valid_micro_f1={valid_score:.4f}"
        )

        if valid_score > best_score:
            best_score = valid_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 4 and epoch >= 6:
                print(f"Early stopping fold {fold + 1}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    test_probabilities = predict_with_tta(model, test_loader, device, num_classes)

    del model, optimizer, scheduler, scaler, best_state
    del train_loader, valid_loader, train_dataset, valid_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return test_probabilities, best_score


def build_submission(
    sample_path: Path,
    test_paths: list[Path],
    predictions: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)
    if "ID" not in sample.columns or "Category" not in sample.columns:
        raise ValueError(
            f"Expected submission columns ID and Category, found {sample.columns.tolist()}"
        )

    predicted_classes = [classes[index] for index in predictions.argmax(axis=1)]
    by_name = {path.name: label for path, label in zip(test_paths, predicted_classes)}
    by_stem = {path.stem: label for path, label in zip(test_paths, predicted_classes)}

    def lookup(image_id) -> str:
        value = str(image_id)
        if value in by_name:
            return by_name[value]
        if Path(value).name in by_name:
            return by_name[Path(value).name]
        if Path(value).stem in by_stem:
            return by_stem[Path(value).stem]
        raise KeyError(f"No prediction matched submission ID: {value}")

    submission = sample.copy()
    submission["Category"] = submission["ID"].map(lookup)
    return submission


def main() -> None:
    seed_everything(SEED)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    train_dir, test_dir, sample_path = discover_inputs(INPUT_ROOT)
    print(f"Train directory: {train_dir}")
    print(f"Test directory: {test_dir}")
    print(f"Sample submission: {sample_path}")

    image_folder = ImageFolder(train_dir)
    train_paths = [path for path, _label in image_folder.samples]
    labels = np.asarray([label for _path, label in image_folder.samples], dtype=np.int64)
    classes = image_folder.classes
    test_paths = image_files(test_dir, recursive=True)

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Training images: {len(train_paths)}")
    print(f"Test images: {len(test_paths)}")
    print("Class counts:")
    for index, name in enumerate(classes):
        print(f"  {name}: {int((labels == index).sum())}")

    if len(classes) != 5:
        raise ValueError(f"Expected 5 classes, found {len(classes)}: {classes}")
    if len(train_paths) != 500:
        print(f"Warning: expected about 500 training images, found {len(train_paths)}.")

    with (WORK_ROOT / "class_to_idx.json").open("w", encoding="utf-8") as handle:
        json.dump(image_folder.class_to_idx, handle, ensure_ascii=False, indent=2)

    train_transform, eval_transform = make_transforms()
    test_dataset = PlantDataset(test_paths, None, eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Warning: GPU is not enabled. Training will be much slower.")

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_predictions: list[np.ndarray] = []
    fold_scores: list[float] = []

    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(train_paths, labels)
    ):
        probabilities, score = train_fold(
            fold=fold,
            train_paths=train_paths,
            labels=labels,
            train_indices=train_indices,
            valid_indices=valid_indices,
            test_loader=test_loader,
            train_transform=train_transform,
            eval_transform=eval_transform,
            num_classes=len(classes),
            device=device,
        )
        fold_predictions.append(probabilities)
        fold_scores.append(score)

    ensemble_probabilities = np.mean(fold_predictions, axis=0)
    submission = build_submission(
        sample_path, test_paths, ensemble_probabilities, classes
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    pd.DataFrame(
        {
            "fold": np.arange(1, N_FOLDS + 1),
            "best_micro_f1": fold_scores,
        }
    ).to_csv(WORK_ROOT / "validation_scores.csv", index=False)

    print(f"Fold scores: {[round(score, 4) for score in fold_scores]}")
    print(f"Mean CV micro-F1: {np.mean(fold_scores):.4f}")
    print(f"Saved submission: {SUBMISSION_PATH}")
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
