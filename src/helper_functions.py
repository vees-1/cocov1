import os
import zipfile
import requests
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torch import nn

import torchvision
from torchvision import transforms
from PIL import Image

def set_seeds(seed: int = 42) -> None:
    """Sets random seeds for Python, NumPy, PyTorch CPU and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def walk_through_dir(dir_path: str) -> None:
    """Prints a summary of subdirectories and file counts inside dir_path."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")


def download_data(
    source: str,
    destination: str,
    remove_source: bool = True,
) -> Path:
    """Downloads a ZIP dataset from source URL and extracts it.

    Args:
        source: URL pointing to a .zip file.
        destination: Folder name under data/ to extract into.
        remove_source: Delete the zip after extraction (default True).

    Returns:
        pathlib.Path to the extracted data folder.

    Example:
        download_data(
            source="https://example.com/data.zip",
            destination="my_dataset"
        )
    """
    data_path  = Path("data/")
    image_path = data_path / destination

    if image_path.is_dir():
        print(f"[INFO] {image_path} already exists, skipping download.")
    else:
        print(f"[INFO] Creating {image_path}...")
        image_path.mkdir(parents=True, exist_ok=True)
        target_file = Path(source).name
        zip_path    = data_path / target_file

        print(f"[INFO] Downloading {target_file} from {source}...")
        with open(zip_path, "wb") as f:
            f.write(requests.get(source).content)

        print(f"[INFO] Extracting {target_file}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(image_path)

        if remove_source:
            os.remove(zip_path)

    return image_path

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Returns accuracy (%) between true labels and predicted labels.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    return (correct / len(y_pred)) * 100


def precision_recall_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Computes precision, recall, and F1 score.

    Args:
        y_true: Ground-truth class indices, shape (N,).
        y_pred: Predicted class indices, shape (N,).
        num_classes: Number of classes.
        average: 'macro' (unweighted mean) or 'weighted' (weighted by support).

    Returns:
        {'precision': ..., 'recall': ..., 'f1': ...}
    """
    precisions, recalls, f1s, supports = [], [], [], []

    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().float()
        fp = ((y_pred == c) & (y_true != c)).sum().float()
        fn = ((y_pred != c) & (y_true == c)).sum().float()
        support = (y_true == c).sum().float()

        p  = tp / (tp + fp + 1e-8)
        r  = tp / (tp + fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)

        precisions.append(p.item())
        recalls.append(r.item())
        f1s.append(f1.item())
        supports.append(support.item())

    supports_arr = np.array(supports)
    total = supports_arr.sum() + 1e-8

    if average == "weighted":
        precision = float(np.average(precisions, weights=supports_arr))
        recall    = float(np.average(recalls,    weights=supports_arr))
        f1        = float(np.average(f1s,        weights=supports_arr))
    else:  # macro
        precision = float(np.mean(precisions))
        recall    = float(np.mean(recalls))
        f1        = float(np.mean(f1s))

    return {"precision": precision, "recall": recall, "f1": f1}

def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Mean Absolute Error.
    """
    return (y_true.float() - y_pred.float()).abs().mean().item()


def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Root Mean Squared Error.
    """
    return ((y_true.float() - y_pred.float()) ** 2).mean().sqrt().item()


def r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Coefficient of determination (R²).
    """
    y_true = y_true.float()
    ss_res = ((y_true - y_pred.float()) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (ss_res / (ss_tot + 1e-8)).item()

def print_train_time(start: float, end: float, device: Optional[str] = None) -> float:
    """
    Prints and returns elapsed time in seconds.
    """
    total = end - start
    label = f" on {device}" if device else ""
    print(f"\nTrain time{label}: {total:.3f} seconds")
    return total

def plot_loss_curves(
    results: Dict[str, List],
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (15, 5),
) -> None:
    """
    Plots training and validation loss and accuracy curves.

    Args:
        results: Dict with keys 'train_loss', 'val_loss' (or 'test_loss'),
                 'train_acc', 'val_acc' (or 'test_acc').
        title: Figure title.
        figsize: Matplotlib figure size.
    """
    # Accept both 'val_' and 'test_' prefixed keys
    val_loss_key = "val_loss" if "val_loss" in results else "test_loss"
    val_acc_key  = "val_acc"  if "val_acc"  in results else "test_acc"

    epochs = range(1, len(results["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title)

    # Loss
    axes[0].plot(epochs, results["train_loss"], label="Train loss")
    axes[0].plot(epochs, results[val_loss_key], label="Val loss", linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, results["train_acc"], label="Train acc")
    axes[1].plot(epochs, results[val_acc_key], label="Val acc", linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_lr_curve(results: Dict[str, List], figsize: Tuple[int, int] = (8, 4)) -> None:
    """
    Plots learning rate over epochs (if 'lr' key exists in results).
    """
    if "lr" not in results:
        print("[WARN] 'lr' key not found in results dict.")
        return
    epochs = range(1, len(results["lr"]) + 1)
    plt.figure(figsize=figsize)
    plt.plot(epochs, results["lr"])
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.yscale("log")
    plt.show()

def plot_decision_boundary(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    resolution: int = 200,
) -> None:
    """
    Plots the decision boundary of a 2-D classification model.

    Args:
        model: Trained model expecting (N, 2) input.
        X: Feature tensor of shape (N, 2).
        y: Label tensor of shape (N,).
        resolution: Grid resolution (higher = smoother boundary).
    """
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min.item(), x_max.item(), resolution),
        np.linspace(y_min.item(), y_max.item(), resolution),
    )
    grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

    model.eval()
    with torch.inference_mode():
        logits = model(grid)

    if len(torch.unique(y)) > 2:
        z = torch.softmax(logits, dim=1).argmax(dim=1)
    else:
        z = torch.round(torch.sigmoid(logits)).squeeze()

    z = z.reshape(xx.shape).numpy()
    plt.contourf(xx, yy, z, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k", s=30)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions=None,
    figsize: Tuple[int, int] = (10, 7),
) -> None:
    """
    Plots training/test data and optional predictions on a 1-D axis.
    """
    plt.figure(figsize=figsize)
    plt.scatter(train_data, train_labels, c="b", s=10, label="Train data")
    plt.scatter(test_data,  test_labels,  c="g", s=10, label="Test data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=10, label="Predictions")
    plt.legend(prop={"size": 14})
    plt.show()

def confusion_matrix(
    preds, 
    targets, 
    class_names=None):
    n = int(max(preds.max(), targets.max())) + 1
    cm = torch.bincount(
        n * targets + preds,
        minlength=n*n
    ).reshape(n, n)

    if class_names:
        print("      " + " ".join(f"{name:>8}" for name in class_names))
        for i, row in enumerate(cm):
            print(f"{class_names[i]:>6} " + " ".join(f"{int(x):>8}" for x in row))
    else:
        print(cm)

    return cm

def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: Optional[List[str]] = None,
    transform=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Predicts on a single image and displays it with the predicted label.

    Works with both file paths and PIL Images.

    Args:
        model: Trained PyTorch model.
        image_path: Path to the image file.
        class_names: List of class name strings.
        transform: Optional torchvision transform. Defaults to ImageNet normalisation.
        device: Device to run inference on.
    """
    img = Image.open(image_path).convert("RGB")

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    model.to(device)
    model.eval()

    with torch.inference_mode():
        tensor = transform(img).unsqueeze(0).to(device)
        logits = model(tensor)

    probs  = torch.softmax(logits, dim=1)
    label  = torch.argmax(probs, dim=1).item()

    title = (
        f"Pred: {class_names[label]} | Prob: {probs.max():.3f}"
        if class_names else
        f"Pred: {label} | Prob: {probs.max():.3f}"
    )

    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()


def pred_and_plot_batch(
    model: torch.nn.Module,
    image_paths: List[str],
    class_names: Optional[List[str]] = None,
    transform=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    cols: int = 4,
) -> None:
    """
    Predicts on a batch of images and displays them in a grid.

    Args:
        model: Trained PyTorch model.
        image_paths: List of image file paths.
        class_names: List of class name strings.
        transform: Optional transform (defaults to ImageNet normalisation).
        device: Device for inference.
        cols: Number of columns in the display grid.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    n    = len(image_paths)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    model.to(device)
    model.eval()

    for i, path in enumerate(image_paths):
        img    = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.inference_mode():
            probs = torch.softmax(model(tensor), dim=1)
        label = torch.argmax(probs, dim=1).item()
        title = f"{class_names[label]} ({probs.max():.2f})" if class_names else str(label)

        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def display_random_images(
    dataset: torch.utils.data.Dataset,
    class_names: Optional[List[str]] = None,
    n: int = 10,
    seed: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 6),
) -> None:
    """Displays n random images from a PyTorch Dataset.

    Args:
        dataset: Any PyTorch Dataset returning (image_tensor, label).
        class_names: List of class names for title display.
        n: Number of images to display.
        seed: Optional random seed for reproducibility.
        figsize: Figure size.
    """
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(dataset)), k=min(n, len(dataset)))

    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        title = class_names[label] if class_names else str(label)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")

    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
