import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from typing import Callable, Dict, List, Optional, Tuple, Union
from PIL import Image

# Default device
device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Default ImageNet transform
_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int] = (224, 224),
    transform: Optional[torchvision.transforms.Compose] = None,
    device: torch.device = device,
) -> Dict[str, Union[str, float]]:
    """Predicts on a target image and displays it with its label and probability.

    Args:
        model: Trained PyTorch model.
        class_names: Ordered list of class name strings.
        image_path: Filepath to the image.
        image_size: Resize target (used only when transform is None).
        transform: Custom transform. Defaults to ImageNet normalisation.
        device: Target device.

    Returns:
        {'class': predicted_class_name, 'probability': float, 'label_idx': int}
    """
    img = Image.open(image_path).convert("RGB")

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    model.to(device)
    model.eval()

    with torch.inference_mode():
        tensor = transform(img).unsqueeze(0).to(device)
        logits = model(tensor)

    probs   = torch.softmax(logits, dim=1)
    label   = torch.argmax(probs, dim=1).item()
    prob    = probs.max().item()

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[label]} | Prob: {prob:.3f}")
    plt.axis("off")
    plt.show()

    return {"class": class_names[label], "probability": prob, "label_idx": label}

def predict_top_k(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    k: int = 5,
    transform: Optional[torchvision.transforms.Compose] = None,
    device: torch.device = device,
) -> List[Dict[str, Union[str, float]]]:
    """Returns the top-k predicted classes and their probabilities.

    Args:
        model: Trained PyTorch model.
        class_names: Ordered list of class name strings.
        image_path: Filepath to the image.
        k: Number of top predictions to return.
        transform: Optional transform (defaults to ImageNet normalisation).
        device: Target device.

    Returns:
        List of dicts [{'class': ..., 'probability': ...}, ...] sorted by prob desc.
    """
    img = Image.open(image_path).convert("RGB")
    transform = transform or _DEFAULT_TRANSFORM

    model.to(device)
    model.eval()

    with torch.inference_mode():
        logits = model(transform(img).unsqueeze(0).to(device))

    probs    = torch.softmax(logits, dim=1).squeeze()
    top_probs, top_idxs = torch.topk(probs, k=min(k, len(class_names)))

    results = []
    for prob, idx in zip(top_probs, top_idxs):
        results.append({"class": class_names[idx.item()], "probability": prob.item()})

    return results

def predict_batch_images(
    model: torch.nn.Module,
    class_names: List[str],
    image_paths: List[str],
    transform: Optional[torchvision.transforms.Compose] = None,
    device: torch.device = device,
    cols: int = 4,
    show_plot: bool = True,
) -> List[Dict[str, Union[str, float]]]:
    """Predicts on a list of images and optionally shows a grid.

    Args:
        model: Trained PyTorch model.
        class_names: Ordered list of class names.
        image_paths: List of image file paths.
        transform: Optional transform.
        device: Target device.
        cols: Columns in the display grid.
        show_plot: Whether to display the image grid.

    Returns:
        List of prediction dicts: [{'image': path, 'class': ..., 'probability': ...}, ...]
    """
    transform = transform or _DEFAULT_TRANSFORM
    model.to(device)
    model.eval()

    results = []
    pil_images = []

    with torch.inference_mode():
        for path in image_paths:
            img    = Image.open(path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            probs  = torch.softmax(model(tensor), dim=1)
            label  = torch.argmax(probs, dim=1).item()
            results.append({
                "image": path,
                "class": class_names[label],
                "probability": probs.max().item(),
            })
            pil_images.append(img)

    if show_plot:
        n    = len(image_paths)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).flatten()

        for i, (img, r) in enumerate(zip(pil_images, results)):
            axes[i].imshow(img)
            axes[i].set_title(f"{r['class']}\n({r['probability']:.2f})", fontsize=8)
            axes[i].axis("off")
        for j in range(len(results), len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        plt.show()

    return results

def predict_with_tta(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    n_augments: int = 10,
    base_transform: Optional[torchvision.transforms.Compose] = None,
    aug_transform: Optional[torchvision.transforms.Compose] = None,
    device: torch.device = device,
) -> Dict[str, Union[str, float, torch.Tensor]]:
    """Predicts using Test-Time Augmentation by averaging n_augments predictions.

    Args:
        model: Trained PyTorch model.
        class_names: Ordered list of class names.
        image_path: Filepath to the image.
        n_augments: Number of augmented forward passes.
        base_transform: Transform applied before augmentation.
        aug_transform: Augmentation transform (default: random hflip + colour jitter).
        device: Target device.

    Returns:
        {'class': ..., 'probability': ..., 'all_probs': tensor of shape (num_classes,)}
    """
    img = Image.open(image_path).convert("RGB")

    if base_transform is None:
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    if aug_transform is None:
        aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(15),
        ])

    model.to(device)
    model.eval()

    all_probs = []
    with torch.inference_mode():
        for _ in range(n_augments):
            augmented = aug_transform(img)
            tensor    = base_transform(augmented).unsqueeze(0).to(device)
            probs     = torch.softmax(model(tensor), dim=1).squeeze()
            all_probs.append(probs)

    mean_probs = torch.stack(all_probs).mean(dim=0)
    label      = mean_probs.argmax().item()

    return {
        "class":       class_names[label],
        "probability": mean_probs.max().item(),
        "all_probs":   mean_probs.cpu(),
    }

def predict_tabular(
    model: torch.nn.Module,
    X: Union[torch.Tensor, np.ndarray],
    task: str = "multiclass",
    class_names: Optional[List[str]] = None,
    device: torch.device = device,
    batch_size: int = 256,
) -> Dict[str, torch.Tensor]:
    """Runs inference on tabular/tensor inputs.

    Args:
        model: Trained PyTorch model.
        X: Input features as Tensor or ndarray.
        task: 'multiclass' | 'binary' | 'multilabel' | 'regression'.
        class_names: Optional class names (for classification).
        device: Target device.
        batch_size: Mini-batch size to avoid OOM on large datasets.

    Returns:
        Dict with 'predictions' (class indices or raw values) and 'probabilities'
        (for classification tasks).
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)

    model.to(device)
    model.eval()

    all_preds, all_probs = [], []

    with torch.inference_mode():
        for i in range(0, len(X), batch_size):
            batch  = X[i : i + batch_size].to(device)
            logits = model(batch)

            if task == "multiclass":
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
            elif task == "binary":
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().squeeze()
            elif task == "multilabel":
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            else:  # regression
                probs = logits
                preds = logits

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    predictions  = torch.cat(all_preds)
    probabilities = torch.cat(all_probs)

    result: Dict[str, torch.Tensor] = {
        "predictions":   predictions,
        "probabilities": probabilities,
    }
    if class_names and task in ("multiclass", "binary"):
        result["class_names"] = [class_names[p.item()] for p in predictions]

    return result
