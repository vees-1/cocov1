import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

def set_seeds(seed: int = 42) -> None:
    """Sets random seeds for Python, NumPy, and PyTorch (CPU + GPU).

    Args:
        seed: Integer seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[INFO] Seeds set to {seed}.")

def get_device() -> str:
    """Returns the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info() -> None:
    """Prints a summary of the available compute devices."""
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU             : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version    : {torch.version.cuda}")
    print(f"MPS available   : {torch.backends.mps.is_available()}")
    print(f"Best device     : {get_device()}")

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str,
) -> Path:
    """Saves a PyTorch model's state_dict to target_dir/model_name.

    Args:
        model: PyTorch model to save.
        target_dir: Directory to save the model in.
        model_name: Filename (must end with '.pth' or '.pt').

    Returns:
        Path to the saved file.

    Example:
        save_model(model, target_dir="models", model_name="my_model.pth")
    """
    assert model_name.endswith((".pth", ".pt")), \
        "model_name must end with '.pt' or '.pth'"
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    save_path = target_dir_path / model_name
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] Model state_dict saved → {save_path}")
    return save_path


def load_model(
    model: torch.nn.Module,
    model_path: str,
    device: Optional[str] = None,
    strict: bool = True,
) -> torch.nn.Module:
    """Loads a state_dict into model from model_path.

    Args:
        model: Instantiated model with the correct architecture.
        model_path: Path to saved .pth / .pt file.
        device: Target device (auto-detected if None).
        strict: Whether to strictly enforce key matching (default True).

    Returns:
        Model with loaded weights, moved to device and set to eval mode.
    """
    device = device or get_device()
    state  = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=strict)
    model.to(device)
    model.eval()
    print(f"[INFO] Model loaded from {model_path} → {device}")
    return model


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    results: Dict,
    target_dir: str,
    model_name: str,
    scheduler=None,
    extra: Optional[Dict] = None,
) -> Path:
    """Saves a full training checkpoint (model + optimizer + epoch + results).

    Args:
        model: Current model.
        optimizer: Current optimizer.
        epoch: Current epoch number.
        results: Training results dict.
        target_dir: Save directory.
        model_name: Filename (should end with '.pth').
        scheduler: Optional LR scheduler state.
        extra: Any extra key-value pairs to store in the checkpoint.

    Returns:
        Path to saved checkpoint.
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    save_path = target_dir_path / model_name

    checkpoint = {
        "epoch":            epoch,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "results":          results,
    }
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if extra:
        checkpoint.update(extra)

    torch.save(checkpoint, save_path)
    print(f"[INFO] Checkpoint saved → {save_path}  (epoch {epoch})")
    return save_path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: Optional[str] = None,
    scheduler=None,
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict]:
    """Loads a checkpoint saved by save_checkpoint().

    Args:
        model: Model with matching architecture.
        optimizer: Optimizer to restore (pass None to skip).
        checkpoint_path: Path to checkpoint file.
        device: Target device (auto-detected if None).
        scheduler: Optional LR scheduler to restore.

    Returns:
        (model, optimizer, checkpoint_dict)  — optimizer may be None.
    """
    device     = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    epoch = checkpoint.get("epoch", 0)
    print(f"[INFO] Checkpoint loaded from {checkpoint_path}  (epoch {epoch})")
    return model, optimizer, checkpoint

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Returns total and trainable parameter counts.

    Args:
        model: PyTorch model.

    Returns:
        {'total': ..., 'trainable': ..., 'non_trainable': ...}
    """
    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params      : {total:,}")
    print(f"Trainable params  : {trainable:,}")
    print(f"Frozen params     : {total - trainable:,}")
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


def model_summary(
    model: torch.nn.Module,
    input_size: Optional[Tuple] = None,
    device: Optional[str] = None,
) -> None:
    """Prints a lightweight model summary.

    If torchinfo is installed and input_size is provided, uses it for
    a detailed summary; otherwise falls back to str(model) + param count.

    Args:
        model: PyTorch model.
        input_size: E.g. (1, 3, 224, 224). Required for torchinfo summary.
        device: Target device.
    """
    try:
        from torchinfo import summary
        device = device or get_device()
        if input_size:
            summary(model, input_size=input_size, device=device)
        else:
            print(model)
            count_parameters(model)
    except ImportError:
        print(model)
        count_parameters(model)

def print_train_time(start: float, end: float, device: Optional[str] = None) -> float:
    """Prints and returns elapsed training time.

    Args:
        start: Start time (from time.time() or timeit).
        end: End time.
        device: Device string for display.

    Returns:
        Elapsed time in seconds.
    """
    total = end - start
    label = f" on {device}" if device else ""
    print(f"\nTrain time{label}: {total:.3f} seconds")
    return total

def confusion_matrix_tensor(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Computes a confusion matrix as a torch.Tensor.

    Args:
        preds: Predicted class indices, shape (N,).
        targets: Ground-truth class indices, shape (N,).
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    save_path: str,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Optional[Dict] = None,
    opset_version: int = 17,
) -> None:
    """Exports a model to ONNX format.

    Args:
        model: Trained PyTorch model (will be set to eval mode).
        dummy_input: Example input tensor matching the model's expected input.
        save_path: Output .onnx file path.
        input_names: Names for input nodes (default ['input']).
        output_names: Names for output nodes (default ['output']).
        dynamic_axes: Dict for dynamic dimensions, e.g.
                      {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}.
        opset_version: ONNX opset version (default 17).
    """
    input_names   = input_names  or ["input"]
    output_names  = output_names or ["output"]
    dynamic_axes  = dynamic_axes or {n: {0: "batch_size"} for n in input_names + output_names}

    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )
    print(f"[INFO] Model exported to ONNX → {save_path}")
