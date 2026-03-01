import torch
from tqdm.auto import tqdm
from typing import Callable, Dict, List, Optional, Tuple

def _get_predictions(logits: torch.Tensor, task: str):
    """Convert raw logits to class predictions according to task type."""
    if task == "multiclass":
        return torch.argmax(torch.softmax(logits, dim=1), dim=1)
    elif task == "binary":
        return torch.round(torch.sigmoid(logits)).long().squeeze()
    elif task == "multilabel":
        return (torch.sigmoid(logits) > 0.5).float()
    elif task == "regression":
        return logits  # no conversion needed
    else:
        raise ValueError(f"Unknown task '{task}'. Choose from: multiclass, binary, multilabel, regression.")


def _compute_accuracy(preds: torch.Tensor, targets: torch.Tensor, task: str) -> float:
    """Compute batch-level accuracy (returns 0 for regression)."""
    if task == "regression":
        return 0.0
    elif task == "multilabel":
        return (preds == targets).float().mean().item()
    else:
        return (preds == targets).float().mean().item()

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str = "multiclass",
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: Optional[float] = None,
    metric_fns: Optional[Dict[str, Callable]] = None,
) -> Dict[str, float]:
    """Trains a PyTorch model for a single epoch.

    Args:
        model: Model to train.
        dataloader: Training DataLoader.
        loss_fn: Loss function.
        optimizer: Optimizer.
        device: Target device.
        task: One of 'multiclass', 'binary', 'multilabel', 'regression'.
        scaler: Optional GradScaler for automatic mixed precision (AMP).
        max_grad_norm: If set, clips gradient norm to this value.
        metric_fns: Optional dict of extra metric callables {name: fn(preds, targets)}.

    Returns:
        Dict with 'loss', 'acc' (where applicable), and any extra metrics.
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0
    extra_totals: Dict[str, float] = {k: 0.0 for k in (metric_fns or {})}

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss   = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss   = loss_fn(logits, y)
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        preds       = _get_predictions(logits, task)
        total_acc  += _compute_accuracy(preds, y, task)

        for name, fn in (metric_fns or {}).items():
            extra_totals[name] += fn(preds, y)

    n = len(dataloader)
    results = {"loss": total_loss / n, "acc": total_acc / n}
    for name in extra_totals:
        results[name] = extra_totals[name] / n
    return results


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    task: str = "multiclass",
    metric_fns: Optional[Dict[str, Callable]] = None,
) -> Dict[str, float]:
    """Tests a PyTorch model for a single epoch.

    Args:
        model: Model to evaluate.
        dataloader: Testing / validation DataLoader.
        loss_fn: Loss function.
        device: Target device.
        task: One of 'multiclass', 'binary', 'multilabel', 'regression'.
        metric_fns: Optional extra metric callables.

    Returns:
        Dict with 'loss', 'acc' (where applicable), and any extra metrics.
    """
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    extra_totals: Dict[str, float] = {k: 0.0 for k in (metric_fns or {})}

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = loss_fn(logits, y)

            total_loss += loss.item()
            preds       = _get_predictions(logits, task)
            total_acc  += _compute_accuracy(preds, y, task)

            for name, fn in (metric_fns or {}).items():
                extra_totals[name] += fn(preds, y)

    n = len(dataloader)
    results = {"loss": total_loss / n, "acc": total_acc / n}
    for name in extra_totals:
        results[name] = extra_totals[name] / n
    return results

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device,
    task: str = "multiclass",
    val_dataloader: Optional[torch.utils.data.DataLoader] = None,
    scheduler: Optional[object] = None,
    early_stopping_patience: Optional[int] = None,
    use_amp: bool = False,
    max_grad_norm: Optional[float] = None,
    metric_fns: Optional[Dict[str, Callable]] = None,
    verbose: bool = True,
) -> Dict[str, List]:
    """Trains and evaluates a PyTorch model.

    Args:
        model: Model to train.
        train_dataloader: Training DataLoader.
        test_dataloader: Test DataLoader (used as validation if val_dataloader is None).
        optimizer: Optimizer.
        loss_fn: Loss function.
        epochs: Total epochs.
        device: Target device string or torch.device.
        task: 'multiclass' | 'binary' | 'multilabel' | 'regression'.
        val_dataloader: Separate validation DataLoader (optional).
        scheduler: LR scheduler (optional). Called with scheduler.step() per epoch,
                   or scheduler.step(val_loss) for ReduceLROnPlateau.
        early_stopping_patience: Stop if val loss doesn't improve for this many epochs.
                                  None = disabled.
        use_amp: Enable automatic mixed precision (CUDA only).
        max_grad_norm: Gradient clipping max norm (None = disabled).
        metric_fns: Extra metrics dict {name: fn(preds, targets) -> float}.
        verbose: Print epoch results.

    Returns:
        Results dict: {train_loss, train_acc, val_loss, val_acc, lr, ...extras}
    """
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

    results: Dict[str, List] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "lr": [],
    }
    for k in (metric_fns or {}):
        results[f"train_{k}"] = []
        results[f"val_{k}"]   = []

    best_val_loss = float("inf")
    patience_counter = 0
    eval_loader = val_dataloader if val_dataloader is not None else test_dataloader

    for epoch in tqdm(range(epochs), desc="Training"):
        train_results = train_step(model, train_dataloader, loss_fn, optimizer, device,
                                   task=task, scaler=scaler, max_grad_norm=max_grad_norm,
                                   metric_fns=metric_fns)
        val_results   = test_step(model, eval_loader, loss_fn, device,
                                  task=task, metric_fns=metric_fns)

        # LR scheduling
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_results["loss"])
            else:
                scheduler.step()

        # Store results
        results["train_loss"].append(train_results["loss"])
        results["train_acc"].append(train_results["acc"])
        results["val_loss"].append(val_results["loss"])
        results["val_acc"].append(val_results["acc"])
        results["lr"].append(current_lr)
        for k in (metric_fns or {}):
            results[f"train_{k}"].append(train_results[k])
            results[f"val_{k}"].append(val_results[k])

        if verbose:
            extras = " | ".join(
                f"train_{k}: {train_results[k]:.4f} | val_{k}: {val_results[k]:.4f}"
                for k in (metric_fns or {})
            )
            print(
                f"Epoch: {epoch+1}/{epochs} | "
                f"train_loss: {train_results['loss']:.4f} | "
                f"train_acc: {train_results['acc']:.4f} | "
                f"val_loss: {val_results['loss']:.4f} | "
                f"val_acc: {val_results['acc']:.4f} | "
                f"lr: {current_lr:.2e}"
                + (f" | {extras}" if extras else "")
            )

        # Early stopping
        if early_stopping_patience is not None:
            if val_results["loss"] < best_val_loss:
                best_val_loss    = val_results["loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n[Early Stopping] No improvement for {early_stopping_patience} epochs. "
                          f"Stopping at epoch {epoch+1}.")
                    break

    return results

def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    task: str = "multiclass",
    metric_fns: Optional[Dict[str, Callable]] = None,
) -> Dict[str, float]:
    """One-shot evaluation on a DataLoader. Returns metrics dict."""
    return test_step(model, dataloader, loss_fn, device, task=task, metric_fns=metric_fns)

def get_all_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    task: str = "multiclass",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs model over full DataLoader and returns (all_preds, all_targets).

    Useful for computing confusion matrices, per-class metrics, etc.
    """
    model.eval()
    all_preds, all_targets = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device)
            logits = model(X)
            preds  = _get_predictions(logits, task).cpu()
            all_preds.append(preds)
            all_targets.append(y)

    return torch.cat(all_preds), torch.cat(all_targets)
