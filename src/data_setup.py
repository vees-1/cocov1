import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from typing import Callable, Dict, List, Optional, Tuple, Union

NUM_WORKERS = os.cpu_count()

def get_device() -> str:
    """Returns the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pin_memory_for(device: str) -> bool:
    return device == "cuda"

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
    val_dir: Optional[str] = None,
    val_split: float = 0.0,
) -> Tuple:
    """Creates training, (optional) validation, and testing DataLoaders from
    image folders.

    Args:
        train_dir: Path to training directory (ImageFolder structure).
        test_dir:  Path to testing directory.
        transform: torchvision transforms applied to every split.
        batch_size: Samples per batch.
        num_workers: Workers per DataLoader.
        val_dir: Optional explicit validation directory.
        val_split: If > 0 and val_dir is None, fraction of train set used for
                   validation (e.g. 0.2 = 20 %).

    Returns:
        If validation is used → (train_dl, val_dl, test_dl, class_names)
        Otherwise            → (train_dl, test_dl, class_names)
    """
    device = get_device()
    pin = _pin_memory_for(device)

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data  = datasets.ImageFolder(test_dir,  transform=transform)
    class_names = train_data.classes

    # Optional validation split
    if val_dir is not None:
        val_data = datasets.ImageFolder(val_dir, transform=transform)
        val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    elif val_split > 0.0:
        n_val   = int(len(train_data) * val_split)
        n_train = len(train_data) - n_val
        train_data, val_data = random_split(train_data, [n_train, n_val])
        val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    else:
        val_dl = None

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    test_dl  = DataLoader(test_data,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)

    if val_dl is not None:
        return train_dl, val_dl, test_dl, class_names
    return train_dl, test_dl, class_names

def create_balanced_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Same as create_dataloaders but uses WeightedRandomSampler so each
    class is seen equally often during training — useful for imbalanced datasets."""
    device = get_device()
    pin    = _pin_memory_for(device)

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data  = datasets.ImageFolder(test_dir,  transform=transform)
    class_names = train_data.classes

    targets      = torch.tensor(train_data.targets)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_dl = DataLoader(train_data, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=pin)
    test_dl  = DataLoader(test_data,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    return train_dl, test_dl, class_names

class CSVDataset(Dataset):
    """Generic dataset for CSV files.

    Args:
        csv_path: Path to CSV file.
        feature_cols: Column names (or indices) to use as features.
                      If None, all columns except label_col are used.
        label_col: Column name/index for the target label.
        transform: Optional callable applied to the feature tensor.
        target_transform: Optional callable applied to the label tensor.
        dtype: torch dtype for features (default float32).
    """
    def __init__(
        self,
        csv_path: str,
        feature_cols: Optional[List[Union[str, int]]] = None,
        label_col: Union[str, int] = -1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
    ):
        df = pd.read_csv(csv_path)
        if feature_cols is None:
            label_name = df.columns[label_col] if isinstance(label_col, int) else label_col
            feature_cols = [c for c in df.columns if c != label_name]
        label_name = df.columns[label_col] if isinstance(label_col, int) else label_col

        self.X = torch.tensor(df[feature_cols].values, dtype=dtype)
        self.y = torch.tensor(df[label_name].values)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

def create_csv_dataloaders(
    csv_path: str,
    feature_cols: Optional[List[Union[str, int]]] = None,
    label_col: Union[str, int] = -1,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates train / val / test DataLoaders from a single CSV file."""
    dataset = CSVDataset(csv_path, feature_cols, label_col, transform, target_transform)
    n       = len(dataset)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl

class ImageCSVDataset(Dataset):
    """Image dataset defined by a CSV with image paths and labels.

    CSV must have at least two columns: image path and label.

    Args:
        csv_path: Path to the CSV manifest.
        img_col: Column name for image file paths.
        label_col: Column name for labels.
        root_dir: Optional root directory prepended to relative image paths.
        transform: torchvision transform applied to each image.
        label_map: Optional dict mapping raw label values → int indices.
    """
    def __init__(
        self,
        csv_path: str,
        img_col: str = "image",
        label_col: str = "label",
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        label_map: Optional[Dict] = None,
    ):
        self.df        = pd.read_csv(csv_path)
        self.img_col   = img_col
        self.label_col = label_col
        self.root_dir  = Path(root_dir) if root_dir else Path(".")
        self.transform = transform

        # Build label map automatically if not provided
        if label_map is None:
            unique = sorted(self.df[label_col].unique())
            label_map = {v: i for i, v in enumerate(unique)}
        self.label_map   = label_map
        self.class_names = list(label_map.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        img_path  = self.root_dir / row[self.img_col]
        image     = Image.open(img_path).convert("RGB")
        label     = self.label_map[row[self.label_col]]
        if self.transform:
            image = self.transform(image)
        return image, label

class MultiLabelImageDataset(Dataset):
    """Image dataset for multi-label classification.

    CSV must have one image-path column and one column per class
    containing 0/1 values.

    Args:
        csv_path: Path to CSV.
        img_col: Column name for image paths.
        label_cols: List of column names that are binary labels.
        root_dir: Root directory for image paths.
        transform: torchvision transform.
    """
    def __init__(
        self,
        csv_path: str,
        img_col: str,
        label_cols: List[str],
        root_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.df         = pd.read_csv(csv_path)
        self.img_col    = img_col
        self.label_cols = label_cols
        self.root_dir   = Path(root_dir) if root_dir else Path(".")
        self.transform  = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        image    = Image.open(self.root_dir / row[self.img_col]).convert("RGB")
        labels   = torch.tensor(row[self.label_cols].values.astype(float), dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels

class TensorDatasetWrapper(Dataset):
    """Wraps pre-loaded numpy arrays or torch tensors into a Dataset.

    Args:
        X: Features as ndarray or Tensor.
        y: Labels as ndarray or Tensor.
        transform: Optional callable applied to each sample.
    """
    def __init__(self, X, y, transform: Optional[Callable] = None):
        self.X = torch.tensor(X, dtype=torch.float32) if isinstance(X, np.ndarray) else X
        self.y = torch.tensor(y) if isinstance(y, np.ndarray) else y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]


def create_tensor_dataloaders(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    transform: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Splits arrays into train/val/test and returns DataLoaders."""
    dataset = TensorDatasetWrapper(X, y, transform)
    n       = len(dataset)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, val_dl, test_dl

class TimeSeriesDataset(Dataset):
    """Sliding-window time-series dataset.

    Args:
        data: 2-D array of shape (timesteps, features) or 1-D (timesteps,).
        seq_len: Length of the input window.
        pred_len: Number of future steps to predict (default 1).
        target_col: Column index to use as the prediction target (default -1).
        stride: Step size between windows (default 1).
    """
    def __init__(
        self,
        data: Union[np.ndarray, torch.Tensor],
        seq_len: int,
        pred_len: int = 1,
        target_col: int = -1,
        stride: int = 1,
    ):
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        self.data       = data
        self.seq_len    = seq_len
        self.pred_len   = pred_len
        self.target_col = target_col
        self.stride     = stride
        self.indices    = list(range(0, len(data) - seq_len - pred_len + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start   = self.indices[idx]
        x       = self.data[start : start + self.seq_len]
        y_block = self.data[start + self.seq_len : start + self.seq_len + self.pred_len]
        y = y_block[:, self.target_col] if self.data.dim() > 1 else y_block
        return x, y
