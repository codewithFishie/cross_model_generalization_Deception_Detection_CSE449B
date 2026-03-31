from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

LABEL_MAP = {"truth": 0, "lie": 1}


@dataclass
class DatasetConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_csv: str = "data/splits/train.csv"
    val_csv: str = "data/splits/val.csv"
    test_csv: str = "data/splits/test.csv"


class DeceptionImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: transforms.Compose | None = None,
        return_domain: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.return_domain = return_domain
        self.domain_to_idx = {
            d: i for i, d in enumerate(sorted(self.df["domain"].unique().tolist()))
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row["path"])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["label"])

        if self.return_domain:
            domain = self.domain_to_idx[row["domain"]]
            return image, torch.tensor(label), torch.tensor(domain)
        return image, torch.tensor(label)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfm = transforms.Compose(
        [
            transforms.Resize((image_size + 16, image_size + 16)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfm, eval_tfm


def _weighted_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    label_counts: Dict[int, int] = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    weights = [1.0 / label_counts[l] for l in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def load_splits(cfg: DatasetConfig) -> Dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(cfg.train_csv),
        "val": pd.read_csv(cfg.val_csv),
        "test": pd.read_csv(cfg.test_csv),
    }


def make_dataloaders(
    cfg: DatasetConfig,
    domain_adversarial: bool = False,
) -> Dict[str, DataLoader]:
    splits = load_splits(cfg)
    train_tfm, eval_tfm = build_transforms(cfg.image_size)

    train_ds = DeceptionImageDataset(
        splits["train"], transform=train_tfm, return_domain=domain_adversarial
    )
    val_ds = DeceptionImageDataset(
        splits["val"], transform=eval_tfm, return_domain=domain_adversarial
    )
    test_ds = DeceptionImageDataset(
        splits["test"], transform=eval_tfm, return_domain=domain_adversarial
    )

    sampler = _weighted_sampler(splits["train"]["label"].tolist())

    return {
        "train": DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
    }


def build_manifest(root: str, output_csv: str, domain_name: str | None = None) -> None:
    root_path = Path(root)
    rows: List[Dict[str, str | int]] = []
    domain = domain_name or root_path.name
    for cls_name, label in LABEL_MAP.items():
        class_dir = root_path / cls_name
        if not class_dir.exists():
            continue
        for p in class_dir.glob("**/*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                rows.append(
                    {
                        "path": str(p.resolve()),
                        "label": label,
                        "label_name": cls_name,
                        "domain": domain,
                    }
                )

    if not rows:
        raise ValueError(f"No images found in {root}")

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def split_manifest(
    input_csv: str,
    out_dir: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    random.seed(seed)
    df = pd.read_csv(input_csv)

    train_rows, val_rows, test_rows = [], [], []
    for (domain, label), g in df.groupby(["domain", "label"]):
        idxs = g.index.tolist()
        random.shuffle(idxs)
        n = len(idxs)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        val_idx = idxs[:n_val]
        test_idx = idxs[n_val : n_val + n_test]
        train_idx = idxs[n_val + n_test :]

        train_rows.append(df.loc[train_idx])
        val_rows.append(df.loc[val_idx])
        test_rows.append(df.loc[test_idx])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pd.concat(train_rows).to_csv(out_path / "train.csv", index=False)
    pd.concat(val_rows).to_csv(out_path / "val.csv", index=False)
    pd.concat(test_rows).to_csv(out_path / "test.csv", index=False)
