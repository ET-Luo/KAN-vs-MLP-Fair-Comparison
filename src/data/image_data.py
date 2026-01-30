from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class ImageDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    input_dim: int
    num_classes: int


def _mnist_transforms() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])


def _cifar_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )


def get_mnist_loaders(
    data_root: Path,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int | None = None,
) -> ImageDataLoaders:
    transform = _mnist_transforms()
    train_full = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    test = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train, val = random_split(train_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return ImageDataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        input_dim=28 * 28,
        num_classes=10,
    )


def get_cifar10_loaders(
    data_root: Path,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int | None = None,
) -> ImageDataLoaders:
    transform = _cifar_transforms()
    train_full = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train, val = random_split(train_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return ImageDataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        input_dim=32 * 32 * 3,
        num_classes=10,
    )


def get_cifar100_loaders(
    data_root: Path,
    batch_size: int,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int | None = None,
) -> ImageDataLoaders:
    transform = _cifar_transforms()
    train_full = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=transform)
    test = datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=transform)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    train, val = random_split(train_full, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return ImageDataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        input_dim=32 * 32 * 3,
        num_classes=100,
    )
