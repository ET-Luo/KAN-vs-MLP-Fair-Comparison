#!/usr/bin/env python3
"""
Download datasets into ./data/raw using standard APIs.

Note: Ensure the conda environment is activated before running:
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate model
"""

from __future__ import annotations

from pathlib import Path

from sklearn.datasets import fetch_20newsgroups
from torchtext.datasets import AG_NEWS
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def is_non_empty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def download_mnist() -> str:
    target = RAW_DIR / "MNIST"
    if is_non_empty_dir(target):
        return "MNIST: skipped (already exists)"
    print("[INFO] Downloading MNIST...")
    datasets.MNIST(root=str(RAW_DIR), train=True, download=True)
    datasets.MNIST(root=str(RAW_DIR), train=False, download=True)
    return "MNIST: downloaded"


def download_cifar10() -> str:
    target = RAW_DIR / "CIFAR10"
    if is_non_empty_dir(target):
        return "CIFAR10: skipped (already exists)"
    print("[INFO] Downloading CIFAR10...")
    datasets.CIFAR10(root=str(RAW_DIR), train=True, download=True)
    datasets.CIFAR10(root=str(RAW_DIR), train=False, download=True)
    return "CIFAR10: downloaded"


def download_cifar100() -> str:
    target = RAW_DIR / "CIFAR100"
    if is_non_empty_dir(target):
        return "CIFAR100: skipped (already exists)"
    print("[INFO] Downloading CIFAR100...")
    datasets.CIFAR100(root=str(RAW_DIR), train=True, download=True)
    datasets.CIFAR100(root=str(RAW_DIR), train=False, download=True)
    return "CIFAR100: downloaded"


def download_ag_news() -> str:
    target = RAW_DIR / "AG_NEWS"
    if is_non_empty_dir(target):
        return "AG_NEWS: skipped (already exists)"
    print("[INFO] Downloading AG_NEWS...")
    # AG_NEWS downloads into the specified root
    _ = list(AG_NEWS(root=str(RAW_DIR)))
    return "AG_NEWS: downloaded"


def download_20newsgroups() -> str:
    target = RAW_DIR / "20newsgroups"
    if is_non_empty_dir(target):
        return "20 Newsgroups: skipped (already exists)"
    print("[INFO] Downloading 20 Newsgroups (train/test)...")
    fetch_20newsgroups(subset="train", data_home=str(target), download_if_missing=True)
    fetch_20newsgroups(subset="test", data_home=str(target), download_if_missing=True)
    return "20 Newsgroups: downloaded"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results: list[str] = []
    results.append(download_mnist())
    results.append(download_cifar10())
    results.append(download_cifar100())
    results.append(download_ag_news())
    results.append(download_20newsgroups())

    print("\n[INFO] Download summary:")
    for item in results:
        print(f"- {item}")


if __name__ == "__main__":
    main()
