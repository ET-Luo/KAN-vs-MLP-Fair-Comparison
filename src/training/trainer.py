from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainResult:
    best_val_accuracy: float
    test_accuracy: float
    epochs_trained: int
    training_time_hours: float


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total_samples += y.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: ReduceLROnPlateau,
    device: str,
    max_epochs: int,
    early_stopping_patience: int,
) -> TrainResult:
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * y.size(0)
            epoch_correct += (torch.argmax(logits, dim=1) == y).sum().item()
            epoch_samples += y.size(0)

        train_loss = epoch_loss / epoch_samples
        train_acc = epoch_correct / epoch_samples
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc = evaluate(model, test_loader, device)
    end = time.time()

    return TrainResult(
        best_val_accuracy=best_val_acc,
        test_accuracy=test_acc,
        epochs_trained=epoch,
        training_time_hours=(end - start) / 3600.0,
    )
