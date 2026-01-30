from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentResult:
    experiment_id: str
    dataset: str
    model: str
    param_budget: int
    actual_params: int
    seed: int
    test_accuracy: float
    best_val_accuracy: float
    epochs_trained: int
    training_time_hours: float
    inference_time_ms: float
    peak_memory_gb: float
    notes: str


def append_result(csv_path: Path, result: ExperimentResult) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(result).keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(result))
