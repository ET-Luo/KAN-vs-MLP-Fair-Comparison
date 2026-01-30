from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from src.models.kan import KANWrapper
from src.models.mlp import MLP


@dataclass(frozen=True)
class KANSearchConfig:
    depth_range: Iterable[int]
    width_range: Iterable[int]
    grid_range: Iterable[int]


@dataclass(frozen=True)
class MLPDepthWidthConfig:
    depth_range: Iterable[int]
    width_range: Iterable[int]


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_kan_params(width: int, depth: int, grid_size: int) -> int:
    return width * width * depth + width * depth * grid_size


def find_best_kan_config(
    input_dim: int,
    output_dim: int,
    param_budget: int,
    search: KANSearchConfig,
    device: str,
) -> tuple[list[int], int, int]:
    best_diff = float("inf")
    best_width = None
    best_depth = None
    best_grid = None

    for depth in search.depth_range:
        for width in search.width_range:
            for grid in search.grid_range:
                approx = estimate_kan_params(width, depth, grid)
                diff = abs(approx - param_budget)
                if diff < best_diff:
                    best_diff = diff
                    best_width = width
                    best_depth = depth
                    best_grid = grid

    if best_width is None or best_depth is None or best_grid is None:
        raise ValueError("No KAN config found.")

    # Validate with actual model parameter count
    width_list = [input_dim] + [best_width] * best_depth + [output_dim]
    model = KANWrapper(width=width_list, grid_size=best_grid).to(device)
    actual = count_parameters(model)

    # Return width list and grid size + actual params
    return width_list, best_grid, actual


def find_best_mlp_config(
    input_dim: int,
    output_dim: int,
    param_budget: int,
    search: MLPDepthWidthConfig,
) -> tuple[list[int], int]:
    best_diff = float("inf")
    best_hidden = None

    for depth in search.depth_range:
        for width in search.width_range:
            hidden_dims = [width] * depth
            model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims)
            actual = count_parameters(model)
            diff = abs(actual - param_budget)
            if diff < best_diff:
                best_diff = diff
                best_hidden = hidden_dims

    if best_hidden is None:
        raise ValueError("No MLP config found.")

    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=best_hidden)
    actual = count_parameters(model)
    return best_hidden, actual
