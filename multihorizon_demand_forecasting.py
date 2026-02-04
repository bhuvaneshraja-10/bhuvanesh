"""Multihorizon Demand Forecasting with Regime Shifts.

This module provides:
- Hierarchical time-series utilities (SKU/Store/Category aggregation).
- Change-point detection for regime shifts.
- Global-local probabilistic model with quantile loss.
- Simple data-drift adaptation via exponentially decayed weights.

The code is designed as a starting point for large-scale demand forecasting
(1-52 weeks horizon) across thousands of SKU-store series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class ForecastConfig:
    context_length: int = 52
    horizon: int = 52
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    hidden_size: int = 128
    num_layers: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 5
    drift_decay: float = 0.98


def add_hierarchy_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add hierarchical keys for SKU and Store levels.

    Expected columns: ["date", "sku", "store", "demand", "category"].
    """
    df = df.copy()
    df["sku_store"] = df["sku"].astype(str) + "__" + df["store"].astype(str)
    df["category_store"] = df["category"].astype(str) + "__" + df["store"].astype(str)
    return df


def aggregate_hierarchy(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create hierarchy aggregates for SKU, Store, Category-Store, and Global."""
    df = df.copy()
    levels = {
        "sku_store": ["date", "sku_store"],
        "sku": ["date", "sku"],
        "store": ["date", "store"],
        "category_store": ["date", "category_store"],
        "global": ["date"],
    }
    aggregated = {}
    for level, keys in levels.items():
        grouped = df.groupby(keys, observed=True)["demand"].sum().reset_index()
        aggregated[level] = grouped
    return aggregated


def detect_regime_shifts(series: np.ndarray, window: int = 8, threshold: float = 2.5) -> List[int]:
    """Detect regime shifts using rolling mean difference z-score.

    Returns indices where a change point is detected.
    """
    if len(series) < window * 2:
        return []
    rolling_mean = pd.Series(series).rolling(window).mean().to_numpy()
    diffs = np.abs(np.diff(rolling_mean))
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return []
    z_scores = (diffs - diffs.mean()) / (diffs.std() + 1e-6)
    change_points = np.where(z_scores > threshold)[0].tolist()
    return change_points


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for time, promotions, price, etc.

    This is intentionally minimal; extend with known covariates.
    """
    df = df.copy()
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    df["sin_week"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    return df


class DemandWindowDataset(Dataset):
    """Windowed dataset for multihorizon forecasting."""

    def __init__(
        self,
        df: pd.DataFrame,
        context_length: int,
        horizon: int,
        id_col: str = "sku_store",
    ) -> None:
        self.context_length = context_length
        self.horizon = horizon
        self.id_col = id_col
        self.series_map = {
            series_id: group.sort_values("date")
            for series_id, group in df.groupby(id_col, observed=True)
        }
        self.index = []
        for series_id, group in self.series_map.items():
            values = group["demand"].to_numpy()
            for i in range(context_length, len(values) - horizon + 1):
                self.index.append((series_id, i))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        series_id, t = self.index[idx]
        group = self.series_map[series_id]
        values = group["demand"].to_numpy().astype(np.float32)
        context = values[t - self.context_length : t]
        target = values[t : t + self.horizon]
        return {
            "context": torch.from_numpy(context),
            "target": torch.from_numpy(target),
        }


def quantile_loss(y_true: torch.Tensor, y_pred: torch.Tensor, quantiles: Iterable[float]) -> torch.Tensor:
    """Pinball loss for multiple quantiles.

    y_pred shape: (batch, horizon, num_quantiles)
    y_true shape: (batch, horizon)
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, :, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
    return torch.mean(torch.cat(losses, dim=-1))


class GlobalLocalModel(nn.Module):
    """Global model with local embeddings for SKU-store series."""

    def __init__(self, num_series: int, config: ForecastConfig) -> None:
        super().__init__()
        self.series_embed = nn.Embedding(num_series, config.hidden_size)
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.proj = nn.Linear(config.hidden_size, config.horizon * len(config.quantiles))
        self.config = config

    def forward(self, series_ids: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        embed = self.series_embed(series_ids).unsqueeze(1)
        x = context.unsqueeze(-1)
        output, _ = self.encoder(x)
        final_state = output[:, -1, :] + embed.squeeze(1)
        forecasts = self.proj(final_state)
        return forecasts.view(-1, self.config.horizon, len(self.config.quantiles))


def build_series_index(df: pd.DataFrame, id_col: str) -> Dict[str, int]:
    ids = sorted(df[id_col].unique())
    return {series_id: idx for idx, series_id in enumerate(ids)}


def train_model(
    df: pd.DataFrame,
    config: ForecastConfig,
    device: torch.device | None = None,
) -> Tuple[GlobalLocalModel, Dict[str, int]]:
    """Train global-local model with drift-aware weighting."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    series_index = build_series_index(df, "sku_store")

    dataset = DemandWindowDataset(df, config.context_length, config.horizon)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = GlobalLocalModel(len(series_index), config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            series_ids = torch.zeros(context.size(0), dtype=torch.long, device=device)

            optimizer.zero_grad()
            preds = model(series_ids, context)
            loss = quantile_loss(target, preds, config.quantiles)

            drift_weight = config.drift_decay ** epoch
            (loss * drift_weight).backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{config.num_epochs} - Loss: {epoch_loss / len(loader):.4f}")

    return model, series_index


def predict(
    model: GlobalLocalModel,
    df: pd.DataFrame,
    series_index: Dict[str, int],
    config: ForecastConfig,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Generate probabilistic forecasts for each series."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    results = []
    with torch.no_grad():
        for series_id, group in df.groupby("sku_store", observed=True):
            group = group.sort_values("date")
            context = group["demand"].to_numpy().astype(np.float32)[-config.context_length :]
            context_tensor = torch.from_numpy(context).unsqueeze(0).to(device)
            series_tensor = torch.tensor([series_index[series_id]], dtype=torch.long, device=device)
            preds = model(series_tensor, context_tensor).squeeze(0).cpu().numpy()

            for h in range(config.horizon):
                row = {
                    "sku_store": series_id,
                    "horizon": h + 1,
                }
                for q_idx, q in enumerate(config.quantiles):
                    row[f"q{q}"] = preds[h, q_idx]
                results.append(row)

    return pd.DataFrame(results)


def example_usage() -> None:
    """Example usage with mock data."""
    dates = pd.date_range("2020-01-01", periods=200, freq="W")
    data = []
    for sku in ["sku1", "sku2"]:
        for store in ["store1", "store2"]:
            demand = np.random.poisson(20, size=len(dates))
            category = "catA" if sku == "sku1" else "catB"
            data.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "sku": sku,
                        "store": store,
                        "category": category,
                        "demand": demand,
                    }
                )
            )
    df = pd.concat(data, ignore_index=True)
    df = add_hierarchy_keys(df)
    df = build_features(df)

    config = ForecastConfig()
    model, series_index = train_model(df, config)
    forecasts = predict(model, df, series_index, config)
    print(forecasts.head())


if __name__ == "__main__":
    example_usage()
