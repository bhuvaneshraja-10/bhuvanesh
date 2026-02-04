"""Clinical Risk Prediction with Longitudinal EHR.

Goal: Predict 30/90-day readmission risk using sparse, irregular clinical records.

This module provides:
- Time-aware Transformer model with missing-not-at-random (MNAR) handling.
- Uncertainty calibration via temperature scaling.
- Fairness constraint utilities (equalized odds / demographic parity regularizers).
- Example dataset + collate for sparse, irregular sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import math
import random

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Batch:
    """Container for a batch of irregular sequences.

    Attributes:
        values: Tensor[batch, seq_len, num_features]
        mask: Tensor[batch, seq_len, num_features] where 1 indicates observed.
        times: Tensor[batch, seq_len] absolute timestamps (float hours).
        static: Tensor[batch, num_static_features]
        labels_30: Tensor[batch] binary label for 30-day readmission.
        labels_90: Tensor[batch] binary label for 90-day readmission.
        sensitive: Optional Tensor[batch] sensitive attribute (e.g., race/sex).
    """

    values: torch.Tensor
    mask: torch.Tensor
    times: torch.Tensor
    static: torch.Tensor
    labels_30: torch.Tensor
    labels_90: torch.Tensor
    sensitive: Optional[torch.Tensor] = None


class MissingnessEncoder(nn.Module):
    """Encode MNAR patterns using value and mask embeddings.

    We embed both observed values and missingness indicators. Mask embeddings
    allow the model to learn patterns of missingness that carry predictive
    signal (MNAR).
    """

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.value_proj = nn.Linear(num_features, embed_dim)
        self.mask_proj = nn.Linear(num_features, embed_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value_emb = self.value_proj(values)
        mask_emb = self.mask_proj(mask)
        return value_emb + mask_emb


class TimeEncoding(nn.Module):
    """Sinusoidal time encoding for irregular intervals."""

    def __init__(self, embed_dim: int, max_timescale: float = 1e4):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_timescale = max_timescale

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        # times: [batch, seq_len]
        device = times.device
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=device, dtype=times.dtype)
            * -(math.log(self.max_timescale) / (half_dim - 1))
        )
        angles = times.unsqueeze(-1) * freqs
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.embed_dim % 2 == 1:
            enc = F.pad(enc, (0, 1))
        return enc


class TimeAwareAttention(nn.Module):
    """Multi-head attention with time decay bias."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.time_gate = nn.Linear(1, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> torch.Tensor:
        # delta_t: [batch, seq_len, seq_len]
        gate = torch.sigmoid(self.time_gate(delta_t.unsqueeze(-1)))
        # Expand to match attention heads
        gate = gate.permute(0, 3, 1, 2)  # [batch, heads, seq, seq]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=~attn_mask)
        return self.dropout(attn_output) * gate.mean(dim=1)


class TransformerBlock(nn.Module):
    """Transformer block with time-aware attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.time_attn = TimeAwareAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.time_attn(self.norm1(x), attn_mask, delta_t))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ClinicalRiskTransformer(nn.Module):
    """Time-aware Transformer for readmission risk prediction."""

    def __init__(
        self,
        num_features: int,
        num_static_features: int,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = MissingnessEncoder(num_features, embed_dim)
        self.time_embed = TimeEncoding(embed_dim)
        self.static_proj = nn.Linear(num_static_features, embed_dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.head_30 = nn.Linear(embed_dim, 1)
        self.head_90 = nn.Linear(embed_dim, 1)

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        times: torch.Tensor,
        static: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # values/mask: [batch, seq_len, features]
        seq_emb = self.embed(values, mask) + self.time_embed(times)
        static_emb = self.static_proj(static).unsqueeze(1)
        x = torch.cat([static_emb, seq_emb], dim=1)

        # Build attention mask: all tokens present including static token
        batch_size, seq_len, _ = x.shape
        attn_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.bool)

        # Delta times between tokens (static token time = 0)
        padded_times = torch.cat([torch.zeros(batch_size, 1, device=times.device), times], dim=1)
        delta_t = padded_times.unsqueeze(2) - padded_times.unsqueeze(1)
        delta_t = delta_t.abs()

        for layer in self.layers:
            x = layer(x, attn_mask, delta_t)

        pooled = x[:, 0]  # use static token
        logits_30 = self.head_30(pooled).squeeze(-1)
        logits_90 = self.head_90(pooled).squeeze(-1)
        return logits_30, logits_90


class TemperatureScaler(nn.Module):
    """Temperature scaling for calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)


def fairness_regularizer(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sensitive: torch.Tensor,
    mode: str = "equalized_odds",
) -> torch.Tensor:
    """Compute fairness regularization term.

    Args:
        logits: Raw model outputs.
        labels: Binary labels.
        sensitive: Sensitive group identifiers (0/1).
        mode: equalized_odds or demographic_parity.
    """

    probs = torch.sigmoid(logits)
    if mode == "demographic_parity":
        mean0 = probs[sensitive == 0].mean()
        mean1 = probs[sensitive == 1].mean()
        return (mean0 - mean1).abs()
    if mode == "equalized_odds":
        reg = torch.tensor(0.0, device=logits.device)
        for label in [0, 1]:
            mask = labels == label
            if mask.any():
                mean0 = probs[(sensitive == 0) & mask].mean()
                mean1 = probs[(sensitive == 1) & mask].mean()
                reg = reg + (mean0 - mean1).abs()
        return reg
    raise ValueError(f"Unknown fairness mode: {mode}")


def compute_loss(
    logits_30: torch.Tensor,
    logits_90: torch.Tensor,
    labels_30: torch.Tensor,
    labels_90: torch.Tensor,
    sensitive: Optional[torch.Tensor] = None,
    fairness_weight: float = 0.0,
) -> torch.Tensor:
    loss_30 = F.binary_cross_entropy_with_logits(logits_30, labels_30.float())
    loss_90 = F.binary_cross_entropy_with_logits(logits_90, labels_90.float())
    loss = loss_30 + loss_90
    if sensitive is not None and fairness_weight > 0.0:
        loss = loss + fairness_weight * fairness_regularizer(logits_90, labels_90, sensitive)
    return loss


class SyntheticEHRDataset(torch.utils.data.Dataset):
    """Synthetic dataset stub for sparse, irregular sequences."""

    def __init__(self, num_samples: int, num_features: int, num_static: int):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_static = num_static

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_len = random.randint(5, 30)
        times = torch.sort(torch.rand(seq_len) * 72.0).values
        values = torch.randn(seq_len, self.num_features)
        mask = (torch.rand(seq_len, self.num_features) > 0.2).float()
        values = values * mask
        static = torch.randn(self.num_static)
        labels_30 = torch.tensor(random.random() > 0.7)
        labels_90 = torch.tensor(random.random() > 0.6)
        sensitive = torch.tensor(random.randint(0, 1))
        return {
            "values": values,
            "mask": mask,
            "times": times,
            "static": static,
            "labels_30": labels_30,
            "labels_90": labels_90,
            "sensitive": sensitive,
        }


def collate_irregular(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    max_len = max(item["values"].shape[0] for item in batch)
    num_features = batch[0]["values"].shape[1]
    batch_size = len(batch)

    values = torch.zeros(batch_size, max_len, num_features)
    mask = torch.zeros(batch_size, max_len, num_features)
    times = torch.zeros(batch_size, max_len)
    static = torch.stack([item["static"] for item in batch])
    labels_30 = torch.stack([item["labels_30"] for item in batch])
    labels_90 = torch.stack([item["labels_90"] for item in batch])
    sensitive = torch.stack([item["sensitive"] for item in batch])

    for i, item in enumerate(batch):
        seq_len = item["values"].shape[0]
        values[i, :seq_len] = item["values"]
        mask[i, :seq_len] = item["mask"]
        times[i, :seq_len] = item["times"]

    return Batch(values, mask, times, static, labels_30, labels_90, sensitive)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fairness_weight: float = 0.1,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = Batch(
            values=batch.values.to(device),
            mask=batch.mask.to(device),
            times=batch.times.to(device),
            static=batch.static.to(device),
            labels_30=batch.labels_30.to(device),
            labels_90=batch.labels_90.to(device),
            sensitive=batch.sensitive.to(device),
        )
        optimizer.zero_grad()
        logits_30, logits_90 = model(batch.values, batch.mask, batch.times, batch.static)
        loss = compute_loss(
            logits_30,
            logits_90,
            batch.labels_30,
            batch.labels_90,
            batch.sensitive,
            fairness_weight=fairness_weight,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticEHRDataset(num_samples=200, num_features=16, num_static=8)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_irregular)

    model = ClinicalRiskTransformer(num_features=16, num_static_features=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(3):
        loss = train_epoch(model, loader, optimizer, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}")


if __name__ == "__main__":
    main()
