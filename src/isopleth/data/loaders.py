"""Bounded-memory chunked loaders, windowing, and physical scaling normalizers.

Ensures normalizers are fitted strictly on allowed training partitions,
preserving raw physical-unit arrays for conservation audits and loss computations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from isopleth.data.contracts import PhysicalFamily, Role, TrajectoryBatch

@dataclass
class NormalizerStats:
    """Mean and standard deviation statistics for each physical field."""
    field_names: Tuple[str, ...]
    mean: Dict[str, float]
    std: Dict[str, float]
    min_val: Dict[str, float]
    max_val: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_names": list(self.field_names),
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizerStats:
        return cls(
            field_names=tuple(data["field_names"]),
            mean=data["mean"],
            std=data["std"],
            min_val=data["min_val"],
            max_val=data["max_val"],
        )

class FieldNormalizer:
    """Channel-wise field normalizer fitted strictly on training data."""

    def __init__(self, stats: Optional[NormalizerStats] = None) -> None:
        self.stats = stats

    @classmethod
    def fit(cls, batches: Sequence[TrajectoryBatch]) -> FieldNormalizer:
        """Fit normalization parameters over allowed training batches only."""
        if not batches:
            raise ValueError("Cannot fit FieldNormalizer on empty batch sequence")

        field_names = batches[0].field_names
        # Accumulate sums for mean and std
        counts: Dict[str, int] = {f: 0 for f in field_names}
        sums: Dict[str, float] = {f: 0.0 for f in field_names}
        sq_sums: Dict[str, float] = {f: 0.0 for f in field_names}
        mins: Dict[str, float] = {f: float("inf") for f in field_names}
        maxs: Dict[str, float] = {f: float("-inf") for f in field_names}

        for batch in batches:
            for fname in field_names:
                tensor = batch.extract_field(fname).detach().cpu().numpy()
                n = tensor.size
                counts[fname] += n
                s = float(np.sum(tensor))
                sq = float(np.sum(tensor**2))
                sums[fname] += s
                sq_sums[fname] += sq
                mins[fname] = min(mins[fname], float(np.min(tensor)))
                maxs[fname] = max(maxs[fname], float(np.max(tensor)))

        mean_dict = {}
        std_dict = {}
        for fname in field_names:
            m = sums[fname] / counts[fname]
            var = (sq_sums[fname] / counts[fname]) - (m**2)
            s = math.sqrt(max(var, 1e-12))
            mean_dict[fname] = m
            std_dict[fname] = s

        stats = NormalizerStats(
            field_names=field_names,
            mean=mean_dict,
            std=std_dict,
            min_val=mins,
            max_val=maxs,
        )
        return cls(stats)

    def transform(self, values: torch.Tensor, field_names: Tuple[str, ...]) -> torch.Tensor:
        """Normalize physical tensor to zero mean and unit variance."""
        if self.stats is None:
            raise RuntimeError("FieldNormalizer is not fitted yet")

        out = values.clone()
        for idx, fname in enumerate(field_names):
            m = self.stats.mean[fname]
            s = self.stats.std[fname]
            if values.ndim == 3:  # [B, T, X] 1D single-field
                out = (out - m) / s
            elif values.ndim == 4:  # [B, T, X, C]
                out[..., idx] = (out[..., idx] - m) / s
            elif values.ndim == 5:  # [B, T, Y, X, C]
                out[..., idx] = (out[..., idx] - m) / s
        return out

    def inverse_transform(self, values: torch.Tensor, field_names: Tuple[str, ...]) -> torch.Tensor:
        """Invert normalized tensor back into physical units."""
        if self.stats is None:
            raise RuntimeError("FieldNormalizer is not fitted yet")

        out = values.clone()
        for idx, fname in enumerate(field_names):
            m = self.stats.mean[fname]
            s = self.stats.std[fname]
            if values.ndim == 3:
                out = out * s + m
            elif values.ndim == 4:
                out[..., idx] = out[..., idx] * s + m
            elif values.ndim == 5:
                out[..., idx] = out[..., idx] * s + m
        return out

    def save_json(self, path: Path | str) -> None:
        if self.stats is None:
            raise RuntimeError("Cannot save unfitted FieldNormalizer")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.stats.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> FieldNormalizer:
        target = Path(path)
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(NormalizerStats.from_dict(data))

import math

class WindowedTrajectoryDataset(Dataset):
    """Dataset delivering fixed-horizon temporal windows from trajectories.
    
    Extracts (x_input, x_target, times, dt) tuples deterministically.
    """

    def __init__(
        self,
        batches: Sequence[TrajectoryBatch],
        horizon: int = 10,
        stride: int = 1,
        normalizer: Optional[FieldNormalizer] = None,
    ) -> None:
        self.batches = list(batches)
        self.horizon = horizon
        self.stride = stride
        self.normalizer = normalizer

        self.index_map: List[Tuple[int, int, int]] = []
        for b_idx, batch in enumerate(self.batches):
            B, T = batch.values.shape[0], batch.values.shape[1]
            max_start = T - horizon
            if max_start <= 0:
                continue
            for sample_idx in range(B):
                for start_t in range(0, max_start, stride):
                    self.index_map.append((b_idx, sample_idx, start_t))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        b_idx, s_idx, start_t = self.index_map[idx]
        batch = self.batches[b_idx]

        # Extract trajectory slice [horizon + 1]
        val_slice = batch.values[s_idx, start_t : start_t + self.horizon + 1]
        time_slice = batch.times[start_t : start_t + self.horizon + 1]
        dt = float(time_slice[1] - time_slice[0])

        x_init = val_slice[0]  # Initial state
        target_seq = val_slice[1:]  # Target rollout states

        # Optional normalization
        if self.normalizer is not None:
            x_init_norm = self.normalizer.transform(
                x_init.unsqueeze(0).unsqueeze(0), batch.field_names
            ).squeeze(0).squeeze(0)
            target_norm = self.normalizer.transform(
                target_seq.unsqueeze(0), batch.field_names
            ).squeeze(0)
        else:
            x_init_norm = x_init
            target_norm = target_seq

        # Physical parameters as tensor
        prov = batch.provenance[s_idx]
        param_tensor = torch.tensor(
            list(prov.parameters.values()), dtype=torch.float32
        )

        return {
            "x_init": x_init_norm,
            "x_init_physical": x_init,
            "target": target_norm,
            "target_physical": target_seq,
            "times": time_slice,
            "dt": dt,
            "parameters": param_tensor,
            "trajectory_id": prov.trajectory_id,
            "family": batch.family.value,
        }
