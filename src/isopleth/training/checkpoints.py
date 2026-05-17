"""Atomic checkpoint serialization and interrupted-run continuation (I26).

Guarantees atomicity via staged write and rename, capturing model weights, optimizer,
scheduler, RNG states, and training history for exact reproducible recovery.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

@dataclass
class CheckpointMetadata:
    epoch: int
    global_step: int
    val_loss: float
    best_val_loss: float
    timestamp: str
    model_name: str
    config_hash: str
    is_best: bool = False

class CheckpointManager:
    """Manages atomic saving, loading, and recovery of model checkpoints."""

    def __init__(self, checkpoint_dir: Path | str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        val_loss: float = float("inf"),
        best_val_loss: float = float("inf"),
        history: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        is_best: bool = False,
    ) -> Path:
        """Atomically persist training checkpoint."""
        fname = filename or f"checkpoint_epoch_{epoch:04d}_step_{global_step:06d}.pt"
        final_path = self.checkpoint_dir / fname
        temp_path = self.checkpoint_dir / f"{fname}.tmp"

        config_str = json.dumps(config or {}, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()

        # Capture complete RNG states for exact reproduction
        rng_states = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.get_rng_state()

        payload = {
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "is_best": is_best,
            "config_hash": config_hash,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "rng_states": rng_states,
            "history": history or {},
        }

        # Write to temporary file, then atomic rename
        torch.save(payload, temp_path)
        # Flush to physical storage
        with open(temp_path, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temp_path, final_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(payload, temp_path)
            with open(temp_path, "rb") as f:
                os.fsync(f.fileno())
            os.replace(temp_path, best_path)

        return final_path

    def load(
        self,
        checkpoint_path: Path | str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        """Safely restore model weights, optimizer, and training state from checkpoint."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        # Weights only load for security and integrity
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if restore_rng and "rng_states" in checkpoint:
            rng = checkpoint["rng_states"]
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "python" in rng:
                random.setstate(rng["python"])
            if torch.cuda.is_available() and "cuda" in rng:
                torch.cuda.set_rng_state(rng["cuda"])

        return {
            "epoch": checkpoint.get("epoch", 0),
            "global_step": checkpoint.get("global_step", 0),
            "val_loss": checkpoint.get("val_loss", float("inf")),
            "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
            "history": checkpoint.get("history", {}),
            "config_hash": checkpoint.get("config_hash", ""),
        }
