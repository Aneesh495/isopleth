"""Original multi-horizon trainer with validation selection and physical diagnostics (I15).

Trains neural operators using progressive multi-horizon curricula, scheduled sampling,
gradient clipping, and atomic checkpointing. Checkpoint selection is strictly governed
by held-out validation loss.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from isopleth.data.contracts import BoundaryCondition
from isopleth.models.conditioning import PhysicalConditioningEncoder
from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D, MultiscaleFaceFluxOperator2D
from isopleth.training.checkpoints import CheckpointManager
from isopleth.training.curriculum import CurriculumScheduler
from isopleth.training.losses import LossComponents, PhysicalLoss

@dataclass
class TrainingHistory:
    train_loss: List[float]
    val_loss: List[float]
    train_mse: List[float]
    val_mse: List[float]
    val_conservation_penalty: List[float]
    learning_rates: List[float]
    epochs: List[int]
    best_epoch: int
    best_val_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_mse": self.train_mse,
            "val_mse": self.val_mse,
            "val_conservation_penalty": self.val_conservation_penalty,
            "learning_rates": self.learning_rates,
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }

class MultiHorizonTrainer:
    """Trainer orchestrating multi-horizon learning curricula and validation checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[PhysicalLoss] = None,
        scheduler: Optional[Any] = None,
        curriculum: Optional[CurriculumScheduler] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        conditioning_encoder: Optional[PhysicalConditioningEncoder] = None,
        device: str | torch.device = "cpu",
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn or PhysicalLoss()
        self.scheduler = scheduler
        self.curriculum = curriculum or CurriculumScheduler()
        self.checkpoint_manager = checkpoint_manager
        self.cond_encoder = conditioning_encoder.to(device) if conditioning_encoder else None
        self.device = torch.device(device)
        self.max_grad_norm = max_grad_norm

        self.history = TrainingHistory(
            train_loss=[],
            val_loss=[],
            train_mse=[],
            val_mse=[],
            val_conservation_penalty=[],
            learning_rates=[],
            epochs=[],
            best_epoch=-1,
            best_val_loss=float("inf"),
        )
        self.global_step = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        cell_measures: Tuple[float, ...],
    ) -> Dict[str, float]:
        """Execute one training epoch with active curriculum horizon."""
        self.model.train()
        stage = self.curriculum.get_stage(epoch)
        horizon = stage.rollout_horizon
        tf_ratio = stage.teacher_forcing_ratio

        total_loss = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch in train_loader:
            x_init = batch["x_init"].to(self.device)  # [B, ...]
            target_seq = batch["target"].to(self.device)  # [B, horizon, ...]
            dt = batch["dt"]
            if torch.is_tensor(dt):
                dt = dt[0].item()
            dx = cell_measures[0]
            dy = cell_measures[1] if len(cell_measures) > 1 else dx

            # Conditioning
            cond = None
            if self.cond_encoder is not None and "parameters" in batch:
                params = batch["parameters"].to(self.device)
                cond = self.cond_encoder(params, dt)

            self.optimizer.zero_grad()

            curr_state = x_init
            batch_loss = torch.tensor(0.0, device=self.device)
            batch_mse = 0.0

            # Multi-horizon rollout loop
            for step in range(min(horizon, target_seq.shape[1])):
                true_next = target_seq[:, step]

                # Model prediction
                if isinstance(self.model, MultiscaleFaceFluxOperator1D):
                    pred_next, _ = self.model(curr_state, dx=dx, dt=dt, cond=cond)
                elif isinstance(self.model, MultiscaleFaceFluxOperator2D):
                    pred_next, _ = self.model(curr_state, dx=dx, dy=dy, dt=dt, cond=cond)
                else:  # FNO / UNet
                    pred_next = self.model(curr_state, cond=cond)

                step_loss_comp = self.loss_fn(
                    pred=pred_next,
                    target=true_next,
                    init=curr_state,
                    cell_measures=cell_measures,
                )
                batch_loss = batch_loss + step_loss_comp.total_loss
                batch_mse += float(step_loss_comp.mse_loss.item())

                # Scheduled sampling
                use_teacher_forcing = (torch.rand(1).item() < tf_ratio)
                if use_teacher_forcing and self.model.training:
                    curr_state = true_next
                else:
                    curr_state = pred_next

            batch_loss.backward()
            if self.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += float(batch_loss.item())
            total_mse += batch_mse / max(1, horizon)
            n_batches += 1
            self.global_step += 1

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(1, n_batches)
        avg_mse = total_mse / max(1, n_batches)
        return {"loss": avg_loss, "mse": avg_mse}

    def validate(
        self,
        val_loader: DataLoader,
        cell_measures: Tuple[float, ...],
    ) -> Dict[str, float]:
        """Evaluate model strictly target-free without teacher forcing on validation partition."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_cons = 0.0
        n_batches = 0

        dx = cell_measures[0]
        dy = cell_measures[1] if len(cell_measures) > 1 else dx

        with torch.no_grad():
            for batch in val_loader:
                x_init = batch["x_init"].to(self.device)
                target_seq = batch["target"].to(self.device)
                dt = batch["dt"]
                if torch.is_tensor(dt):
                    dt = dt[0].item()

                cond = None
                if self.cond_encoder is not None and "parameters" in batch:
                    params = batch["parameters"].to(self.device)
                    cond = self.cond_encoder(params, dt)

                curr_state = x_init
                horizon = target_seq.shape[1]
                batch_loss = 0.0
                batch_mse = 0.0
                batch_cons = 0.0

                for step in range(horizon):
                    true_next = target_seq[:, step]

                    if isinstance(self.model, MultiscaleFaceFluxOperator1D):
                        pred_next, _ = self.model(curr_state, dx=dx, dt=dt, cond=cond)
                    elif isinstance(self.model, MultiscaleFaceFluxOperator2D):
                        pred_next, _ = self.model(curr_state, dx=dx, dy=dy, dt=dt, cond=cond)
                    else:
                        pred_next = self.model(curr_state, cond=cond)

                    comp = self.loss_fn(
                        pred=pred_next,
                        target=true_next,
                        init=curr_state,
                        cell_measures=cell_measures,
                    )
                    batch_loss += float(comp.total_loss.item())
                    batch_mse += float(comp.mse_loss.item())
                    batch_cons += float(comp.conservation_penalty.item())

                    # Strictly target-free rollout
                    curr_state = pred_next

                total_loss += batch_loss / max(1, horizon)
                total_mse += batch_mse / max(1, horizon)
                total_cons += batch_cons / max(1, horizon)
                n_batches += 1

        return {
            "val_loss": total_loss / max(1, n_batches),
            "val_mse": total_mse / max(1, n_batches),
            "val_cons": total_cons / max(1, n_batches),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        cell_measures: Tuple[float, ...],
    ) -> TrainingHistory:
        """Execute full training and validation campaign."""
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, epoch, cell_measures)
            val_metrics = self.validate(val_loader, cell_measures)
            elapsed = time.time() - t0

            cur_lr = self.optimizer.param_groups[0]["lr"]
            v_loss = val_metrics["val_loss"]

            self.history.epochs.append(epoch)
            self.history.train_loss.append(train_metrics["loss"])
            self.history.val_loss.append(v_loss)
            self.history.train_mse.append(train_metrics["mse"])
            self.history.val_mse.append(val_metrics["val_mse"])
            self.history.val_conservation_penalty.append(val_metrics["val_cons"])
            self.history.learning_rates.append(cur_lr)

            is_best = v_loss < self.history.best_val_loss
            if is_best:
                self.history.best_val_loss = v_loss
                self.history.best_epoch = epoch

            if self.checkpoint_manager:
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    global_step=self.global_step,
                    val_loss=v_loss,
                    best_val_loss=self.history.best_val_loss,
                    history=self.history.to_dict(),
                    is_best=is_best,
                )

        return self.history
