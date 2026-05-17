"""Tests for multi-horizon trainer and validation selection (I15)."""

import pytest
import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D
from isopleth.training.checkpoints import CheckpointManager
from isopleth.training.curriculum import CurriculumScheduler, CurriculumStage
from isopleth.training.trainer import MultiHorizonTrainer

class MockDataset:
    def __init__(self, n_samples: int = 16, nx: int = 32, horizon: int = 2) -> None:
        self.n_samples = n_samples
        self.nx = nx
        self.horizon = horizon

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return {
            "x_init": torch.randn(1, self.nx),
            "target": torch.randn(self.horizon, 1, self.nx),
            "dt": 0.01,
            "parameters": torch.tensor([0.01, 0.4]),
        }

def test_trainer_fit_loop():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = MultiscaleFaceFluxOperator1D(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=4,
            num_layers=1,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        manager = CheckpointManager(tmpdir)
        curriculum = CurriculumScheduler(
            stages=[CurriculumStage(max_epoch=2, rollout_horizon=2, teacher_forcing_ratio=0.5)]
        )

        trainer = MultiHorizonTrainer(
            model=model,
            optimizer=opt,
            curriculum=curriculum,
            checkpoint_manager=manager,
            device="cpu",
        )

        train_loader = DataLoader(MockDataset(n_samples=8, nx=32, horizon=2), batch_size=4)
        val_loader = DataLoader(MockDataset(n_samples=4, nx=32, horizon=2), batch_size=4)

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            cell_measures=(1.0 / 32,),
        )

        assert len(history.epochs) == 2
        assert len(history.train_loss) == 2
        assert len(history.val_loss) == 2
        assert history.best_val_loss < float("inf")
        assert (Path(tmpdir) / "best_model.pt").exists()
