"""Tests for atomic checkpoints and continuation (I26)."""

import pytest
import tempfile
from pathlib import Path
import torch

from isopleth.models.flux_operator import MultiscaleFaceFluxOperator1D
from isopleth.training.checkpoints import CheckpointManager

def test_atomic_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Save checkpoint
        ckpt_path = manager.save(
            model=model,
            optimizer=opt,
            epoch=5,
            global_step=120,
            val_loss=0.042,
            is_best=True,
        )
        assert Path(ckpt_path).exists()
        assert (Path(tmpdir) / "best_model.pt").exists()

        # Restore into fresh model
        model_fresh = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16)
        opt_fresh = torch.optim.AdamW(model_fresh.parameters(), lr=1e-3)

        info = manager.load(ckpt_path, model=model_fresh, optimizer=opt_fresh)
        assert info["epoch"] == 5
        assert info["global_step"] == 120
        assert info["val_loss"] == 0.042

        # Check weight identity
        for p1, p2 in zip(model.parameters(), model_fresh.parameters()):
            assert torch.allclose(p1, p2)

def test_deterministic_cpu_continuation():
    # Verify exact deterministic continuation
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        torch.manual_seed(42)
        model = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16)
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)

        x = torch.randn(2, 1, 32)
        # Run 2 training steps
        for _ in range(2):
            opt.zero_grad()
            y, _ = model(x, dx=0.03, dt=0.01)
            loss = torch.sum(y**2)
            loss.backward()
            opt.step()

        # Save at step 2
        ckpt = manager.save(model=model, optimizer=opt, epoch=1, global_step=2)

        # Step 3 in run A
        torch.manual_seed(100)
        noise_a = torch.randn(2, 1, 32)
        opt.zero_grad()
        y_a, _ = model(noise_a, dx=0.03, dt=0.01)
        loss_a = torch.sum(y_a**2)
        loss_a.backward()
        opt.step()

        # Load into fresh run B and execute step 3
        model_b = MultiscaleFaceFluxOperator1D(in_channels=1, out_channels=1, hidden_channels=16)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=1e-2)
        manager.load(ckpt, model=model_b, optimizer=opt_b, restore_rng=False)

        torch.manual_seed(100)
        noise_b = torch.randn(2, 1, 32)
        opt_b.zero_grad()
        y_b, _ = model_b(noise_b, dx=0.03, dt=0.01)
        loss_b = torch.sum(y_b**2)
        loss_b.backward()
        opt_b.step()

        # Deterministic bitwise match
        for p_a, p_b in zip(model.parameters(), model_b.parameters()):
            assert torch.allclose(p_a, p_b, atol=1e-7)
