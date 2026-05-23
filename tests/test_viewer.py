"""Tests for scientific visualization payload builder and server."""

import json
import urllib.request
import pytest
import torch

from isopleth.viewer import (
    ScientificViewerServer,
    ScientificVisualizationBuilder,
    ViewerPayload,
)


def test_scientific_visualization_builder_1d_and_2d() -> None:
    """Verifies 1D and 2D payload generation from tensors."""
    # 1D payload
    x = torch.linspace(0, 1, 32)
    traj_1d = torch.stack([torch.sin(2 * 3.14159 * x), torch.cos(2 * 3.14159 * x)], dim=0)
    payload_1d = ScientificVisualizationBuilder.build_1d_payload(
        predicted_trajectory=traj_1d,
        ground_truth_trajectory=traj_1d,
        dt=0.01,
    )

    assert payload_1d.spatial_dimension == 1
    assert payload_1d.grid_shape == [32]
    assert payload_1d.snapshots_1d is not None
    assert len(payload_1d.snapshots_1d) == 2
    assert len(payload_1d.spectral_cascade) > 0
    assert len(payload_1d.conservation_balances) == 2

    json_str = payload_1d.to_json()
    assert len(json_str) > 100

    # 2D payload
    traj_2d = torch.ones(3, 2, 16, 16)
    payload_2d = ScientificVisualizationBuilder.build_2d_payload(
        predicted_trajectory=traj_2d,
        ground_truth_trajectory=traj_2d,
        channel_names=["depth", "velocity_x"],
        dt=0.005,
    )

    assert payload_2d.spatial_dimension == 2
    assert payload_2d.grid_shape == [16, 16]
    assert payload_2d.snapshots_2d is not None
    assert len(payload_2d.snapshots_2d) > 0


def test_viewer_http_server_endpoints() -> None:
    """Verifies that the viewer server serves workbench HTML and REST endpoints."""
    server = ScientificViewerServer(port=8799)
    server.start(blocking=False)

    try:
        # Test index.html endpoint
        req_index = urllib.request.Request("http://127.0.0.1:8799/")
        with urllib.request.urlopen(req_index) as response:
            assert response.status == 200
            content = response.read().decode("utf-8")
            assert "Isopleth Laboratory Workbench" in content

        # Test api/payload endpoint
        req_api = urllib.request.Request("http://127.0.0.1:8799/api/payload")
        with urllib.request.urlopen(req_api) as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            assert "system_family" in data

        # Test api/health endpoint
        req_health = urllib.request.Request("http://127.0.0.1:8799/api/health")
        with urllib.request.urlopen(req_health) as response:
            assert response.status == 200
            health = json.loads(response.read().decode("utf-8"))
            assert health["status"] == "ok"
    finally:
        server.stop()
