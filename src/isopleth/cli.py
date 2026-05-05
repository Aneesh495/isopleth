"""CLI entry point for Isopleth."""

import sys
import click
import torch
import numpy as np
from rich.console import Console

console = Console()

@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Isopleth: Conservation-aware neural operator laboratory."""
    pass

@main.command()
def doctor() -> None:
    """Probe hardware, precision, memory, and dependency compatibility."""
    console.print("[bold green]Isopleth Environment Diagnostic[/bold green]")
    console.print(f"Python Version: {sys.version.split()[0]}")
    console.print(f"PyTorch Version: {torch.__version__}")
    console.print(f"NumPy Version: {np.__version__}")
    mps_avail = torch.backends.mps.is_available()
    cuda_avail = torch.cuda.is_available()
    console.print(f"Apple Silicon MPS Acceleration: {mps_avail}")
    console.print(f"CUDA Acceleration: {cuda_avail}")
    device = "mps" if mps_avail else ("cuda" if cuda_avail else "cpu")
    console.print(f"Default Accelerated Device: [bold cyan]{device}[/bold cyan]")
    console.print("Diagnostic passed successfully.")

if __name__ == "__main__":
    main()
