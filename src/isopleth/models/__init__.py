"""Neural operator architectures, baselines, and interface modules for Isopleth."""

from isopleth.models.conditioning import (
    FiLMBlock,
    FourierEmbedding,
    PhysicalConditioningEncoder,
)
from isopleth.models.fno import (
    FNO1D,
    FNO2D,
    SpectralConv1d,
    SpectralConv2d,
)
from isopleth.models.unet import (
    UNet1D,
    UNet2D,
)
from isopleth.models.interfaces import (
    DiscreteFluxDivergence,
    InterfaceFluxHead1D,
    InterfaceFluxHead2D,
)
from isopleth.models.flux_operator import (
    MultiscaleFaceFluxOperator1D,
    MultiscaleFaceFluxOperator2D,
)

__all__ = [
    "DiscreteFluxDivergence",
    "FNO1D",
    "FNO2D",
    "FiLMBlock",
    "FourierEmbedding",
    "InterfaceFluxHead1D",
    "InterfaceFluxHead2D",
    "MultiscaleFaceFluxOperator1D",
    "MultiscaleFaceFluxOperator2D",
    "PhysicalConditioningEncoder",
    "SpectralConv1d",
    "SpectralConv2d",
    "UNet1D",
    "UNet2D",
]
