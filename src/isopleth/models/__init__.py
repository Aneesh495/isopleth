"""Neural operator architectures, baselines, and interface modules for Isopleth."""

from isopleth.models.conditioning import (
    FiLMBlock,
    FourierEmbedding,
    PhysicalConditioningEncoder,
)
from isopleth.models.deeponet import (
    BranchMLP,
    ConservativeFluxDeepONet1D,
    DeepONet1D,
    DeepONet2D,
    TrunkMLP,
)
from isopleth.models.flux_operator import (
    MultiscaleFaceFluxOperator1D,
    MultiscaleFaceFluxOperator2D,
)
from isopleth.models.fno import (
    FNO1D,
    FNO2D,
    SpectralConv1d,
    SpectralConv2d,
)
from isopleth.models.interfaces import (
    DiscreteFluxDivergence,
    InterfaceFluxHead1D,
    InterfaceFluxHead2D,
)
from isopleth.models.unet import (
    UNet1D,
    UNet2D,
)
from isopleth.models.wavelet_operator import (
    DWT1D,
    IDWT1D,
    WaveletConv1D,
    WaveletConv2D,
    WaveletNeuralOperator1D,
    WaveletNeuralOperator2D,
)

__all__ = [
    "BranchMLP",
    "ConservativeFluxDeepONet1D",
    "DWT1D",
    "DeepONet1D",
    "DeepONet2D",
    "DiscreteFluxDivergence",
    "FNO1D",
    "FNO2D",
    "FiLMBlock",
    "FourierEmbedding",
    "IDWT1D",
    "InterfaceFluxHead1D",
    "InterfaceFluxHead2D",
    "MultiscaleFaceFluxOperator1D",
    "MultiscaleFaceFluxOperator2D",
    "PhysicalConditioningEncoder",
    "SpectralConv1d",
    "SpectralConv2d",
    "TrunkMLP",
    "UNet1D",
    "UNet2D",
    "WaveletConv1D",
    "WaveletConv2D",
    "WaveletNeuralOperator1D",
    "WaveletNeuralOperator2D",
]
