import numpy as np
from wirenec_optimization.optimization_configs.utils import parse_config
from pathlib import Path


def freq_maker(general_freq: float, band_width: float, number_freq: int) -> tuple[float]:
    frequencies = np.linspace(
        general_freq - band_width / 2,
        general_freq + band_width / 2,
        number_freq,
        endpoint=True,
    )
    frequencies = tuple(map(float, frequencies))
    return frequencies
