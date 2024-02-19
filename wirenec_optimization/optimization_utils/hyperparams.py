import numpy as np


def freq_maker(general_freq: float, band_width: float) -> tuple[float]:
    number_freq = 5
    frequencies = np.linspace(
        general_freq - band_width / 2,
        general_freq + band_width / 2,
        number_freq,
        endpoint=True,
    )
    frequencies = tuple(map(float, frequencies))
    return frequencies
