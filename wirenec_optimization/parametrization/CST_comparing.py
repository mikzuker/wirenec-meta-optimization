import matplotlib.pyplot as plt
import csv
from itertools import islice
from omegaconf import DictConfig, OmegaConf
from wirenec_optimization.optimization_configs.utils import parse_config
from wirenec_optimization.optimization_utils.bandwidth_creating import (
    freq_maker,)
from pathlib import Path

config_path: Path = Path()
config = parse_config(config_path)
optimization_hyperparams = OmegaConf.to_container(config.optimization_hyperparams, resolve=True)

if config.get("parametrization_class") == "diffuser":
    assert config.optimization_hyperparams.bandwidth is not None
    config.optimization_hyperparams.frequencies = tuple(
        freq_maker(
            config.optimization_hyperparams.general_frequency,
            config.optimization_hyperparams.bandwidth,
            config.optimization_hyperparams.number_of_frequencies,
        )
    )

with open() as datafile:
    plotting = csv.reader(datafile, delimiter="\t")

    X_CST_str = []
    Y_CST_str = []
    X_CST = []
    Y_CST = []

    for line in islice(datafile, 24, None):

        for ROWS in plotting:
            X_CST_str.append(ROWS[0])
            Y_CST_str.append(ROWS[1])
    X_CST_str = [i.replace(',', '.') for i in X_CST_str]
    Y_CST_str = [i.replace(',', '.') for i in Y_CST_str]

    for i in range(len(X_CST_str)):
        X_CST.append(float(X_CST_str[i])*10**3)
        Y_CST.append(float(Y_CST_str[i])*10**4)


with open() as datafile:
    plotting = csv.reader(datafile, delimiter="\t")

    X_str = []
    Y_str = []
    X = []
    Y = []

    Y_init_str = []
    Y_init = []

    for line in islice(datafile, 1, None):

        for ROWS in plotting:
            X_str.append(ROWS[0])
            Y_str.append(ROWS[1])
            Y_init_str.append(ROWS[2])

        for i in range(len(X_str)):
            X.append(float(X_str[i]))
            Y.append(float(Y_str[i]))
            Y_init.append(float(Y_init_str[i]))


fig, ax = plt.subplots(1, figsize=(7, 6))

ax.plot(X_CST, Y_CST, lw=2)
ax.plot(X, Y, lw=2)
ax.plot(X, Y_init, linestyle=":", lw=2)

ax.axhline(0, color="k", lw=1)
ax.scatter(
    optimization_hyperparams["frequencies"],
    [0] * len(optimization_hyperparams["frequencies"]),
    color="darkgreen",
    marker="s",
    alpha=0.5,
    label="Optimized frequencies",
    )
ax.fill_between(
    optimization_hyperparams["frequencies"],
    0,
    150 * 1.1,
    color="darkgreen",
    alpha=0.1,
    label="Optimized area",
    )

plt.xlim(4000, 9000)

plt.legend(['CST', 'Evolutionary algorithm', 'Initial object'])
plt.title('Comparing CST and algorithm data')
plt.xlabel('frequency, MHz')
plt.ylabel('Backward scattering, $cm^2$')

plt.show()
