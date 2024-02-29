import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from wirenec_optimization.metrics.metrics import SpectrumMetrics


def process_folder(folder_path: Path) -> pd.Series:
    hyperparams_folder = folder_path / "hyperparams/"

    with (hyperparams_folder / "optimization_hyperparams.json").open("rb") as fp:
        optimization_hyperparams = json.load(fp)

    df = pd.read_csv(folder_path / "scat_data.txt", sep="\t")

    frequencies = optimization_hyperparams.get("frequencies")
    df["freq_min"], df["freq_max"] = min(frequencies), max(frequencies)

    metrics = SpectrumMetrics()
    res: pd.Series = metrics.calculate(df)

    res["seed"] = optimization_hyperparams.get("seed")
    res["bandwidth"] = optimization_hyperparams["bandwidth"]

    return res


if __name__ == "__main__":
    root_dir = Path.cwd()
    experiments_folder = root_dir / "data/bandwidth_optimization/exp1_10_seeds/"
    folders = [x for x in experiments_folder.iterdir() if x.is_dir()]

    df_agg = pd.DataFrame([process_folder(p) for p in folders])
    print(df_agg)

    # Plotting results
    fig, ax = plt.subplots()
    boxplot = df_agg.boxplot(column=["rmse"])
    plt.show()

    # Saving results
    # df_agg.to_csv(experiments_folder / "aggregated_metrics.csv")
