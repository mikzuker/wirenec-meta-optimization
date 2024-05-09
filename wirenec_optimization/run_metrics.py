import json
from pathlib import Path

import pandas as pd

from wirenec_optimization.metrics.metrics import SpectrumMetrics, Visualizer


def process_folder(folder_path: Path) -> pd.Series:
    hyperparams_folder = folder_path / "hyperparams/"

    with (hyperparams_folder / "optimization_hyperparams.json").open("rb") as fp:
        optimization_hyperparams = json.load(fp)
    with (hyperparams_folder / "object_hyperparams.json").open("rb") as fp:
        object_hyperparams = json.load(fp)

    df = pd.read_csv(folder_path / "scat_data.json", sep="\t")

    frequencies = optimization_hyperparams.get("frequencies")
    df["freq_min"], df["freq_max"] = min(frequencies), max(frequencies)

    metrics = SpectrumMetrics()
    res: pd.Series = metrics.calculate(df)

    res["seed"] = optimization_hyperparams.get("seed")
    res["bandwidth"] = optimization_hyperparams.get("bandwidth")
    res["number_of_freq"] = optimization_hyperparams.get("number_of_frequencies")
    res["dist_from_surf"] = object_hyperparams.get("dist_from_obj_to_surf")
    res["folder_path"] = str(folder_path)

    return res


if __name__ == "__main__":
    root_dir = Path.cwd()
    base_folder = root_dir / "data/bandwidth_optimization/"
    experiments_folders = [
        Path("/Users/konstantingrotov/Downloads/big_exp_90_seeds"),
        Path("/Users/konstantingrotov/Downloads/big_exp_2_500_seeds"),
    ]
    folders = []
    for base in experiments_folders:
        folders.extend([x for x in base.iterdir() if x.is_dir()])
    print(len(folders))

    df_agg = pd.DataFrame([process_folder(p) for p in folders])
    print(df_agg)

    # Plotting results
    # fig, ax = plt.subplots()
    # boxplot = df_agg.boxplot(column=["rmse"])
    # plt.show()

    # Saving results
    df_agg.to_csv(base_folder / "aggregated_metrics.csv")

    Plot = Visualizer()
    Plot.visualize_metrics(
        folder_path=base_folder,
        y_data_column="mean",
        x_data_column="number_of_freq",
        # y_data_column='number_of_freq',
        plt_type="scatterplot_with_gradient",
    )
