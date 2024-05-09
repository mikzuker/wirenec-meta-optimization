from abc import ABC
from typing import Iterable, Optional, Callable

import numpy as np
import pandas as pd
import sklearn.metrics

from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from plotly import graph_objs as go
import plotly.offline as py
import os


class BaseMetric(ABC):
    def calculate(self, **kwargs) -> float | pd.Series:
        pass


class DummyMetric(BaseMetric):
    def calculate(self, **kwargs) -> float | pd.Series:
        return 42


class SpectrumMetrics(BaseMetric):
    def __init__(self, metric_list: Optional[Iterable[str]] = None):
        self.available_metrics = {
            "rmse": self.rmse,
            "max": self.meta_pandas_func("max"),
            "min": self.meta_pandas_func("min"),
            "mean": self.meta_pandas_func("mean"),
            "std": self.meta_pandas_func("std"),
        }
        self.opt_scattering_col = "scaterring_180"
        self.initial_scattering_col = "initial"
        self.metric_list = metric_list if metric_list is not None else list(self.available_metrics.keys())

    @staticmethod
    def crop_spectrum(df: pd.DataFrame) -> pd.DataFrame:
        freq_min, freq_max = df[["freq_min", "freq_max"]].iloc[0]
        mask = (df.frequency >= freq_min) & (df.frequency <= freq_max)
        return df[mask]

    def meta_pandas_func(self, fun: str) -> Callable:
        """
        The method which allows to avoid implementing identical functions,
        which differs only in the pandas method at the end.
        For example, df.min(), df.max(), df.mean(), etc.
        """
        return lambda x: getattr(self.crop_spectrum(x)[self.opt_scattering_col], fun)()

    def rmse(self, df: pd.DataFrame) -> float:
        """
        Calculate the Root Mean Square Error (RMSE) between the optimized and initial scattering columns of a DataFrame.
        """
        df_cropped = self.crop_spectrum(df)
        return np.sqrt(
            sklearn.metrics.mean_squared_error(
                df_cropped[self.opt_scattering_col],
                df_cropped[self.initial_scattering_col],
            )
        )

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        results = pd.Series(
            {name: fun(df) for name, fun in self.available_metrics.items() if name in self.available_metrics}
        )
        return results


# class Visualizer(ABC):
#     def __init__(self):
#         self.available_plotting = ["lineplot", "boxplot", "scatter3d", "scatterplot_with_gradient"]
#
#     def show_plots(self):
#         print(self.available_plotting)
#
#     def visualize_metrics(self, folder_path: Path,
#                           y_data_column: str,
#                           x_data_column: str,
#                           plt_type: str,
#                           z_data_column: Optional[str] = None,
#                           hue_data_column: Optional[str] = None,
#                           ):
#         if plt_type in self.available_plotting:
#             with (folder_path / "aggregated_metrics.csv").open("rb") as file:
#                 data = pd.read_csv(file)
#             directory = str(folder_path)
#
#             if plt_type == 'boxplot':
#                 plot_data = sns.boxplot(data, y=y_data_column, x=x_data_column)
#                 plt.title('Metrics aggregation')
#                 plt.show()
#
#             if plt_type == 'lineplot':
#                 plot_data = sns.lineplot(data, y=y_data_column, x=x_data_column)
#                 plt.title('Metrics aggregation')
#                 plt.show()
#
#             if plt_type == 'scatterplot_with_gradient':
#                 plot_data = sns.scatterplot(data, y=y_data_column, x=x_data_column, hue=hue_data_column)
#                 plt.title('Metrics aggregation')
#                 plt.show()
#
#             if plt_type == 'scatter3d':
#                 plot_data = go.Figure(data=[go.Scatter3d(x=data[f'{x_data_column}'],
#                                                          y=data[f'{y_data_column}'],
#                                                          z=data[f'{z_data_column}'],
#                                                          marker=dict(opacity=0.9,
#                                                                      reversescale=True,
#                                                                      colorscale='Blues',
#                                                                      size=5),
#                                                          line=dict(width=0.02),
#                                                          mode='markers')])
#                 plot_data.update_layout(title='Metrics aggregation')
#                 plot_data.update_layout(scene=dict(xaxis=dict(title=f"{x_data_column}"),
#                                                    yaxis=dict(title=f"{y_data_column}"),
#                                                    zaxis=dict(title=f"{z_data_column}")), )
#                 filename_data = f'{x_data_column}_{y_data_column}_{z_data_column}_3D.html'
#
#                 filepath = os.path.join(directory, filename_data)
#                 if os.path.exists(filepath):
#                     print(f'These graphic exists already in {filepath} directory')
#                 else:
#                     py.plot(plot_data, filename=filepath)
#                     plt.show()
#         else:
#             print('Unexpected plot, to see available plots please call Visualizer.show_plots()')


from typing import Callable, Any


class Visualizer:
    def __init__(self):
        self.available_plotting: dict[str, Callable] = {
            "boxplot": self.box_plot,
        }

    @staticmethod
    def box_plot(data: pd.DataFrame, x_data_column: str, y_data_column: str):
        fig, ax = plt.subplots()
        sns.boxplot(data, y=y_data_column, x=x_data_column, ax=ax)
        return fig, ax

    def show_plots(self):
        print(self.available_plotting)

    def visualize_metrics(self, data: pd.DataFrame, plt_type: str, **kwargs: Any):
        if plt_type not in self.available_plotting.keys():
            raise Exception(f"Invalid plt type: {plt_type}")

        fig, ax = self.available_plotting[plt_type](data, **kwargs)
        return fig, ax


if __name__ == '__main__':
    tp = 'boxplot'
    d = pd.read_csv(...)
    viz = Visualizer()
    fig, ax = viz.visualize_metrics(d, tp, )
# if __name__ == "__main__":
#     metrics = DummyMetric()
# print(metrics.calculate())
# o = Visualizer()
# print(o.show_plots())
