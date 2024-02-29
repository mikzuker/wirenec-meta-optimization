from abc import ABC
from typing import Iterable, Optional, Callable

import numpy as np
import pandas as pd
import sklearn.metrics


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
        self.metric_list = (
            metric_list
            if metric_list is not None
            else list(self.available_metrics.keys())
        )

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
            {
                name: fun(df)
                for name, fun in self.available_metrics.items()
                if name in self.available_metrics
            }
        )
        return results


if __name__ == "__main__":
    metrics = DummyMetric()
    print(metrics.calculate())
