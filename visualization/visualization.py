import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def visualize(resulting_df: pd.DataFrame, metric_name: str):
    df_subset = resulting_df.iloc[np.where(resulting_df["is_anomaly"])]
    _, ax = plt.subplots()
    resulting_df[metric_name].plot.line(ax=ax)

    if np.any(resulting_df["is_anomaly"]):
        df_subset[metric_name].plot.line(ax=ax, color="r")
    plt.show()
