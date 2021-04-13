import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

X_LABEL = "timestamp"

def visualize(results_df: pd.DataFrame, metric_name: str):

    results_df.index = results_df.index.set_names([X_LABEL])
    df_subset = results_df.iloc[np.where(results_df["is_anomaly"])]

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    results_df[metric_name].plot.line(ax=ax)

    if np.any(results_df["predictions"]):
        results_df.reset_index().plot.scatter(
            x=X_LABEL,
            y="predictions",
            ax=ax,
            color="g")

    if np.any(results_df["is_anomaly"]):
        df_subset.reset_index().plot.scatter(
            x=X_LABEL,
            y=metric_name,
            ax=ax,
            color="r")

    plt.legend(["Actual TS", "Predictions", "Anomalies"])
