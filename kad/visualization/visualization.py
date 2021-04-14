import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

X_LABEL: str = "timestamp"

PREDICTIONS_COLUMN: str = "predictions"
ANOMALIES_COLUMN: str = "is_anomaly"
GROUND_TRUTH_COLUMN: str = "gt_is_anomaly"


def visualize(results_df: pd.DataFrame, metric_name: str):
    results_df[ANOMALIES_COLUMN] = results_df[ANOMALIES_COLUMN].fillna(False)
    results_df.index = results_df.index.set_names([X_LABEL])

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    results_df[metric_name].plot.line(ax=ax)

    columns_labels: list = ["Actual TS"]

    if PREDICTIONS_COLUMN in results_df and np.any(results_df[PREDICTIONS_COLUMN]):
        results_df.reset_index().plot.scatter(
            x=X_LABEL,
            y="predictions",
            ax=ax,
            color="g")
        columns_labels.append("Predictions")

    if ANOMALIES_COLUMN in results_df and np.any(results_df[ANOMALIES_COLUMN]):
        results_df[results_df[ANOMALIES_COLUMN]].reset_index().plot.scatter(
            x=X_LABEL,
            y=metric_name,
            ax=ax,
            color="r")
        columns_labels.append("Anomalies")

    if GROUND_TRUTH_COLUMN in results_df and np.any(results_df[GROUND_TRUTH_COLUMN]):
        results_df[results_df[GROUND_TRUTH_COLUMN]].reset_index().plot.scatter(
            x=X_LABEL,
            y="value",
            ax=ax,
            color="g")
        columns_labels.append("GT Anomalies")

    plt.legend(columns_labels)
