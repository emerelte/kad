import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from kad.kad_utils.kad_utils import X_LABEL, PREDICTIONS_COLUMN, ANOMALIES_COLUMN, GROUND_TRUTH_COLUMN, \
    ANOM_SCORE_COLUMN


def visualize(results_df: pd.DataFrame, metric_name: str, title: str = "Anomaly visualization"):
    results_df.index = results_df.index.set_names([X_LABEL])

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20, 10)
    results_df[metric_name].plot.line(ax=ax)

    columns_labels: list = ["Actual TS"]

    if PREDICTIONS_COLUMN in results_df and np.any(results_df[PREDICTIONS_COLUMN]):
        results_df.reset_index().plot.scatter(
            x=X_LABEL,
            y=PREDICTIONS_COLUMN,
            ax=ax,
            color="b")
        columns_labels.append("Predictions")

    if ANOMALIES_COLUMN in results_df and np.any(results_df[ANOMALIES_COLUMN]):
        results_df[ANOMALIES_COLUMN] = results_df[ANOMALIES_COLUMN].fillna(False)
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

    if GROUND_TRUTH_COLUMN in results_df and ANOMALIES_COLUMN in results_df and np.any(
            results_df[results_df[GROUND_TRUTH_COLUMN] & results_df[ANOMALIES_COLUMN]]):
        results_df[results_df[GROUND_TRUTH_COLUMN] & results_df[ANOMALIES_COLUMN]].reset_index().plot.scatter(
            x=X_LABEL,
            y="value",
            ax=ax,
            color=["magenta"])
        columns_labels.append("Predicted & GT Anomalies")

    ax.legend(columns_labels)
    fig.suptitle(title, fontsize=16)

    if ANOM_SCORE_COLUMN in results_df:
        fig2, ax2 = plt.subplots(1, 1)
        fig2.set_size_inches(20, 10)
        results_df[ANOM_SCORE_COLUMN].reset_index().plot(
            x=X_LABEL,
            y=ANOM_SCORE_COLUMN,
            ax=ax2,
            kind="bar")
        ax2.axes.get_xaxis().set_visible(False)
        ax2.axes.set_ylabel("anomaly score")
