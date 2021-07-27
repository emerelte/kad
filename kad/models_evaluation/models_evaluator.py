import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from kad.kad_utils.kad_utils import GROUND_TRUTH_COLUMN, ANOMALIES_COLUMN, ANOM_SCORE_COLUMN


class ModelsEvaluator:

    def __init__(self, df: pd.DataFrame):
        """
        :param df: pd.Dataframe, columns: is_anomaly (bool) | gt_is_anomaly (bool)
        """
        self.df = df

    def get_accuracy(self):
        return round(metrics.accuracy_score(y_true=self.df[GROUND_TRUTH_COLUMN], y_pred=self.df[ANOMALIES_COLUMN]), 2)

    def plot_confusion_matrix(self):
        cm = metrics.confusion_matrix(y_true=self.df[GROUND_TRUTH_COLUMN], y_pred=self.df[ANOMALIES_COLUMN])
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)
        metrics.ConfusionMatrixDisplay(cm).plot(ax=ax)

    def plot_precision_recall_curve(self):
        plt.figure(figsize=(20, 10))
        precision, recall, thresholds = metrics.precision_recall_curve(y_true=self.df[GROUND_TRUTH_COLUMN],
                                                                       probas_pred=self.df[ANOM_SCORE_COLUMN])
        plt.step(recall, precision, color="k", alpha=0.7, where="post")
        plt.fill_between(recall, precision, step="post", alpha=0.3, color="k")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

    def get_average_precision(self):
        return round(
            metrics.average_precision_score(y_true=self.df[GROUND_TRUTH_COLUMN], y_score=self.df[ANOM_SCORE_COLUMN]), 2)

    def get_recall_score(self):
        return round(metrics.recall_score(y_true=self.df[GROUND_TRUTH_COLUMN], y_pred=self.df[ANOMALIES_COLUMN]), 2)

    def get_auroc(self):
        fpr, tpr, thresholds = metrics.roc_curve(y_true=self.df[GROUND_TRUTH_COLUMN],
                                                 y_score=self.df[ANOM_SCORE_COLUMN])
        return round(metrics.auc(fpr, tpr), 2)

    def plot_roc(self):
        fpr, tpr, thresholds = metrics.roc_curve(y_true=self.df[GROUND_TRUTH_COLUMN],
                                                 y_score=self.df[ANOM_SCORE_COLUMN])
        area_under_roc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(20, 10))
        plt.plot(fpr, tpr, color="r", lw=2, label="ROC curve")
        plt.plot([0, 1], [0, 1], color="k", lw=2, linestyle="--")
        plt.xlim([0.0, 1.05])
        plt.xlabel("False positive rate")
        plt.xlabel("True positive rate")
        plt.title(f"Receiver operationg characteristic: AUC = {area_under_roc:.2f}")
        plt.legend(loc="lower right")

    def calculate_first_scoring_component(self) -> float:
        anomaly_window = self.df[self.df[GROUND_TRUTH_COLUMN]].reset_index()
        anomaly_window_middle = anomaly_window.iloc[int(len(anomaly_window) / 2)]

        gt_anom_idx = anomaly_window.index[anomaly_window["timestamp"] == anomaly_window_middle["timestamp"]]

        detected_idxs = anomaly_window.index[anomaly_window[ANOMALIES_COLUMN]]
        if detected_idxs.empty:
            return 0.0

        dist_to_closest_pred = min([abs(gt_anom_idx - det_idx) for det_idx in detected_idxs])

        return 1.0 - dist_to_closest_pred / gt_anom_idx[0]

    @staticmethod
    def __calculate_positive_scoring_function(x) -> np.ndarray:
        coef = 0.5
        middle = int(len(x) / 2)

        return 1 / (1 + np.exp(coef * abs(x - middle)))

    def calculate_second_scoring_component(self) -> float:
        anomaly_window = self.df[self.df[GROUND_TRUTH_COLUMN]][[GROUND_TRUTH_COLUMN, ANOMALIES_COLUMN]].reset_index()

        anomaly_window["positive_scoring_func"] = self.__calculate_positive_scoring_function(anomaly_window.index)

        total_auc = np.sum(anomaly_window["positive_scoring_func"])
        detected_anomalies_auc = np.sum(anomaly_window[anomaly_window[ANOMALIES_COLUMN]]["positive_scoring_func"])
        print(2 * detected_anomalies_auc / total_auc)

        return min([1.0, 2 * detected_anomalies_auc / total_auc])

    @staticmethod
    def __calculate_negative_scoring_function(x) -> np.ndarray:
        coef = 0.01

        middle = int(len(x) / 2)

        return -1 / (1 + np.exp(-coef * abs(x - middle)))

    def calculate_third_scoring_component(self) -> float:
        all_but_anomaly_window = self.df[self.df[GROUND_TRUTH_COLUMN] == False][
            [GROUND_TRUTH_COLUMN, ANOMALIES_COLUMN]].reset_index()

        all_but_anomaly_window["negative_scoring_func"] = self.__calculate_negative_scoring_function(
            all_but_anomaly_window.index)

        plt.plot(all_but_anomaly_window.index.to_numpy(), all_but_anomaly_window["negative_scoring_func"])
        plt.show()

        total_auc = np.sum(all_but_anomaly_window["negative_scoring_func"])
        false_positives_auc = np.sum(
            all_but_anomaly_window[all_but_anomaly_window[ANOMALIES_COLUMN]]["negative_scoring_func"])

        return max([0.0, 1 - false_positives_auc / total_auc])
