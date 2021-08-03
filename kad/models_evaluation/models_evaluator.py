import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics

from kad.kad_utils.kad_utils import GROUND_TRUTH_COLUMN, ANOMALIES_COLUMN, ANOM_SCORE_COLUMN, SCORING_FUNCTION_COLUMN


class ModelsEvaluator:

    def __init__(self, df: pd.DataFrame):
        """
        :param df: pd.Dataframe, columns: is_anomaly (bool) | gt_is_anomaly (bool)
        """
        self.df = df.reset_index()
        self.__calculate_scoring_function()

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

    def __calculate_scoring_function(self):
        temp_df = self.df.reset_index()
        temp_df["value_grp"] = (temp_df[GROUND_TRUTH_COLUMN].diff(1) != 0).astype("int").cumsum()
        grouped = temp_df.groupby("value_grp")

        total_index = self.df.reset_index().index.to_numpy()

        self.df[SCORING_FUNCTION_COLUMN] = -1

        for name, group in grouped:
            if group[GROUND_TRUTH_COLUMN].all():
                anom_idx_in_window = int(len(group) / 2)
                anom_idx = group.index[anom_idx_in_window]
                new_scoring_func = 2 / (1 + np.exp(np.abs(total_index - anom_idx) - anom_idx_in_window)) - 1
                self.df[SCORING_FUNCTION_COLUMN] = np.maximum(new_scoring_func,
                                                              self.df[SCORING_FUNCTION_COLUMN].to_numpy())

        plt.plot(self.df.index.to_numpy(), self.df[SCORING_FUNCTION_COLUMN])
        plt.show()

    def calculate_first_scoring_component(self) -> float:
        temp_df = self.df.reset_index()
        temp_df["value_grp"] = (temp_df[GROUND_TRUTH_COLUMN].diff(1) != 0).astype("int").cumsum()
        grouped = temp_df.groupby("value_grp")

        partial_scores = np.array([])

        for name, anomaly_window in grouped:
            if anomaly_window[GROUND_TRUTH_COLUMN].all():
                anom_idx_in_window = int(len(anomaly_window) / 2)
                anomaly_window_middle = anomaly_window.iloc[int(len(anomaly_window) / 2)]
                gt_anom_idx = anomaly_window.index[anomaly_window["timestamp"] == anomaly_window_middle["timestamp"]]
                detected_idxs = anomaly_window.index[anomaly_window[ANOMALIES_COLUMN]]
                if detected_idxs.empty:
                    partial_scores = np.append(partial_scores, 0.0)
                    continue

                dist_to_closest_pred = min([abs(gt_anom_idx - det_idx) for det_idx in detected_idxs])
                partial_scores = np.append(partial_scores, 1.0 - dist_to_closest_pred[0] / anom_idx_in_window)

        return float(np.mean(partial_scores))

    def calculate_second_scoring_component(self) -> float:
        positive_scoring_function = np.array([v if v > 0 else 0 for v in self.df[SCORING_FUNCTION_COLUMN]])
        total_auc = np.sum(positive_scoring_function)

        detected_anomalies = np.array([sf*anom for sf, anom in zip(positive_scoring_function, self.df[ANOMALIES_COLUMN])])
        detected_anomalies_auc = np.sum(detected_anomalies)

        return min([1.0, detected_anomalies_auc / total_auc])

    def calculate_third_scoring_component(self) -> float:
        negative_scoring_function = np.array([v if v < 0 else 0 for v in self.df[SCORING_FUNCTION_COLUMN]])
        total_auc = np.sum(negative_scoring_function)

        detected_anomalies = np.array([sf*anom for sf, anom in zip(negative_scoring_function, self.df[ANOMALIES_COLUMN])])
        detected_anomalies_auc = np.sum(detected_anomalies)

        return min([1.0, 1.0 - detected_anomalies_auc / total_auc])

    def get_customized_score(self) -> float:
        print("1st: ", self.calculate_first_scoring_component())
        print("2nd: ", self.calculate_second_scoring_component())
        print("3rd: ", self.calculate_third_scoring_component())

        return (self.calculate_first_scoring_component() +
                self.calculate_second_scoring_component() +
                self.calculate_third_scoring_component()) / 3
