import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics

PREDICTED_COLUMN: str = "is_anomaly"
GT_COLUMN: str = "gt_is_anomaly"


class ModelsEvaluator:

    def __init__(self, df: pd.DataFrame):
        """
        :param df: pd.Dataframe, columns: is_anomaly (bool) | gt_is_anomaly (bool)
        """
        self.df = df

    def get_accuracy(self):
        return metrics.accuracy_score(y_true=self.df[GT_COLUMN], y_pred=self.df[PREDICTED_COLUMN])

    def plot_confusion_matrix(self):
        cm = metrics.confusion_matrix(y_true=self.df[GT_COLUMN], y_pred=self.df[PREDICTED_COLUMN])
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)
        metrics.ConfusionMatrixDisplay(cm).plot(ax=ax)

    def plot_precision_recall_curve(self):
        plt.figure(figsize=(20, 10))
        precision, recall, thresholds = metrics.precision_recall_curve(y_true=self.df[GT_COLUMN],
                                                                       probas_pred=self.df[PREDICTED_COLUMN])
        plt.step(recall, precision, color="k", alpha=0.7, where="post")
        plt.fill_between(recall, precision, step="post", alpha=0.3, color="k")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

    def get_average_precision(self):
        return metrics.average_precision_score(y_true=self.df[GT_COLUMN], y_score=self.df[PREDICTED_COLUMN])

    def plot_roc(self):
        fpr, tpr, thresholds = metrics.roc_curve(y_true=self.df[GT_COLUMN], y_score=self.df[PREDICTED_COLUMN])
        area_under_roc = metrics.auc(fpr, tpr)

        plt.figure(figsize=(20, 10))
        plt.plot(fpr, tpr, color="r", lw=2, label="ROC curve")
        plt.plot([0, 1], [0, 1], color="k", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.xlim([0.0, 1.05])
        plt.xlabel("False positive rate")
        plt.xlabel("True positive rate")
        plt.title(f"Receiver operationg characteristic: AUC = {area_under_roc:.2f}")
        plt.legend(loc="lower right")
