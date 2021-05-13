import logging
from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from kad.kad_utils.kad_utils import ANOMALIES_COLUMN, PREDICTIONS_COLUMN, calculate_anomaly_score, ANOM_SCORE_COLUMN
from kad.model import i_model


class SarimaModel(i_model.IModel):
    def __init__(self, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int], train_valid_ratio=0.7):
        self.model = None
        self.model_results = None
        self.threshold: float = 0.0
        self.result_df = None
        self.order = order
        self.seasonal_order = seasonal_order
        self.train_valid_ratio = train_valid_ratio

    def train(self, train_df: pd.DataFrame):
        split_idx = int(self.train_valid_ratio * len(train_df))
        self.model = SARIMAX(train_df[:split_idx].values,
                             exog=None,
                             order=self.order,
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=True,
                             enforce_invertibility=False)

        self.model_results = self.model.fit()
        print(self.model_results.summary())

        samples_to_forecast = len(train_df) - split_idx
        forecast = self.model_results.forecast(samples_to_forecast)
        ground_truth = train_df[-samples_to_forecast:].to_numpy().flatten()
        resid = np.abs(forecast - ground_truth)
        self.model_results = self.model_results.append(ground_truth)

        self.result_df = train_df.copy()
        self.result_df.loc[-samples_to_forecast:, "predictions"] = forecast
        self.result_df.loc[-samples_to_forecast:, "residuals"] = resid
        self.result_df["is_anomaly"] = np.full(len(self.result_df), None)
        self.result_df[ANOMALIES_COLUMN] = np.full(len(self.result_df), False)
        self.result_df[ANOM_SCORE_COLUMN] = calculate_anomaly_score(self.result_df["residuals"])

        logging.debug("SARIMA anomaly threshold set to: " + str(self.threshold))

    def test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        forecast = self.model_results.forecast(len(test_df))
        resid = np.abs(forecast - test_df.to_numpy().flatten())

        if np.any(self.result_df.iloc[-len(forecast):][ANOMALIES_COLUMN]):
            self.model_results = self.model_results.append(forecast)
        else:
            self.model_results = self.model_results.append(test_df.values)

        temp_df = test_df.copy()
        temp_df[PREDICTIONS_COLUMN] = forecast

        self.result_df = pd.concat([self.result_df, temp_df])
        self.result_df.loc[-len(forecast):, "residuals"] = resid
        self.result_df.loc[-len(forecast):, ANOM_SCORE_COLUMN] = calculate_anomaly_score(self.result_df["residuals"])[
                                                                 -len(forecast):]
        self.result_df[ANOMALIES_COLUMN].iloc[-len(forecast):] = self.result_df.iloc[-len(forecast):][
                                                                     ANOM_SCORE_COLUMN].to_numpy().flatten() > 0.9

        return self.result_df
