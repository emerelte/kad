import logging
from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

        self.result_df = train_df.copy()
        self.result_df["predictions"] = np.full(len(self.result_df), None)
        self.result_df["pred_err"] = np.full(len(self.result_df), None)
        self.result_df["is_anomaly"] = np.full(len(self.result_df), None)

        # TODO think of a better way of choosing threshold
        samples_to_forecast = len(train_df) - split_idx
        ground_truth = train_df[-samples_to_forecast:].values.squeeze()
        forecast = self.model_results.forecast(samples_to_forecast)
        self.model_results = self.model_results.append(ground_truth)
        absolute_error = np.abs(ground_truth - forecast)
        self.threshold = max(np.max(absolute_error), self.threshold)

        logging.debug("SARIMA anomaly threshold set to: " + str(self.threshold))

    def test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        forecast = self.model_results.forecast(len(test_df))
        residuals = test_df.values.squeeze() - forecast
        absolute_error = np.abs(residuals)

        anomalies = absolute_error > self.threshold
        for anom_idx in np.where(anomalies)[0]:
            logging.debug(f"Anomaly detected at idx: {anom_idx}. Forecasting error: {absolute_error[anom_idx]}")
        temp_df = test_df.copy()
        temp_df["predictions"] = forecast
        temp_df["pred_err"] = absolute_error
        temp_df["is_anomaly"] = anomalies

        self.result_df = pd.concat([self.result_df, temp_df])

        if np.any(anomalies):
            self.model_results = self.model_results.append(forecast)
        else:
            self.model_results = self.model_results.append(test_df.values)

        return self.result_df
