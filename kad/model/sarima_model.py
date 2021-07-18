import logging
import warnings
from typing import Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
import kad.kad_utils.kad_utils as kad_utils
from kad.model import i_model


class SarimaModel(i_model.IModel):
    def __init__(self, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int], train_valid_ratio=0.7):
        self.model = None
        self.model_results = None
        self.error_threshold: float = 0.0
        self.anomaly_score_threshold: float = 0.95
        self.results_df = None
        self.order = order
        self.seasonal_order = seasonal_order
        self.train_valid_ratio = train_valid_ratio

    @staticmethod
    def __calculate_threshold(valid_errors: np.ndarray) -> float:
        return 2 * np.max(valid_errors)

    def train(self, train_df: pd.DataFrame) -> float:
        """
        @:param train_df: training data frame
        Takes training dataframe and:
            - fits the model using [:self.train_valid_ratio] part of the passed dataframe
            - using the fitted model predicts [self.train_valid_ratio:] part of the passed dataframe and stores:
                a) predictions
                b) forecast error
                c) threshold set to 2*max(forecast error) which is used in testing part to calculate the anomaly score
        @:returns validation error
        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            tr_df, valid_df = train_test_split(train_df, shuffle=False, train_size=self.train_valid_ratio)
            self.model = SARIMAX(tr_df.values,
                                 exog=None,
                                 order=self.order,
                                 seasonal_order=self.seasonal_order,
                                 enforce_stationarity=True,
                                 enforce_invertibility=False)
            self.model_results = self.model.fit()
            # print(self.model_results.summary())

            forecast: np.ndarray = self.model_results.forecast(len(valid_df))
            ground_truth = valid_df.to_numpy().flatten()
            self.model_results = self.model_results.append(ground_truth)

            abs_error = np.abs(forecast - ground_truth)
            self.error_threshold = self.__calculate_threshold(abs_error)

            self.results_df = train_df.copy()
            self.results_df[kad_utils.PREDICTIONS_COLUMN] = np.full(len(self.results_df), None)
            self.results_df[kad_utils.ERROR_COLUMN] = np.full(len(self.results_df), None)
            self.results_df[kad_utils.ANOM_SCORE_COLUMN] = np.full(len(self.results_df), None)

            self.results_df[kad_utils.ANOMALIES_COLUMN] = np.full(len(self.results_df), False)
            self.results_df.loc[:, kad_utils.PREDICTIONS_COLUMN].iloc[-len(valid_df):] = forecast
            self.results_df.loc[:, kad_utils.ERROR_COLUMN].iloc[-len(valid_df):] = abs_error

            logging.info("SARIMA anomaly threshold set to: " + str(self.error_threshold))
            return kad_utils.calculate_validation_err(forecast, ground_truth)

    def test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            forecast = self.model_results.forecast(len(test_df))
            abs_error = np.abs(forecast - test_df.to_numpy().flatten())

            self.results_df = pd.concat([self.results_df, test_df.copy()])
            self.results_df.loc[:, kad_utils.PREDICTIONS_COLUMN].iloc[-len(forecast):] = forecast
            self.results_df.loc[:, kad_utils.ERROR_COLUMN].iloc[-len(forecast):] = abs_error
            self.results_df.loc[:, kad_utils.ANOM_SCORE_COLUMN].iloc[-len(forecast):] = kad_utils.calculate_anomaly_score(
                self.results_df[kad_utils.ERROR_COLUMN], self.error_threshold)[-len(forecast):]
            self.results_df.loc[:, kad_utils.ANOMALIES_COLUMN].iloc[-len(forecast):] = \
                np.any(self.results_df[kad_utils.ANOM_SCORE_COLUMN].iloc[-len(forecast):]
                       .to_numpy().flatten() >= self.anomaly_score_threshold)
            self.results_df[kad_utils.ANOMALIES_COLUMN] = self.results_df[kad_utils.ANOMALIES_COLUMN].astype("bool")

            if np.any(self.results_df.iloc[-len(forecast):][kad_utils.ANOMALIES_COLUMN]):
                self.model_results = self.model_results.append(forecast)
            else:
                self.model_results = self.model_results.append(test_df.values)

            return self.results_df
