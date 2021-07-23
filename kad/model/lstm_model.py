import logging
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from kad.kad_utils import kad_utils
from matplotlib import pyplot as plt

from kad.kad_utils.kad_utils import ANOM_SCORE_COLUMN, ANOMALIES_COLUMN, PREDICTIONS_COLUMN, ERROR_COLUMN, \
    calculate_anomaly_score
from kad.model.i_model import IModel, ModelException


class LstmModel(IModel):

    def __init__(self, time_steps: int = kad_utils.TIME_STEPS, batch_size=12, train_valid_ratio=0.7):
        super().__init__()
        if time_steps < batch_size:
            raise ModelException(
                f"Improper parameters for LstmModel: time_steps({time_steps}) must be higher or equal than batch_size({batch_size})")

        self.error_threshold = None
        self.anomaly_score_threshold: float = 0.95
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.results_df = None
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.nn = None
        self.train_valid_ratio = train_valid_ratio

    def __initialize_nn(self, train_df: pd.DataFrame):
        self.x_train, self.y_train = kad_utils.embed_data(data=train_df.to_numpy().flatten(), steps=self.time_steps)
        self.nn = keras.Sequential(
            [
                layers.LSTM(64, activation="relu", input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
                            return_sequences=True),
                layers.LSTM(64, activation="relu", return_sequences=False),
                layers.Dropout(rate=0.1),
                layers.Dense(self.y_train.shape[1]),
            ]
        )
        self.nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        self.nn.summary()

    def __update_threshold(self):
        # TODO dynamically update threshold during testing phase
        pass

    def __calculate_pred_and_err(self, x_data, y_data):
        predictions = self.nn.predict(x_data)
        mae_loss = np.mean(np.abs(predictions - y_data), axis=1)
        original_indexes = kad_utils.calculate_original_indexes(len(x_data), self.time_steps)

        return kad_utils.decode_data(predictions, original_indexes), kad_utils.decode_data(mae_loss, original_indexes)

    def train(self, train_df: pd.DataFrame) -> float:
        """
        Takes training dataframe as input and computes internal states that will be used to predict the test data classes
        """

        logging.info("LSTM TRAIN CALLED")

        tr_df, valid_df = train_test_split(train_df, shuffle=False, train_size=self.train_valid_ratio)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.__initialize_nn(tr_df)

            history = self.nn.fit(
                self.x_train,
                self.y_train,
                epochs=50,
                batch_size=self.batch_size,
                validation_split=0.5,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
                ],
            )

            plt.plot(history.history["loss"], label="Training Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.legend()
            plt.show()

            self.results_df = train_df.copy()

            train_predictions, train_mae = self.__calculate_pred_and_err(self.x_train, self.y_train)
            self.results_df.loc[:len(tr_df), PREDICTIONS_COLUMN] = train_predictions
            self.results_df.loc[:len(tr_df), ERROR_COLUMN] = train_mae

            x_valid, y_valid = kad_utils.embed_data(data=valid_df.to_numpy().flatten(), steps=self.time_steps)

            valid_predictions, valid_mae = self.__calculate_pred_and_err(x_valid, y_valid)
            self.results_df.loc[-len(valid_df):, PREDICTIONS_COLUMN] = valid_predictions
            self.results_df.loc[-len(valid_df):, ERROR_COLUMN] = valid_mae

            self.results_df[ANOM_SCORE_COLUMN] = calculate_anomaly_score(self.results_df[ERROR_COLUMN])
            self.results_df[ANOMALIES_COLUMN] = False

            self.error_threshold = np.max(train_mae)

            self.trained = True

            return kad_utils.calculate_validation_err(valid_predictions, valid_df.to_numpy().flatten())

    def test(self, test_df: pd.DataFrame):
        """
        Appends a column to the df with classes
        """

        logging.info("LSTM Model tests!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.x_train is None or self.nn is None:
                raise ModelException("Model not trained, cannot test")

            x_test, y_test = kad_utils.embed_data(data=test_df.to_numpy().flatten(), steps=self.time_steps)

            test_pred, test_err = self.__calculate_pred_and_err(x_test, y_test)

            anomalies = test_err > self.error_threshold
            for anom_idx in np.where(anomalies)[0]:
                logging.info(f"Anomaly detected at idx: {anom_idx}. Forecasting error: {test_err[anom_idx]}")

            self.results_df = pd.concat([self.results_df, test_df.copy()])
            self.results_df.loc[-len(test_df):, kad_utils.PREDICTIONS_COLUMN] = test_pred
            self.results_df.loc[-len(test_df):, ERROR_COLUMN] = test_err

            self.results_df.loc[-len(test_df):, ANOM_SCORE_COLUMN] = \
                calculate_anomaly_score(self.results_df[ERROR_COLUMN], self.error_threshold)[-len(test_df):]

            self.results_df.loc[:, ANOMALIES_COLUMN].iloc[-len(test_df):] = \
                np.any(self.results_df[kad_utils.ANOM_SCORE_COLUMN].iloc[
                       -len(test_df):].to_numpy().flatten() > self.anomaly_score_threshold)
            self.results_df[ANOMALIES_COLUMN] = self.results_df[ANOMALIES_COLUMN].astype("bool")

            self.__update_threshold()

            logging.info("LSTM Model ended testing!")

            return self.results_df
