import logging
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from kad.kad_utils import kad_utils
from matplotlib import pyplot as plt

from kad.kad_utils.kad_utils import ANOM_SCORE_COLUMN, ANOMALIES_COLUMN, PREDICTIONS_COLUMN, ERROR_COLUMN, \
    calculate_anomaly_score
from kad.model.i_model import IModel, ModelException


class LstmModel(IModel):

    def __init__(self, time_steps: int = kad_utils.TIME_STEPS, batch_size=12):
        if time_steps < batch_size:
            raise ModelException(
                f"Improper parameters for LstmModel: time_steps({time_steps}) must be higher or equal than batch_size({batch_size})")

        self.threshold = None
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.results_df = None
        self.x_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.nn = None

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

    def train(self, train_df: pd.DataFrame):
        """
        Takes training dataframe as input and computes internal states that will be used to predict the test data classes
        """

        logging.info("LSTM TRAIN CALLED")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.__initialize_nn(train_df)

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

            forecast = self.nn.predict(self.x_train)
            train_mae_loss = np.mean(np.abs(forecast - self.y_train), axis=1)

            original_indexes = kad_utils.calculate_original_indexes(len(self.x_train), self.time_steps)

            self.threshold = np.max(train_mae_loss)
            self.results_df = train_df.copy()
            self.results_df.loc[-len(forecast):, PREDICTIONS_COLUMN] = forecast
            self.results_df[ANOMALIES_COLUMN] = False
            self.results_df[ERROR_COLUMN] = kad_utils.decode_data(train_mae_loss, original_indexes)
            self.results_df[ANOM_SCORE_COLUMN] = calculate_anomaly_score(self.results_df[ERROR_COLUMN])

    def test(self, test_df: pd.DataFrame):
        """
        Appends a column to the df with classes
        """

        logging.info("LSTM Model tests!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.x_train is None or self.nn is None:
                raise ModelException("Model not trained, cannot test")

            extended_test_df = np.concatenate((self.results_df[test_df.columns[0]].to_numpy()[-self.time_steps:],
                                               test_df.to_numpy().flatten()))

            x_test, y_test = kad_utils.embed_data(data=extended_test_df, steps=self.time_steps)

            forecast = self.nn.predict(x_test)
            residuals = test_df.values.squeeze() - forecast.flatten()
            absolute_error = np.abs(residuals)

            anomalies = absolute_error > self.threshold
            for anom_idx in np.where(anomalies)[0]:
                logging.info(f"Anomaly detected at idx: {anom_idx}. Forecasting error: {absolute_error[anom_idx]}")

            temp_df = test_df.copy()
            temp_df["is_anomaly"] = anomalies
            temp_df.loc[-len(forecast):, "residuals"] = absolute_error
            temp_df.loc[-len(forecast):, "predictions"] = forecast

            self.results_df = pd.concat([self.results_df, temp_df])
            self.results_df.loc[:, ANOM_SCORE_COLUMN].iloc[-len(forecast):] = kad_utils.calculate_anomaly_score(
                self.results_df["residuals"])[-len(forecast):]

            self.__update_threshold()

            logging.info("LSTM Model ended testing!")

            return self.results_df
