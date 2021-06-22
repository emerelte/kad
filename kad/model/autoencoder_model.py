import logging
import warnings
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kad.kad_utils import kad_utils
from matplotlib import pyplot as plt

from kad.kad_utils.kad_utils import calculate_anomaly_score, ANOM_SCORE_COLUMN, ERROR_COLUMN, ANOMALIES_COLUMN
from kad.model.i_model import IModel, ModelException


class AutoEncoderModel(IModel):

    def __init__(self, time_steps: int = kad_utils.TIME_STEPS, batch_size=12, learning_rate=0.001):
        if time_steps < batch_size:
            raise ModelException(
                f"Improper parameters for Autoencoder: time_steps({time_steps}) must be higher or equal than batch_size({batch_size})")

        self.error_threshold = None
        self.anomaly_score_threshold: float = 0.95
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.lr = learning_rate
        self.results_df = None
        self.x_train: np.ndarray = None
        self.nn = None

    @staticmethod
    def __calculate_threshold(valid_errors: np.ndarray) -> float:
        return 2 * np.max(valid_errors)

    def __initialize_nn(self, train_df: pd.DataFrame):
        self.results_df = train_df
        self.x_train, _ = kad_utils.embed_data(data=train_df.to_numpy().flatten(), steps=self.time_steps)
        self.nn = keras.Sequential(
            [
                layers.LSTM(128, input_shape=(self.x_train.shape[1], self.x_train.shape[2])),
                layers.Dropout(rate=0.2),
                layers.RepeatVector(self.x_train.shape[1]),
                layers.LSTM(128, return_sequences=True),
                layers.Dropout(rate=0.2),
                layers.TimeDistributed(layers.Dense(self.x_train.shape[2])),
            ]
        )
        self.nn.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss="mse")
        self.nn.summary()

    def train(self, train_df: pd.DataFrame):
        """
        @:param train_df: training data frame
        Takes training dataframe and:
            - fits the model
        """

        logging.info("Autoencoder model training started")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.__initialize_nn(train_df)

            validation_size = 0.1

            history = self.nn.fit(
                self.x_train,
                self.x_train,
                epochs=50,
                batch_size=self.batch_size,
                validation_split=validation_size,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
                ],
            )

            plt.figure()
            plt.plot(history.history["loss"], label="Training Loss")
            plt.plot(history.history["val_loss"], label="Validation Loss")
            plt.legend()
            plt.show()

            x_train_pred = self.nn.predict(self.x_train)
            train_mae_loss = np.mean(np.abs(x_train_pred - self.x_train), axis=1)

            original_indexes = kad_utils.calculate_original_indexes(len(self.x_train), self.time_steps)

            self.results_df[kad_utils.PREDICTIONS_COLUMN] = kad_utils.decode_data(x_train_pred, original_indexes)
            self.results_df[ERROR_COLUMN] = kad_utils.decode_data(train_mae_loss, original_indexes)
            self.results_df[ANOM_SCORE_COLUMN] = calculate_anomaly_score(self.results_df[ERROR_COLUMN])
            self.results_df[kad_utils.ANOMALIES_COLUMN] = np.full(len(self.results_df), False, dtype=bool)

            self.error_threshold = self.__calculate_threshold(
                train_mae_loss[-int(len(train_mae_loss) * 0.5):])

    def test(self, test_df: pd.DataFrame):
        """
        Appends a column to the df with classes
        """

        logging.info("Autoencoder tests!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.x_train is None or self.nn is None:
                raise ModelException("Model not trained, cannot test")

            # fixme magic number
            if len(test_df) < 5 * self.time_steps:
                raise ModelException(
                    "Autencoder should get at least 5*self.time_steps long data to give reasonable results")

            x_test, _ = kad_utils.embed_data(data=test_df.to_numpy().flatten(), steps=self.time_steps)
            original_indexes = kad_utils.calculate_original_indexes(len(x_test), self.time_steps)

            x_test_pred = self.nn.predict(x_test)
            test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1).reshape((-1))

            self.results_df = pd.concat([self.results_df, test_df.copy()])
            self.results_df.loc[:, kad_utils.PREDICTIONS_COLUMN].iloc[-len(test_df):] = \
                kad_utils.decode_data(x_test_pred, original_indexes)
            self.results_df.loc[-len(test_df):, ERROR_COLUMN] = kad_utils.decode_data(test_mae_loss, original_indexes)
            self.results_df.loc[-len(test_df):, ANOM_SCORE_COLUMN] = \
                calculate_anomaly_score(self.results_df[ERROR_COLUMN], self.error_threshold)[-len(test_df):]
            self.results_df.loc[:, ANOMALIES_COLUMN].iloc[-len(test_df):] = \
                np.any(self.results_df[kad_utils.ANOM_SCORE_COLUMN].iloc[
                       -len(test_df):].to_numpy().flatten() > self.anomaly_score_threshold)
            self.results_df[ANOMALIES_COLUMN] = self.results_df[ANOMALIES_COLUMN].astype("bool")

            logging.info("Autoencoder ended testing!")

            return self.results_df
