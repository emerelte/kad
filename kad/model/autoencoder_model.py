import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from kad.kad_utils import kad_utils
from matplotlib import pyplot as plt

from kad.kad_utils.kad_utils import calculate_anomaly_score, ANOM_SCORE_COLUMN, ERROR_COLUMN, ANOMALIES_COLUMN, \
    PREDICTIONS_COLUMN
from kad.model.i_model import IModel, ModelException


class AutoEncoderModel(IModel):

    def __init__(self, time_steps: int = kad_utils.TIME_STEPS, batch_size=12, learning_rate=0.001,
                 train_valid_ratio=0.7):
        super().__init__()
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
        self.train_valid_ratio = train_valid_ratio
        self.nn = None

    @staticmethod
    def __calculate_threshold(valid_errors: np.ndarray) -> float:
        return 2 * np.max(valid_errors)

    def __initialize_nn(self, train_df: pd.DataFrame):
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

    def __calculate_pred_and_err(self, data):
        predictions = self.nn.predict(data)
        mae_loss = np.mean(np.abs(predictions - data), axis=1)
        original_indexes = kad_utils.calculate_original_indexes(len(data), self.time_steps)

        return kad_utils.decode_data(predictions, original_indexes), kad_utils.decode_data(mae_loss, original_indexes)

    def train(self, train_df: pd.DataFrame) -> float:
        """
        @:param train_df: training data frame
        Takes training dataframe and:
            - fits the model
        @:returns validation error
        """

        logging.info("Autoencoder model training started")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            tr_df, valid_df = train_test_split(train_df, shuffle=False, train_size=self.train_valid_ratio)

            self.__initialize_nn(tr_df)

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

            self.results_df = train_df.copy()

            train_predictions, train_mae = self.__calculate_pred_and_err(self.x_train)
            self.results_df.loc[:len(tr_df), kad_utils.PREDICTIONS_COLUMN] = train_predictions
            self.results_df.loc[:len(tr_df), ERROR_COLUMN] = train_mae

            x_valid, _ = kad_utils.embed_data(data=valid_df.to_numpy().flatten(), steps=self.time_steps)

            valid_predictions, valid_mae = self.__calculate_pred_and_err(x_valid)
            self.results_df.loc[-len(valid_df):, PREDICTIONS_COLUMN] = valid_predictions
            self.results_df.loc[-len(valid_df):, ERROR_COLUMN] = valid_mae

            self.results_df[ANOM_SCORE_COLUMN] = calculate_anomaly_score(self.results_df[ERROR_COLUMN])
            self.results_df[kad_utils.ANOMALIES_COLUMN] = False

            self.error_threshold = self.__calculate_threshold(train_mae[-int(len(train_mae) * 0.5):])

            self.trained = True

            return kad_utils.calculate_validation_err(valid_predictions, valid_df.to_numpy().flatten())

    def test(self, test_df: pd.DataFrame):
        """
        Appends a column to the df with classes
        """

        logging.info("Autoencoder tests!")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.x_train is None or self.nn is None:
                raise ModelException("Model not trained, cannot test")

            if len(test_df) < 5 * self.time_steps:
                raise ModelException(
                    f"Autencoder should get at least 5*self.time_steps long data to give reasonable results, got {len(test_df)}")

            x_test, _ = kad_utils.embed_data(data=test_df.to_numpy().flatten(), steps=self.time_steps)

            test_pred, test_err = self.__calculate_pred_and_err(x_test)

            self.results_df = pd.concat([self.results_df, test_df.copy()])
            self.results_df.loc[-len(test_df):, kad_utils.PREDICTIONS_COLUMN] = test_pred
            self.results_df.loc[-len(test_df):, ERROR_COLUMN] = test_err

            self.results_df.loc[-len(test_df):, ANOM_SCORE_COLUMN] = \
                calculate_anomaly_score(self.results_df[ERROR_COLUMN], self.error_threshold)[-len(test_df):]

            self.results_df.loc[:, ANOMALIES_COLUMN].iloc[-len(test_df):] = \
                np.any(self.results_df[kad_utils.ANOM_SCORE_COLUMN].iloc[
                       -len(test_df):].to_numpy().flatten() > self.anomaly_score_threshold)
            self.results_df[ANOMALIES_COLUMN] = self.results_df[ANOMALIES_COLUMN].astype("bool")

            logging.info("Autoencoder ended testing!")

            return self.results_df
