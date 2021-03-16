import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from kad_utils import kad_utils
from matplotlib import pyplot as plt
from model.i_model import IModel


# TODO add base class and implement useful models
class AutoEncoderModel(IModel):

    def __init__(self):
        self.time_steps = kad_utils.TIME_STEPS
        self.x_train = None
        self.nn = None

    def initialize_nn(self, train_df: pd.DataFrame):
        self.x_train = kad_utils.create_sequences(train_df.values, self.time_steps)
        self.nn = keras.Sequential(
            [
                layers.Input(shape=(self.x_train.shape[1], self.x_train.shape[2])),
                layers.Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Dropout(rate=0.2),
                layers.Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
            ]
        )
        self.nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        self.nn.summary()
        self.threshold = None

    """
    Takes training dataframe as input and computes internal states that will be used to predict the test data classes
    """

    def train(self, train_df: pd.DataFrame):
        self.initialize_nn(train_df)

        history = self.nn.fit(
            self.x_train,
            self.x_train,
            epochs=50,
            batch_size=12,
            validation_split=0.1,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
            ],
        )

        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.show()

        x_train_pred = self.nn.predict(self.x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - self.x_train), axis=1)
        self.threshold = np.max(train_mae_loss)

    """
    Appends a column to the df with classes
    """

    def test(self, test_df: pd.DataFrame):

        if self.x_train is None or self.nn is None:
            raise Exception("Model not trained, cannot test")

        x_test = kad_utils.create_sequences(test_df.values)
        x_test_pred = self.nn.predict(x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))

        anomalies = test_mae_loss > self.threshold

        anomalous_data_indices = []
        for data_idx in range(self.time_steps - 1, len(test_df) - self.time_steps + 1):
            if np.any(anomalies[data_idx - self.time_steps + 1: data_idx]):
                anomalous_data_indices.append(data_idx)

        result_df = test_df.copy()
        is_anomaly = [i in anomalous_data_indices for i in range(len(result_df))]
        result_df["is_anomaly"] = is_anomaly
        return result_df
