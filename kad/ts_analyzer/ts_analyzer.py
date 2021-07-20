import concurrent.futures
import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from statsmodels.tsa.stattools import adfuller

import kad.model.i_model
from kad.kad_utils import kad_utils
from kad.model import autoencoder_model, sarima_model, hmm_model, lstm_model

executor = concurrent.futures.ProcessPoolExecutor()


def task_executor(model: kad.model.i_model.IModel, data: pd.DataFrame):
    return model.__class__.__name__, model.train(data)


class TsAnalyzerException(Exception):
    def __init__(self, message="TS Analyzer Exception"):
        self.message = message

    def __str__(self):
        return f"{self.message}"


class TsAnalyzer:
    """
    Ts analyzer provides a set of methods describing characteristics of the data
    """

    def __init__(self, ts: pd.DataFrame):
        """
        :param ts: pd.Dataframe which has one column
        """
        required_num_cols = 1
        num_cols = len(ts.columns)
        if num_cols != required_num_cols:
            raise Exception(
                f"Improper number of columns passed to TsAnalyzer constr: {num_cols}, whereas it should be {required_num_cols}")

        self.data = ts

    def select_model(self):
        validErrByModel: Dict[str, Tuple[kad.model.i_model.IModel, float]] = {
            "AutoEncoderModel": (autoencoder_model.AutoEncoderModel(), None),
            "HmmModel": (hmm_model.HmmModel(), None),
            "LstmModel": (lstm_model.LstmModel(), None),
            "SarimaModel": (sarima_model.SarimaModel(order=(0, 0, 0),
                                                     seasonal_order=(
                                                         1, 0, 1,
                                                         self.calculate_dominant_frequency())), None)}

        futures_table = list()

        for model in validErrByModel.values():
            futures_table.append(
                executor.submit(task_executor, model[0], self.data))

        for future in futures_table:
            result: Tuple[str, float] = future.result()
            validErrByModel[result[0]] = (validErrByModel[result[0]][0], result[1])

        return min(validErrByModel.items(), key=lambda elem: elem[1][1])[1][1]

    def is_stationary(self):
        # print(self.data.head())

        # decompose = seasonal_decompose(self.data.resample('M').sum(), period=12)
        # fig = plt.figure()
        # fig = decompose.plot()
        # fig.set_size_inches(12, 8)
        # plt.show()

        adf = adfuller(self.data.iloc[:, 0], 12)
        kad_utils.print_adf_results(adf)

        significance = 0.05

        return kad_utils.get_statistic_test(adf) < kad_utils.get_critical_values(adf)["1%"] \
               and kad_utils.get_pvalue(adf) < significance

    # todo fix accuracy when not much data
    def calculate_dominant_frequency(self):
        N = len(self.data)
        T = 1
        divisor = 2

        x_fft = fftfreq(N, T)[:N // divisor]
        y_fft = fft(self.data.to_numpy().flatten())
        y_fft = np.abs(y_fft[0:N // divisor])

        # fig = plt.figure(figsize=(20, 10))
        # ax = fig.add_subplot(111)
        # plt.plot(x_fft, 2.0 / N * y_fft)
        #
        # plt.title("FFT")
        # ax.tick_params(axis="x", labelsize=22)
        # ax.tick_params(axis="y", labelsize=22)
        # plt.show()

        return round(1 / (T * x_fft[np.argsort(-y_fft)[1]]))
