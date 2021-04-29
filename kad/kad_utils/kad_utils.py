from typing import List

import numpy as np
import pandas as pd
from flask import Response
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

PromQueryResponse = List[dict]

TIME_STEPS = 40


def get_dummy_data():
    daily_jumpsup_csv_path = "notebooks/data/archive/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv"

    original_df = pd.read_csv(
        daily_jumpsup_csv_path, parse_dates=True, index_col="timestamp"
    )

    original_df = original_df.resample("h").agg(np.mean)

    scaler = MinMaxScaler(feature_range=(-1, 0))
    original_df["value"] = scaler.fit_transform(original_df.values)

    first_training_samples = int(len(original_df) * 2 / 3)
    return original_df[:first_training_samples], original_df[first_training_samples:]


def embed_data(data: np.ndarray, steps: int):
    n = len(data)

    if steps >= n:
        raise Exception(f"Cannot embed data when steps({steps}) >= len(data)({n})")

    embedded_data = np.zeros((n - steps, steps))
    labels = data[steps:]

    for i in np.arange(steps, n):
        embedded_data[i - steps] = data[i - steps:i]

    return embedded_data[:, :, np.newaxis], labels[:, np.newaxis]


def create_sequences(values, time_steps=TIME_STEPS):
    output = []

    for i in range(len(values) - time_steps):
        output.append(values[i: (i + time_steps)])

    return np.stack(output)


def normalize(values: pd.DataFrame, mean: float, std: float):
    values -= mean
    values /= std
    return values


# ADF utils
def get_statistic_test(adf) -> float:
    return adf[0]


def get_pvalue(adf) -> float:
    return adf[1]


def get_critical_values(adf) -> dict:
    return adf[4]


def print_adf_results(adf):
    print("\nStatistics analysis\n")
    print("Statistic Test : ", get_statistic_test(adf))
    print("p-value : ", get_pvalue(adf))
    print("# n_lags : ", adf[2])
    print("No of observation: ", adf[3])
    for key, value in get_critical_values(adf).items():
        print(f" critical value {key} : {value}")


class EndpointAction(object):
    """
    Class for Flask endpoints handlers
    """

    def __init__(self, action):
        self.action = action
        self.response = Response(status=200, headers={})

    def __call__(self, *args):
        self.response = self.action()
        return self.response


def customize_matplotlib(color="white", labelsize=16, fontsize="xx-large"):
    plt.rcParams["xtick.color"] = color
    plt.rcParams["xtick.labelsize"] = labelsize
    plt.rcParams["ytick.color"] = color
    plt.rcParams["ytick.labelsize"] = labelsize
    plt.rcParams["axes.titlesize"] = labelsize
    plt.rcParams["axes.titlecolor"] = color
    plt.rcParams["axes.labelsize"] = labelsize
    plt.rcParams["axes.labelcolor"] = color
    plt.rcParams["legend.fontsize"] = fontsize
