from typing import List
import pandas as pd
import numpy as np

PromQueryResponse = List[dict]

TIME_STEPS = 40


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
