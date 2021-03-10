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
