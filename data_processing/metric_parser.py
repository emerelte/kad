import pandas as pd
import datetime


def metric_to_dataframe(metric: dict, metric_name: str):
    metric_timestamps = [datetime.datetime.utcfromtimestamp(t) for t, _ in metric["values"]]
    metric_values = [v for _, v in metric["values"]]
    return pd.DataFrame(metric_values, columns=[metric_name], index=metric_timestamps)
