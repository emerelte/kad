import pandas as pd
import datetime
from typing import Tuple


def metric_to_dataframe(metric: dict, metric_name: str):
    metric_timestamps = [datetime.datetime.utcfromtimestamp(t) for t, _ in metric["values"]]
    metric_values = [v for _, v in metric["values"]]
    return pd.DataFrame(metric_values, columns=[metric_name], index=metric_timestamps)


def split_dataset(original_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_ratio = 3.5/5
    return original_df[:int(len(original_df) * split_ratio)], original_df[
                                                        int(len(original_df) * split_ratio):]
