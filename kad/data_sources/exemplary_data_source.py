import logging

import numpy as np
import pandas as pd

from kad.data_sources.i_data_source import IDataSource, DataSourceException
from kad.data_processing import prom_wrapper, response_validator, metric_parser
import datetime


class ExemplaryDataSource(IDataSource):

    def __init__(self, path: str, metric_name: str, start_time: datetime.datetime,
                 stop_time: datetime.datetime, update_interval_hours: int):
        self.metric_name = metric_name
        self.start_time = start_time
        self.stop_time = stop_time
        self.update_interval_hours = update_interval_hours

        self.original_df = pd.read_csv(
            path, parse_dates=True, index_col="timestamp"
        )
        self.original_df = self.original_df.resample("h").agg(np.mean)

        self.next_timestamp = None
        self.basic_timedelta = None

    def update_next_timestamp(self, df: pd.DataFrame):
        self.next_timestamp = df.index[-1] + self.basic_timedelta

    def set_basic_timedelta(self, train_df: pd.DataFrame):
        if len(train_df) < 2:
            raise DataSourceException("Cannot set timedelta when training df len is < 2")
        last_timestamps = train_df[:2].index
        self.basic_timedelta = last_timestamps[1] - last_timestamps[0]

    def get_train_data(self) -> pd.DataFrame:
        train_df = self.original_df[self.start_time:self.stop_time]

        self.set_basic_timedelta(train_df)
        self.update_next_timestamp(train_df)

        return train_df

    def get_next_batch(self):
        new_data = self.original_df.loc[
                   self.next_timestamp:self.next_timestamp + datetime.timedelta(seconds=self.update_interval_hours*60*60)]

        self.update_next_timestamp(new_data)
        return new_data
