import logging

import numpy as np
import pandas as pd

from kad.data_processing.response_validator import MetricValidatorException
from kad.data_sources.i_data_source import IDataSource, DataSourceException
from kad.data_processing import prom_wrapper, response_validator, metric_parser
import datetime


class ExemplaryDataSource(IDataSource):

    def __init__(self, path: str, metric_name: str, start_time: datetime.datetime,
                 stop_time: datetime.datetime, update_interval_hours: int):
        if metric_name != "value":
            raise MetricValidatorException("No metrics found")

        self.metric_name = metric_name
        self.start_time = start_time
        self.stop_time = stop_time
        self.update_interval_hours = update_interval_hours

        self.original_df = pd.read_csv(
            path, parse_dates=True, index_col="timestamp"
        )
        self.original_df = self.original_df.resample("1h").agg(np.mean)

        self.last_processed_timestamp = None
        self.latest_timestamp = None
        self.basic_timedelta = None

    def update_latest_timestamp(self, df: pd.DataFrame):
        self.latest_timestamp = df.index[-1] + self.basic_timedelta

    def undo_next_timestamp(self):
        self.latest_timestamp = self.last_processed_timestamp

    def set_basic_timedelta(self, train_df: pd.DataFrame):
        # if len(train_df) < 2:
        #     raise DataSourceException("Cannot set timedelta when training df len is < 2")
        # last_timestamps = train_df[:2].index
        # self.basic_timedelta = last_timestamps[1] - last_timestamps[0]
        if train_df.index.freq is None:
            raise DataSourceException("Cannot set timedelta when data frame index has no freq")

        self.basic_timedelta = pd.to_timedelta(train_df.index.freq)

    def get_train_data(self) -> pd.DataFrame:
        train_df = self.original_df[self.start_time:self.stop_time]

        self.set_basic_timedelta(train_df)
        self.update_latest_timestamp(train_df)
        self.update_last_processed_timestamp()

        return train_df

    def get_next_batch(self):
        new_data = self.original_df.loc[
                   self.last_processed_timestamp:self.latest_timestamp + datetime.timedelta(
                       seconds=self.update_interval_hours * 60 * 60)]

        if new_data.empty:
            raise DataSourceException("No new data to fetch!")

        self.update_latest_timestamp(new_data)
        return new_data

    def update_last_processed_timestamp(self):
        self.last_processed_timestamp = self.latest_timestamp
