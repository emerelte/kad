import logging

import pandas as pd

from kad.data_sources.i_data_source import IDataSource, DataSourceException
from kad.data_processing import prom_wrapper, response_validator, metric_parser
import datetime


class PrometheusDataSource(IDataSource):

    def __init__(self, query: str, prom_url: str, metric_name: str, start_time: datetime.datetime,
                 stop_time: datetime.datetime, update_interval_sec: int):
        self.query = query
        self.prom_url = prom_url
        self.metric_name = metric_name
        self.start_time = start_time
        self.stop_time = stop_time
        self.update_interval_sec = update_interval_sec

        self.prom = prom_wrapper.PrometheusConnectWrapper(prom_url)

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
        metric_range = self.prom.perform_query(query=self.query,
                                               start_time=self.start_time,
                                               end_time=self.stop_time)

        metric = response_validator.validate(metric_range)
        train_df = metric_parser.metric_to_dataframe(metric, self.metric_name).astype(float)
        self.set_basic_timedelta(train_df)
        self.update_next_timestamp(train_df)

        return train_df

    def get_next_batch(self):
        metric_range = self.prom.perform_query(query=self.query,
                                               start_time=self.start_time,
                                               end_time=datetime.datetime.now())
        metric = response_validator.validate(metric_range)
        whole_df = metric_parser.metric_to_dataframe(metric, self.metric_name).astype(float)
        new_data = whole_df.loc[
                   self.next_timestamp:self.next_timestamp + datetime.timedelta(seconds=self.update_interval_sec)]

        if len(new_data) < 1:
            raise DataSourceException("No new data to fetch")

        self.update_next_timestamp(new_data)
        return new_data
