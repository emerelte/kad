import pandas as pd

from kad.data_processing.i_data_processor import IDataProcessor


class Downsampler(IDataProcessor):
    def __init__(self, period: str, agg_method):
        self.period = period
        self.agg_method = agg_method

    def transform_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return input_df.resample(self.period).agg(self.agg_method)