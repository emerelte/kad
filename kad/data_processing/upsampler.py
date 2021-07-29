import pandas as pd

from kad.data_processing.i_data_processor import IDataProcessor


class Upsampler(IDataProcessor):
    def __init__(self, period: str):
        self.period = period

    def transform_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return input_df.resample(self.period).bfill()
