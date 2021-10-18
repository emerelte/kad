import pandas as pd

from kad.data_processing.i_data_processor import IDataProcessor


class Ewm(IDataProcessor):
    def __init__(self, com: float = 0.5):
        self.com = com

    def transform_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return input_df.ewm(com=self.com).mean()
