import pandas as pd

from kad.data_processing.i_data_processor import IDataProcessor
from typing import List


class CompositeDataProcessor(IDataProcessor):
    def __init__(self, data_processors: List[IDataProcessor]):
        self.data_processors = data_processors

    def transform_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        for processor in self.data_processors:
            input_df = processor.transform_data(input_df)
        return input_df
