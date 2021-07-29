from abc import ABC
import pandas as pd


class IDataProcessor(ABC):
    def transform_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        pass
