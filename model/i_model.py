from abc import ABC, abstractmethod
import pandas as pd


class IModel(ABC):
    def train(self, train_df: pd.DataFrame):
        pass

    def test(self, test_df: pd.DataFrame):
        pass
