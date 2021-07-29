from abc import ABC
import pandas as pd


class ModelException(Exception):
    def __init__(self, message="Model Exception"):
        self.message = message

    def __str__(self):
        return f"{self.message}"


class IModel(ABC):
    def __init__(self):
        self.trained: bool = False

    def is_trained(self) -> bool:
        return self.trained

    def train(self, train_df: pd.DataFrame) -> float:
        pass

    def test(self, test_df: pd.DataFrame):
        pass
