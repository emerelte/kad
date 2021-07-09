from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class ModelException(Exception):
    def __init__(self, message="Model Exception"):
        self.message = message

    def __str__(self):
        return f"{self.message}"


class IModel(ABC):

    def train(self, train_df: pd.DataFrame) -> float:
        pass

    def test(self, test_df: pd.DataFrame):
        pass
