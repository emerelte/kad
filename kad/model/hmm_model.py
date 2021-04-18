import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings

from kad.model import i_model


class HmmModel(i_model.IModel):

    def __init__(self):
        self.model = GaussianHMM(n_components=2, covariance_type='tied', n_iter=1000)
        self.threshold: float = 0.0
        self.result_df = None

    def train(self, train_df: pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model.fit(train_df)

        train_states = self.model.predict(train_df)
        train_samples = [self.model._generate_sample_from_state(s)[0] for s in train_states]

        residuals = train_df.values.squeeze() - train_samples
        absolute_error = np.abs(residuals)
        self.threshold = np.max(absolute_error)

        self.result_df = train_df.copy()

    def test(self, test_df: pd.DataFrame) -> pd.DataFrame:
        if self.result_df is None:
            raise i_model.ModelException("Model not trained, cannot test")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states = self.model.predict(test_df)
            samples = [self.model._generate_sample_from_state(s)[0] for s in states]

            residuals = test_df.values.squeeze() - samples
            absolute_error = np.abs(residuals)
            is_anomaly = absolute_error > self.threshold

            temp_df = test_df.copy()
            temp_df["predictions"] = samples
            temp_df["is_anomaly"] = is_anomaly

            self.result_df = pd.concat([self.result_df, temp_df])

            return self.result_df
