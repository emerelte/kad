import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import warnings

from sklearn.model_selection import train_test_split

from kad.kad_utils import kad_utils
from kad.kad_utils.kad_utils import calculate_anomaly_score, ANOM_SCORE_COLUMN
from kad.model import i_model


class HmmModel(i_model.IModel):

    def __init__(self, train_valid_ratio=0.7):
        super().__init__()
        self.model = GaussianHMM(n_components=2, covariance_type='tied', n_iter=1000)
        self.threshold: float = 0.0
        self.result_df = None
        self.train_valid_ratio = train_valid_ratio

    def train(self, train_df: pd.DataFrame) -> float:
        tr_df, valid_df = train_test_split(train_df, shuffle=False, train_size=self.train_valid_ratio)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model.fit(tr_df)

        train_states = self.model.predict(tr_df)
        train_samples = [self.model._generate_sample_from_state(s)[0] for s in train_states]

        residuals = tr_df.values.squeeze() - train_samples
        absolute_error = np.abs(residuals)
        self.threshold = np.max(absolute_error)

        self.result_df = train_df.copy()

        ground_truth = valid_df.to_numpy().flatten()
        valid_states = self.model.predict(valid_df)
        valid_samples = [self.model._generate_sample_from_state(s)[0] for s in valid_states]

        self.trained = True

        return kad_utils.calculate_validation_err(valid_samples, ground_truth)

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
            temp_df["residuals"] = absolute_error
            temp_df[ANOM_SCORE_COLUMN] = calculate_anomaly_score(temp_df["residuals"])

            self.result_df = pd.concat([self.result_df, temp_df])

            return self.result_df
