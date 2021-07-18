import os
import unittest
from unittest.mock import patch, Mock
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ts_analyzer import TsAnalyzer

NUM_DAYS: int = 1000


class TestTimeSeriesAnalyzerArtificialData(unittest.TestCase):

    def setUp(self) -> None:
        base = datetime.datetime.today()
        self.date_list = [base + datetime.timedelta(days=x) for x in range(NUM_DAYS)]

    def test_is_stationary_should_return_true_given_white_noise(self):
        white_noise_arr: np.ndarray = np.random.normal(0, 1, NUM_DAYS)
        stationary_df = pd.DataFrame(data=white_noise_arr, index=self.date_list)

        self.assertTrue(TsAnalyzer(stationary_df).is_stationary())

    def test_is_stationary_should_return_false_given_random_walk(self):
        step_set = [-1, 0, 1]
        origin = np.zeros((1, 1))

        step_shape = (NUM_DAYS - 1, 1)
        steps = np.random.choice(a=step_set, size=step_shape)
        random_walk_arr = np.concatenate([origin, steps]).cumsum(0)
        random_walk_df = pd.DataFrame(data=random_walk_arr, index=self.date_list)

        self.assertFalse(TsAnalyzer(random_walk_df).is_stationary())

    def test_calculate_dominant_frequency(self):
        t_start = 0
        t_stop = 400 * np.pi
        num_samples = NUM_DAYS
        sin_T = 2 * np.pi
        expected_freq = round(num_samples / ((t_stop - t_start) / sin_T))
        t = np.linspace(t_start, t_stop, num_samples)
        sin_sig = np.sin(t)
        sin_df = pd.DataFrame(data=sin_sig, index=self.date_list).asfreq("d")

        self.assertEqual(expected_freq, TsAnalyzer(sin_df).calculate_dominant_frequency())


class TestTimeSeriesAnalyzerModelSelection(unittest.TestCase):

    def setUp(self) -> None:
        data_dir = "../../notebooks/data/archive/"
        file_dir = "artificialWithAnomaly"
        file_name = "artificialWithAnomaly/art_daily_flatmiddle.csv"

        file_path = os.path.join(data_dir, file_dir, file_name)\

        df = pd.read_csv(
            file_path, parse_dates=True, index_col="timestamp"
        ).resample("h").agg(np.mean)

        scaler = MinMaxScaler(feature_range=(-1, 0))
        df["value"] = scaler.fit_transform(df.values)

        self.sut: TsAnalyzer = TsAnalyzer(df[["value"]])

    @patch('kad.model.sarima_model.SarimaModel')
    @patch('kad.model.hmm_model.HmmModel')
    @patch('kad.model.lstm_model.LstmModel')
    @patch('kad.model.autoencoder_model.AutoEncoderModel')
    def test_select_model_with_lowest_valid_err(self, MockAutoencoder, LstmModel, HmmModel, MockSarima):
        mock_autoencoder = MockAutoencoder.return_value
        mock_autoencoder.train.return_value = 1.0

        mock_lstm = LstmModel.return_value
        mock_lstm.train.return_value = 1.6

        mock_hmm = HmmModel.return_value
        mock_hmm.train.return_value = 1.6

        mock_sarima = MockSarima.return_value
        mock_sarima.train.return_value = 2.0

        self.assertEqual(mock_autoencoder, self.sut.select_model())

    def test_select_model_on_real_data(self):
        self.sut.select_model()
