import unittest
import datetime
import pandas as pd
import numpy as np
from ts_analyzer import TsAnalyzer

NUM_DAYS: int = 1000


class TestTimeSeriesAnalyzer(unittest.TestCase):

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

