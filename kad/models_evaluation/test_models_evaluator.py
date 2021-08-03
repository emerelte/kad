import json
import os
import unittest

import numpy as np
import pandas as pd

from kad.kad_utils.kad_utils import GROUND_TRUTH_COLUMN, ANOMALIES_COLUMN
from kad.models_evaluation import models_evaluator


class TestScoringComponents(unittest.TestCase):
    def setUp(self) -> None:
        data_dir = "../../notebooks/data/archive/"
        file_dir = "artificialWithAnomaly"
        file_name = "artificialWithAnomaly/art_daily_flatmiddle.csv"

        file_path = os.path.join(data_dir, file_dir, file_name)

        original_df = pd.read_csv(
            file_path, parse_dates=True, index_col="timestamp"
        )

        original_df = original_df.resample("h").agg(np.mean)

        with open("../../notebooks/data/archive/combined_windows.json") as f:
            true_anomalies = json.load(f)

        true_anomalies_ranges = true_anomalies[file_name]

        ground_true_anomalies_df = pd.DataFrame()
        for anom_range in true_anomalies_ranges:
            ground_true_anomalies_df = ground_true_anomalies_df.append(original_df.loc[anom_range[0]:anom_range[1]])

        self.test_df = original_df.copy().reset_index()
        self.test_df[GROUND_TRUTH_COLUMN] = [idx in ground_true_anomalies_df.index for idx in
                                             original_df.index.tolist()]
        self.test_df[ANOMALIES_COLUMN] = False

        anomaly_window = self.test_df[self.test_df[GROUND_TRUTH_COLUMN]]

        self.anomaly_window_start = anomaly_window.iloc[0]
        self.anomaly_window_middle = anomaly_window.iloc[int(len(anomaly_window) / 2)]
        self.anomaly_window_end = anomaly_window.iloc[-1]

        self.sut = models_evaluator.ModelsEvaluator(self.test_df)


class TestFirstScoringComponent(TestScoringComponents):

    def test_return_1_if_alert_matches_anomaly(self):
        detected_anom_idx = self.sut.df.index[self.sut.df["timestamp"] == self.anomaly_window_middle["timestamp"]]
        self.sut.df.loc[detected_anom_idx, ANOMALIES_COLUMN] = True

        self.assertEqual(1.0, self.sut.calculate_first_scoring_component())

    def test_return_0_if_alert_on_the_edge_of_anom_window(self):
        detected_anom_idx = self.sut.df.index[self.sut.df["timestamp"] == self.anomaly_window_start["timestamp"]]
        self.sut.df.loc[detected_anom_idx, ANOMALIES_COLUMN] = True

        self.assertEqual(0.0, self.sut.calculate_first_scoring_component())

    def test_return_0_if_no_anomalies_found(self):
        self.assertEqual(0.0, self.sut.calculate_first_scoring_component())

    def test_return_1_if_one_of_the_alerts_on_the_edge_of_anom_window(self):
        detected_anom_idx = self.sut.df.index[self.sut.df["timestamp"] == self.anomaly_window_start["timestamp"]]
        self.sut.df.loc[detected_anom_idx, ANOMALIES_COLUMN] = True

        detected_anom_idx = self.sut.df.index[self.sut.df["timestamp"] == self.anomaly_window_middle["timestamp"]]
        self.sut.df.loc[detected_anom_idx, ANOMALIES_COLUMN] = True

        self.assertEqual(1.0, self.sut.calculate_first_scoring_component())

    def test_return_0_5_if_alert_matches_one_of_two_anomalies(self):
        self.test_df[GROUND_TRUTH_COLUMN] = False
        self.test_df.loc[10:20, GROUND_TRUTH_COLUMN] = True
        self.test_df.loc[30:40, GROUND_TRUTH_COLUMN] = True

        self.test_df.loc[15, ANOMALIES_COLUMN] = True

        self.sut = models_evaluator.ModelsEvaluator(self.test_df)

        self.assertEqual(0.5, self.sut.calculate_first_scoring_component())


class TestSecondScoringComponent(TestScoringComponents):
    def test_return_1_if_covered_whole_anomaly_window(self):
        self.sut.df.loc[:, ANOMALIES_COLUMN] = True

        self.assertEqual(1.0, self.sut.calculate_second_scoring_component())

    def test_return_0_5_if_covered_whole_first_anomaly_window(self):
        self.test_df[GROUND_TRUTH_COLUMN] = False
        self.test_df.loc[10:20, GROUND_TRUTH_COLUMN] = True
        self.test_df.loc[30:40, GROUND_TRUTH_COLUMN] = True

        self.test_df.loc[10:20, ANOMALIES_COLUMN] = True

        self.sut = models_evaluator.ModelsEvaluator(self.test_df)

        self.assertEqual(0.5, self.sut.calculate_second_scoring_component())

    def test_return_0_if_no_anomalies_found(self):
        self.assertEqual(0.0, self.sut.calculate_second_scoring_component())


class TestThirdScoringComponent(TestScoringComponents):
    def test_return_1_if_no_false_positives(self):
        self.assertEqual(1.0, self.sut.calculate_third_scoring_component())

    def test_return_0_if_all_false_positives(self):
        self.sut.df.loc[:, ANOMALIES_COLUMN] = True

        self.assertEqual(0.0, self.sut.calculate_third_scoring_component())

    def test_return_if_detection_between_anomaly_windows(self):
        self.test_df[GROUND_TRUTH_COLUMN] = False
        self.test_df.loc[10:20, GROUND_TRUTH_COLUMN] = True
        self.test_df.loc[30:40, GROUND_TRUTH_COLUMN] = True

        self.test_df.loc[25, ANOMALIES_COLUMN] = True

        self.sut = models_evaluator.ModelsEvaluator(self.test_df)

        second_scoring_comp = self.sut.calculate_third_scoring_component()
        self.assertLess(0.0, second_scoring_comp)
        self.assertGreater(1.0, second_scoring_comp)
