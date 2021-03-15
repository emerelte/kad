from data_processing import prom_wrapper, response_validator, metric_parser
from model import i_model, autoencoder_model
from visualization import visualization
from utils import utils
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import prometheus_client
import prometheus_api_client as prom_api
import requests
import datetime
from prometheus_client.parser import text_string_to_metric_families

PROMETHEUS_URL = "http://localhost:9090/"
METRIC_NAME = "rest_client_requests_total"
# TODO remove hardcodes
START_TIME = datetime.datetime.strptime("2021-03-15 17:31:00", "%Y-%m-%d %H:%M:%S")
END_TIME = datetime.datetime.strptime("2021-03-15 17:46:00", "%Y-%m-%d %H:%M:%S")
# LABEL_CONFIG = {"code": "200", "host": "kind-control-plane:6443", "instance": "kind-control-plane",
#                 "job": "kubernetes-nodes", "method": "PATCH"}
# TODO dictionary 2 prometheus labels format
LABEL_CONFIG_STR = "{code=\"200\",host=\"kind-control-plane:6443\",instance=\"kind-control-plane\"," \
                   "job=\"kubernetes-nodes\",method=\"PATCH\"} "
QUERY = "rate(" + METRIC_NAME + LABEL_CONFIG_STR + "[1m])"


class KAD:

    def __init__(self):
        self.prom = prom_wrapper.PrometheusConnectWrapper(PROMETHEUS_URL)
        self.model: i_model.IModel = autoencoder_model.AutoEncoderModel()

    def main(self):
        metric_range = self.prom.perform_query(query=QUERY,
                                               start_time=START_TIME,
                                               end_time=END_TIME)

        metric = response_validator.validate(metric_range)
        metric_df = metric_parser.metric_to_dataframe(metric, METRIC_NAME).astype(float)
        train_df, test_df = metric_parser.split_dataset(metric_df)

        test_df = utils.normalize(test_df, train_df.mean(), train_df.std())
        train_df = utils.normalize(train_df, train_df.mean(), train_df.std())

        train_df[METRIC_NAME].plot()
        test_df[METRIC_NAME].plot()
        plt.show()

        self.model.train(train_df)
        result_df = self.model.test(test_df)

        visualization.visualize(result_df, METRIC_NAME)


if __name__ == "__main__":
    kad = KAD()
    kad.main()
