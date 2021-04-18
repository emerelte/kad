import io
import datetime
import logging
import os
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from matplotlib import pyplot as plt
import pandas as pd
from kad.data_processing import response_validator
from kad.data_sources import i_data_source
from kad.data_sources.exemplary_data_source import ExemplaryDataSource
from kad.data_sources.i_data_source import DataSourceException
from kad.data_sources.prom_data_source import PrometheusDataSource
from kad.kad_utils import kad_utils
from kad.kad_utils.kad_utils import EndpointAction
from kad.model import i_model
from kad.model.autoencoder_model import AutoEncoderModel
from kad.model.hmm_model import HmmModel
from kad.model.sarima_model import SarimaModel
from flask import Flask, send_file, Response
from flask_cors import cross_origin

from kad.visualization.visualization import visualize

PROMETHEUS_URL = "http://localhost:9090/"
PROM_METRIC_NAME = "rest_client_requests_total"
EXEMPLARY_METRIC_NAME = "value"
METRIC_NAME = EXEMPLARY_METRIC_NAME
# TODO remove hardcodes
START_TIME = datetime.datetime.strptime("2021-04-13 16:17:00", "%Y-%m-%d %H:%M:%S")
END_TIME = datetime.datetime.strptime("2021-04-13 18:17:00", "%Y-%m-%d %H:%M:%S")
# END_TIME = datetime.datetime.now()
# LABEL_CONFIG = {"code": "200", "host": "kind-control-plane:6443", "instance": "kind-control-plane",
#                 "job": "kubernetes-nodes", "method": "POST"}
# TODO dictionary 2 prometheus labels format
LABEL_CONFIG_STR = "{code=\"201\",host=\"kind-control-plane:6443\",instance=\"kind-control-plane\"," \
                   "job=\"kubernetes-nodes\", method=\"POST\"} "
QUERY = "rate(" + METRIC_NAME + LABEL_CONFIG_STR + "[1m])"

APP_URL = "http://localhost:5000"
UPDATE_DATA_ENDPOINT = "/update_data"
PLOT_RESULTS_ENDPOINT = "/plot_results"
UPDATE_INTERVAL_SEC = 1


def request_new_data():
    logging.debug("Requesting new data...")
    r = requests.get(APP_URL + UPDATE_DATA_ENDPOINT)

    if r.status_code != 200:
        logging.warning("Something went wrong when requesting data update")


class KAD(object):
    app = None

    def __init__(self):
        self.app = Flask("KAD app")

        file = "data/archive/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv"
        daily_jumpsup_csv_path = os.path.join(
            "/home/maciek/Documents/Magisterka/kubernetes-anomaly-detector/notebooks/",
            file)

        self.data_source: i_data_source = ExemplaryDataSource(
            path=daily_jumpsup_csv_path,
            metric_name=METRIC_NAME,
            start_time=datetime.datetime.strptime("2014-04-01 14:00:00", "%Y-%m-%d %H:%M:%S"),
            stop_time=datetime.datetime.strptime("2014-04-09 14:00:00", "%Y-%m-%d %H:%M:%S"),
            update_interval_hours=10)
        # self.data_source: i_data_source = PrometheusDataSource(query=QUERY, prom_url=PROMETHEUS_URL,
        #                                                        metric_name=METRIC_NAME,
        #                                                        start_time=START_TIME, stop_time=END_TIME,
        #                                                        update_interval_sec=UPDATE_INTERVAL_SEC)
        # self.model: i_model.IModel = SarimaModel(order=(0, 0, 0), seasonal_order=(1, 0, 1, 24))
        self.model: i_model.IModel = AutoEncoderModel(time_steps=12)
        # self.model: i_model.IModel = HmmModel()
        self.results_df: pd.DataFrame = None
        self.train_mean = None
        self.train_std = None

    def get_train_data(self) -> pd.DataFrame:
        train_df = self.data_source.get_train_data()
        self.train_mean = train_df.mean()
        self.train_std = train_df.std()
        return kad_utils.normalize(train_df, self.train_mean, self.train_std)

    def train_model(self, train_df: pd.DataFrame):
        self.model.train(train_df)
        if len(train_df) < 2:
            logging.warning("Almost empty training df (len < 2)")
        logging.debug("Model trained")

    def test(self, test_df):
        self.results_df = self.model.test(test_df)

    def get_latest_image(self):
        if self.results_df is None:
            logging.warning("Results not obtained yet")
            return None

        visualize(self.results_df, METRIC_NAME)

        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format="png")
        bytes_image.seek(0)
        return bytes_image

    def run(self):
        self.app.run(debug=True, threaded=True)

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None):
        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler))

    @cross_origin(supports_credentials=True)
    def plot_results(self):
        logging.debug("Results plot requested")
        bytes_obj = self.get_latest_image()

        if bytes_obj is None:
            logging.warning("Image empty!")
            return Response(status=404, headers={})

        return send_file(bytes_obj,
                         attachment_filename="plot.png",
                         mimetype="image/png")

    @cross_origin(supports_credentials=True)
    def update_data(self):
        logging.debug("Updating data")

        try:
            new_data = self.data_source.get_next_batch()
        except DataSourceException as dsexc:
            logging.warning(str(dsexc))
            return Response(status=200, headers={})  # TODO is 200 code ok here?

        if len(new_data) == 0:
            logging.warning("No new data has been obtained")
            return Response(status=200, headers={})

        new_data = kad_utils.normalize(new_data, self.train_mean, self.train_std)

        try:
            self.test(new_data)
            self.data_source.update_last_processed_timestamp()
        except Exception as exc:
            logging.warning("Impossible to test: " + str(exc))

        return Response(status=200, headers={})


if __name__ == "__main__":
    logging.basicConfig(filename="/tmp/kad.log", filemode="w", format="[%(levelname)s] %(filename)s:%(lineno)d: %("
                                                                      "message)s", level=logging.DEBUG)

    kad = KAD()

    try:
        train_df = kad.get_train_data()

        train_df[METRIC_NAME].plot()
        plt.show()

        kad.train_model(train_df)

        kad.add_endpoint(endpoint=PLOT_RESULTS_ENDPOINT, endpoint_name="plot_results", handler=kad.plot_results)
        kad.add_endpoint(endpoint=UPDATE_DATA_ENDPOINT, endpoint_name="update_data", handler=kad.update_data)

        scheduler = BackgroundScheduler()
        job = scheduler.add_job(request_new_data, "interval", minutes=UPDATE_INTERVAL_SEC / 60)
        scheduler.start()

        kad.run()

    except requests.exceptions.ConnectionError:
        logging.error("Could not perform query to Prometheus API")
    except response_validator.MetricValidatorException as exc:
        logging.error("Malformed metrics: " + str(exc))
    except DataSourceException as exc:
        logging.error("Too small training df: " + str(exc))
