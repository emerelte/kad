import io
import datetime
import logging
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from matplotlib import pyplot as plt
import pandas as pd
from data_processing import prom_wrapper, response_validator, metric_parser
from kad_utils import kad_utils
from kad_utils.kad_utils import EndpointAction
from model import i_model, autoencoder_model
from model.sarima_model import SarimaModel
from visualization import visualization
from flask import Flask, send_file, Response
from flask_cors import cross_origin

PROMETHEUS_URL = "http://localhost:9090/"
METRIC_NAME = "rest_client_requests_total"
# TODO remove hardcodes
START_TIME = datetime.datetime.strptime("2021-03-16 19:00:00", "%Y-%m-%d %H:%M:%S")
END_TIME = datetime.datetime.strptime("2021-03-16 19:46:00", "%Y-%m-%d %H:%M:%S")
# LABEL_CONFIG = {"code": "200", "host": "kind-control-plane:6443", "instance": "kind-control-plane",
#                 "job": "kubernetes-nodes", "method": "PATCH"}
# TODO dictionary 2 prometheus labels format
LABEL_CONFIG_STR = "{code=\"200\",host=\"kind-control-plane:6443\",instance=\"kind-control-plane\"," \
                   "job=\"kubernetes-nodes\", method=\"PATCH\"} "
QUERY = "rate(" + METRIC_NAME + LABEL_CONFIG_STR + "[1m])"

APP_URL = "http://localhost:5000"
UPDATE_DATA_ENDPOINT = "/update_data"
PLOT_RESULTS_ENDPOINT = "/plot_results"
UPDATE_INTERVAL_SEC = 5


def request_new_data():
    logging.debug("Requesting new data...")
    r = requests.get(APP_URL + UPDATE_DATA_ENDPOINT)
    if r.status_code != 200:
        logging.warning("Something went wrong when requesting data update")


class KAD(object):
    app = None
    next_timestamp = None

    def __init__(self):
        self.app = Flask("KAD app")

        self.prom = prom_wrapper.PrometheusConnectWrapper(PROMETHEUS_URL)
        self.model: i_model.IModel = SarimaModel(order=(0, 0, 0), seasonal_order=(1, 0, 1, 24))
        self.results_df: pd.DataFrame = None

    def update_next_timestamp(self, df: pd.DataFrame):
        last_timestamps = df[-2:].index
        basic_timedelta = last_timestamps[1] - last_timestamps[0]
        self.next_timestamp = last_timestamps[1] + basic_timedelta

    def train_model(self, train_df: pd.DataFrame):
        self.model.train(train_df)
        if len(train_df) < 2:
            logging.warning("Almost empty training df (len < 2)")
            return
        self.update_next_timestamp(train_df)
        logging.debug("Model trained")

    def test(self, test_df):
        self.results_df = self.model.test(test_df)

    def get_latest_image(self):
        if self.results_df is None:
            logging.warning("Results not obtained yet")
            return None

        fig, ax = plt.subplots()
        anomalies_df = self.results_df.loc[self.results_df["is_anomaly"] == True]
        self.results_df["value"].plot(ax=ax)
        self.results_df.reset_index().plot.scatter(x="timestamp", y="predictions", ax=ax, color="g")
        anomalies_df.reset_index().plot.scatter(x="timestamp", y="value", ax=ax, color="r")
        plt.legend(["Actual TS", "Predictions", "Anomalies"])

        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format="png")
        bytes_image.seek(0)
        return bytes_image

    def run(self):
        self.app.run(debug=True)

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
        df1, df2 = kad_utils.get_dummy_data()  # TODO for testing purpose only
        whole_data = pd.concat([df1, df2])
        dummy_update_interval = 1 * 24 * 60 * 60  # one day

        new_data = whole_data.loc[
                   self.next_timestamp:self.next_timestamp + datetime.timedelta(seconds=dummy_update_interval)]

        if len(new_data) == 0:
            logging.warning("No new data has been obtained")
            return Response(status=200, headers={})

        self.test(new_data)
        self.update_next_timestamp(new_data)

        return Response(status=200, headers={})


if __name__ == "__main__":
    logging.basicConfig(filename="/tmp/kad.log", filemode="w", format="[%(levelname)s] %(filename)s:%(lineno)d: %("
                                                                      "message)s")

    kad = KAD()

    try:
        # metric_range = self.prom.perform_query(query=QUERY,
        #                                        start_time=START_TIME,
        #                                        end_time=END_TIME)
        #
        # metric = response_validator.validate(metric_range)
        # metric_df = metric_parser.metric_to_dataframe(metric, METRIC_NAME).astype(float)
        # train_df, test_df = metric_parser.split_dataset(metric_df)

        train_df, test_df = kad_utils.get_dummy_data()  # TODO replace dummy

        # test_df = kad_utils.normalize(test_df, train_df.mean(), train_df.std())
        # train_df = kad_utils.normalize(train_df, train_df.mean(), train_df.std())

        # train_df[METRIC_NAME].plot()
        # test_df[METRIC_NAME].plot()
        # plt.show()

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
