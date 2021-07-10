import io
import datetime
import json
import logging
import os
from time import sleep

import requests
import sys
import yaml

sys.path.insert(1, "..")
from apscheduler.schedulers.background import BackgroundScheduler
from matplotlib import pyplot as plt
import pandas as pd
from flask import Flask, send_file, Response, jsonify, request
from flask_cors import cross_origin, CORS

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
from kad.visualization.visualization import visualize


def request_new_data(p_config: dict):
    logging.info("Requesting new data...")
    r = requests.get(p_config["APP_URL"] + p_config["UPDATE_DATA_ENDPOINT"])

    if r.status_code != 200:
        logging.warning("Something went wrong when requesting data update")


class KAD(object):
    app = None

    def __init__(self, p_config: dict):
        self.app = Flask("KAD app")
        CORS(self.app, resources={r"/*": {"origins": "*"}})

        self.app.add_url_rule(rule="/" + config["PLOT_RESULTS_ENDPOINT"],
                              endpoint=config["PLOT_RESULTS_ENDPOINT"],
                              view_func=self.plot_results)
        self.app.add_url_rule(rule="/" + config["GET_RESULTS_ENDPOINT"],
                              endpoint=config["GET_RESULTS_ENDPOINT"],
                              view_func=self.get_results)
        self.app.add_url_rule(rule="/" + config["UPDATE_DATA_ENDPOINT"],
                              endpoint=config["UPDATE_DATA_ENDPOINT"],
                              view_func=self.update_data)
        self.app.add_url_rule(rule="/" + config["UPDATE_CONFIG_ENDPOINT"],
                              endpoint=config["UPDATE_CONFIG_ENDPOINT"],
                              view_func=self.update_config,
                              methods=["POST"])

        self.config: dict = p_config
        self.config["START_TIME"] = eval(self.config["START_TIME"])
        self.config["END_TIME"] = eval(self.config["END_TIME"])
        self.data_source: i_data_source = None
        self.model: i_model.IModel = None
        self.metric_name: str = ""
        self.last_train_sample = None
        self.results_df: pd.DataFrame = None
        self.train_mean = None
        self.train_std = None

        self.set_up()

    def reset(self):
        self.data_source = None
        self.model = None
        self.metric_name = ""
        self.last_train_sample = None
        self.results_df = None
        self.train_mean = None
        self.train_std = None

    def set_up(self):
        self.metric_name = self.config["METRIC_NAME"]

        file = "data/archive/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv"
        daily_jumpsup_csv_path = os.path.join(
            "/home/maciek/Documents/Magisterka/kubernetes-anomaly-detector/notebooks/",
            file)
        self.data_source = ExemplaryDataSource(
            path=daily_jumpsup_csv_path,
            metric_name=self.metric_name,
            start_time=self.config["START_TIME"],
            stop_time=self.config["END_TIME"],
            update_interval_hours=10)
        # self.data_source = PrometheusDataSource(query=p_config["QUERY"],
        #                                                        prom_url=p_config["PROMETHEUS_URL"],
        #                                                        metric_name=p_config["METRIC_NAME"],
        #                                                        start_time=p_config["START_TIME"],
        #                                                        stop_time=p_config["END_TIME"],
        #                                                        update_interval_sec=p_config["UPDATE_INTERVAL_SEC"])

        self.model = SarimaModel(order=(0, 0, 0), seasonal_order=(1, 0, 1, 24))
        # self.model = AutoEncoderModel(time_steps=12)
        # self.model = HmmModel()
        self.results_df = None
        self.last_train_sample = None

    def get_train_data(self) -> pd.DataFrame:
        train_df = self.data_source.get_train_data()
        self.last_train_sample = len(train_df)
        self.train_mean = train_df.mean()
        self.train_std = train_df.std()
        return kad_utils.normalize(train_df, self.train_mean, self.train_std)

    def train_model(self):
        train_df = self.get_train_data()
        self.model.train(train_df)
        if len(train_df) < 2:
            logging.warning("Almost empty training df (len < 2)")
        logging.info("Model trained")

    def test(self, test_df):
        self.results_df = self.model.test(test_df)

    def get_latest_image(self):
        if self.results_df is None:
            logging.warning("Results not obtained yet")
            return None

        visualize(self.results_df, self.metric_name, self.last_train_sample)

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
        logging.info("Results plot requested")
        bytes_obj = self.get_latest_image()

        if bytes_obj is None:
            logging.warning("Image empty!")
            return Response(status=404, headers={})

        return send_file(bytes_obj,
                         attachment_filename="plot.png",
                         mimetype="image/png")

    @cross_origin(supports_credentials=True)
    def get_results(self):
        logging.info("Results in raw format requested")

        if self.results_df is None:
            logging.warning("Results not obtained yet")
            return Response(status=404, headers={})

        results_json = json.loads(self.results_df.to_json())
        results_json["metric"] = self.metric_name
        return jsonify(results_json)

    @cross_origin(supports_credentials=True)
    def update_data(self):
        logging.info("Updating data")

        if self.data_source is None:
            logging.warning("Data source not present while requesting data update")
            return Response(status=200, headers={})

        try:
            new_data = self.data_source.get_next_batch()
        except DataSourceException as dsexc:
            logging.warning(str(dsexc))
            return Response(status=200, headers={})

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

    @cross_origin(supports_credentials=True)
    def update_config(self):
        logging.info("Updating config")
        json_data = request.get_json()
        print(json_data)
        print(json_data["METRIC_NAME"])
        print(datetime.datetime.fromtimestamp(json_data["START_TIME"]))

        logging.debug("Updating config...")
        try:
            self.config["METRIC_NAME"] = json_data["METRIC_NAME"]
            self.config["START_TIME"] = datetime.datetime.fromtimestamp(json_data["START_TIME"])
            self.config["END_TIME"] = datetime.datetime.fromtimestamp(json_data["END_TIME"])
            self.set_up()

            logging.debug("Training once again")
            self.train_model()
        except Exception:
            logging.error("Unsuccessfull config update - resetting config")
            self.reset()
            return jsonify({"Error": "Unsuccessfull config update - resetting config"})

        return Response(status=200, headers={})


if __name__ == "__main__":
    logging.basicConfig(filename="/tmp/kad.log", filemode="w", format="[%(levelname)s] %(filename)s:%(lineno)d: %("
                                                                      "message)s", level=logging.INFO)

    RETRY_INTERV = 10

    while 1:
        with open("kad/config/config.yaml", 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.error(exc)

        kad = KAD(config)

        try:
            train_df = kad.get_train_data()

            train_df[config["METRIC_NAME"]].plot()
            plt.show()

            kad.train_model()

            scheduler = BackgroundScheduler()
            job = scheduler.add_job(lambda: request_new_data(config), "interval",
                                    minutes=config["UPDATE_INTERVAL_SEC"] / 60)
            scheduler.start()

            kad.run()

        except requests.exceptions.ConnectionError:
            logging.error("Could not perform query to Prometheus API")
        except response_validator.MetricValidatorException as exc:
            logging.error("Malformed metrics: " + str(exc))
        except DataSourceException as exc:
            logging.error("Too small training df: " + str(exc))
        finally:
            logging.info("Kad process failed. Retrying after: " + str(RETRY_INTERV) + " seconds")
            sleep(RETRY_INTERV)
