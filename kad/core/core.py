import datetime
import io
import json
import logging
import os
import sys

from kad.model.model_utils import name_2_model
from kad.ts_analyzer import ts_analyzer

sys.path.insert(1, "..")
from matplotlib import pyplot as plt
import pandas as pd
from flask import Flask, send_file, Response, jsonify, request
from flask_cors import cross_origin, CORS

from kad.data_sources import i_data_source
from kad.data_sources.exemplary_data_source import ExemplaryDataSource
from kad.data_sources.i_data_source import DataSourceException
from kad.kad_utils import kad_utils
from kad.kad_utils.kad_utils import EndpointAction
from kad.model import i_model
from kad.visualization.visualization import visualize


class Core(object):
    app = None

    def __init__(self, p_config: dict):
        self.app = Flask("KAD app")
        CORS(self.app, resources={r"/*": {"origins": "*"}})

        self.config: dict = p_config

        self.app.add_url_rule(rule="/" + self.config["PLOT_RESULTS_ENDPOINT"],
                              endpoint=self.config["PLOT_RESULTS_ENDPOINT"],
                              view_func=self.plot_results)
        self.app.add_url_rule(rule="/" + self.config["GET_RESULTS_ENDPOINT"],
                              endpoint=self.config["GET_RESULTS_ENDPOINT"],
                              view_func=self.get_results)
        self.app.add_url_rule(rule="/" + self.config["UPDATE_DATA_ENDPOINT"],
                              endpoint=self.config["UPDATE_DATA_ENDPOINT"],
                              view_func=self.update_data)
        self.app.add_url_rule(rule="/" + self.config["UPDATE_CONFIG_ENDPOINT"],
                              endpoint=self.config["UPDATE_CONFIG_ENDPOINT"],
                              view_func=self.update_config,
                              methods=["POST"])

        self.data_source: i_data_source = None
        self.model_selector: ts_analyzer.TsAnalyzer = None
        self.model: i_model.IModel = None
        self.last_train_sample = None
        self.results_df: pd.DataFrame = None
        self.train_mean = None
        self.train_std = None

        self.set_up()

    def reset(self):
        self.data_source = None
        self.model = None
        self.last_train_sample = None
        self.results_df = None
        self.train_mean = None
        self.train_std = None

    def set_up(self):
        file = "data/archive/artificialWithAnomaly/artificialWithAnomaly/art_daily_jumpsup.csv"
        daily_jumpsup_csv_path = os.path.join(
            "/home/maciek/Documents/Magisterka/kubernetes-anomaly-detector/notebooks/",
            file)
        self.data_source = ExemplaryDataSource(
            path=daily_jumpsup_csv_path,
            metric_name=self.config["METRIC_NAME"],
            start_time=datetime.datetime.strptime(self.config["START_TIME"], "%Y-%m-%d %H:%M:%S"),
            stop_time=datetime.datetime.strptime(self.config["END_TIME"], "%Y-%m-%d %H:%M:%S"),
            update_interval_hours=10)
        # self.data_source = PrometheusDataSource(query=p_config["QUERY"],
        #                                                        prom_url=p_config["PROMETHEUS_URL"],
        #                                                        metric_name=p_config["METRIC_NAME"],
        #                                                        start_time=p_config["START_TIME"],
        #                                                        stop_time=p_config["END_TIME"],
        #                                                        update_interval_sec=p_config["UPDATE_INTERVAL_SEC"])

        # self.model = SarimaModel(order=(0, 0, 0), seasonal_order=(1, 0, 1, 24))
        # self.model = AutoEncoderModel(time_steps=12)
        # self.model = HmmModel()
        self.results_df = None
        self.last_train_sample = None

    def __get_train_data(self) -> pd.DataFrame:
        train_df = self.data_source.get_train_data()
        self.last_train_sample = len(train_df)
        self.train_mean = train_df.mean()
        self.train_std = train_df.std()
        return kad_utils.normalize(train_df, self.train_mean, self.train_std)

    def __select_and_train_model(self):
        train_df = self.__get_train_data()

        self.model_selector = ts_analyzer.TsAnalyzer(train_df)
        if "MODEL_NAME" in self.config:
            self.model = name_2_model(self.config["MODEL_NAME"], self.model_selector)
        else:
            self.model = self.model_selector.select_model()
        logging.info("Selected model: " + self.model.__class__.__name__)

        self.model.train(train_df)  # TODO remove or add a separate option w/o extra validation

        if len(train_df) < 2:
            logging.warning("Almost empty training df (len < 2)")
        logging.info("Model trained")

    def test(self, test_df):
        self.results_df = self.model.test(test_df)

    def get_latest_image(self):
        if self.results_df is None:
            logging.warning("Results not obtained yet")
            return None

        visualize(self.results_df, self.config["METRIC_NAME"], self.last_train_sample)

        bytes_image = io.BytesIO()
        plt.savefig(bytes_image, format="png")
        bytes_image.seek(0)
        return bytes_image

    def run(self):
        self.__select_and_train_model()
        self.app.run(debug=True, threaded=False)

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
        results_json["metric"] = self.config["METRIC_NAME"]
        results_json["model"] = self.model.__class__.__name__
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

        if not self.are_changes_in_config(json_data):
            logging.warning("No changes in config")
            return Response(status=200, headers={})

        logging.info("Updating config accepted")
        try:
            self.config["METRIC_NAME"] = json_data["METRIC_NAME"]
            self.config["START_TIME"] = json_data["START_TIME"]
            self.config["END_TIME"] = json_data["END_TIME"]
            self.config["MODEL_NAME"] = json_data["MODEL_NAME"]
            self.set_up()

            logging.info("Training once again")
            self.__select_and_train_model()
        except Exception:
            logging.error("Unsuccessfull config update - resetting config")
            self.reset()
            return jsonify({"Error": "Unsuccessfull config update - resetting config"})

        return Response(status=200, headers={})

    def are_changes_in_config(self, new_config: dict) -> bool:
        shared_items = {k: self.config[k] for k in self.config if k in new_config and self.config[k] == new_config[k]}

        return len(shared_items) != len(new_config.keys())
