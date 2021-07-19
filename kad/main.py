import logging
import sys
from time import sleep

import requests
import yaml

from kad.core.core import Core

sys.path.insert(1, "..")
from apscheduler.schedulers.background import BackgroundScheduler
from matplotlib import pyplot as plt

from kad.data_processing import response_validator
from kad.data_sources.i_data_source import DataSourceException


def request_new_data(p_config: dict):
    logging.info("Requesting new data...")
    r = requests.get(p_config["APP_URL"] + p_config["UPDATE_DATA_ENDPOINT"])

    if r.status_code != 200:
        logging.warning("Something went wrong when requesting data update")


if __name__ == "__main__":
    logging.basicConfig(filename="/tmp/kad.log", filemode="w",
                        format="[%(levelname)s] %(filename)s:%(lineno)d: %("
                               "message)s", level=logging.INFO)

    RETRY_INTERV = 10

    while 1:
        with open("kad/config/config.yaml", 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.error(exc)

        kad = Core(config)

        try:
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
