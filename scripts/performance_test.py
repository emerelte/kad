import numpy as np
import requests
import datetime
from typing import List, Dict
import pickle
from matplotlib import pyplot as plt

NUM_PROBES = 10
NUM_REPETITIONS = 10

# start_time = datetime.datetime.strptime("2021-10-26 15:15:00", "%Y-%m-%d %H:%M:%S")
# stop_time = datetime.datetime.strptime("2021-10-26 15:25:00", "%Y-%m-%d %H:%M:%S")
start_time = datetime.datetime.strptime("2014-04-01 14:00:00", "%Y-%m-%d %H:%M:%S")
stop_time = datetime.datetime.strptime("2014-04-09 14:00:00", "%Y-%m-%d %H:%M:%S")

url: str = "http://localhost:5000/update_config"

request_dur_by_length: Dict[int, List] = {}
len_train_df = -1

for i in range(NUM_PROBES):
    print(i)
    request_dur: List = []
    for _ in range(NUM_REPETITIONS):
        config_json: Dict = {
            "METRIC_NAME": "value",#'rate(request_duration_seconds_count{job="kubernetes-service-endpoints", kubernetes_name="front-end", kubernetes_namespace="sock-shop", method="get", name="front-end", route="/", service="front-end", status_code="200"}[1m])',
   #         "MODEL_NAME": "SarimaModel",
            "START_TIME": str(start_time),
            "END_TIME": str(stop_time)}
        try:
            response = requests.post(url, json=config_json)
            print(response.json())
            len_train_df: int = response.json()["train_df_len"]
            request_dur.append(response.elapsed.total_seconds())
        except requests.exceptions.ConnectionError as ce:
            print(str(ce))
            continue
    request_dur_by_length[len_train_df] = request_dur
    stop_time = stop_time + datetime.timedelta(seconds=100)

eval_file = open("eval.pkl", "wb")
pickle.dump(request_dur_by_length, eval_file)
