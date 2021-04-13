from kad.kad_utils import kad_utils
import datetime
import prometheus_api_client as prom_api
from prometheus_api_client import MetricsList


class PrometheusConnectWrapper:
    def __init__(self, prometheus_url: str):
        self.prom = prom_api.PrometheusConnect(url=prometheus_url, disable_ssl=True)

    def perform_query(self, query: str, start_time: datetime, end_time: datetime) -> kad_utils.PromQueryResponse:
        return self.prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step="10")

    def fetch_metric_range_data(self, metric_name: str, start_time: datetime, end_time: datetime,
                                label_config: dict) -> MetricsList:
        return MetricsList(self.prom.get_metric_range_data(metric_name=metric_name,
                                                           start_time=start_time,
                                                           end_time=end_time,
                                                           label_config=label_config))
