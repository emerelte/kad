import prometheus_client
# import prometheus_api_client as prom_api
import requests
import datetime

metric_name = "container_cpu_cfs_periods_total"
# start_time = datetime.fromtimestamp(1000000000)
# end_time = datetime.fromtimestamp(1000000001)
label_config = {}

PROMETHEUS_URL = 'http://localhost:9090/'
prom_token = 'prometheus access token'
training_window = 'integer representing weeks'
threshold = 'float'

# prom = prom_api.prometheus_connect.PrometheusConnect(url=PROMETHEUS_URL,
#                                                      headers={"Authorization": "bearer {}".format(prom_token)},
#                                                      disable_ssl=True)

response = requests.get(PROMETHEUS_URL + "/api/v1/query", params={"query": metric_name})
results = response.json()['data']['result']

print(results)
# end_of_month = datetime.datetime.today().replace(day=1).date()
#
# last_day = end_of_month - datetime.timedelta(days=1)
# print('{:%B %Y}:'.format(last_day))
# for result in results:
#     print(' {metric}: {value[1]}'.format(**result))
