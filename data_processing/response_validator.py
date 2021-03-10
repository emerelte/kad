from utils import utils


def validate(metric_range: utils.PromQueryResponse):
    if len(metric_range) < 1:
        print("[WRN] No metric found")
        return None
    if len(metric_range) > 1:
        print("[WRN] Multiple metrics found")
        return None
    return metric_range[0]
