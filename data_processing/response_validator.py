from kad_utils import kad_utils


class MetricValidatorException(Exception):
    def __init__(self, message="Unspecified Metric Validator Exception"):
        self.message = message

    def __str__(self):
        return f"{self.message}"


def validate(metric_range: kad_utils.PromQueryResponse):
    if len(metric_range) < 1:
        raise MetricValidatorException("No metrics found")
    if len(metric_range) > 1:
        raise MetricValidatorException("Multiple metrics found")
    return metric_range[0]
