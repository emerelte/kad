import pandas as pd
from statsmodels.tsa.stattools import adfuller
from kad.kad_utils import kad_utils


class TsAnalyzer:
    """
    Ts analyzer provides a set of methods describing characteristics of the data
    Constructor assumes that the DataFrame has only one column
    """

    def __init__(self, ts: pd.DataFrame):
        self.data = ts

    def is_stationary(self):
        # print(self.data.head())

        # decompose = seasonal_decompose(self.data.resample('M').sum(), period=12)
        # fig = plt.figure()
        # fig = decompose.plot()
        # fig.set_size_inches(12, 8)
        # plt.show()

        adf = adfuller(self.data.iloc[:, 0], 12)
        kad_utils.print_adf_results(adf)

        significance = 0.05

        return kad_utils.get_statistic_test(adf) < kad_utils.get_critical_values(adf)["1%"] \
               and kad_utils.get_pvalue(adf) < significance
