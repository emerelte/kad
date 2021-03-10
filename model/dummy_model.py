# TODO add base class and implement useful models
class DummyModel:
    """
    Takes training dataframe as input and computes internal states that will be used to predict the test data classes
    """

    def train(self, train_df):
        pass

    """
    Appends a column to the df with classes
    """

    def test(self, test_df):
        return test_df
