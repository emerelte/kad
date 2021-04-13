from abc import ABC


class DataSourceException(Exception):
    def __init__(self, message="Data Source Exception"):
        self.message = message

    def __str__(self):
        return f"{self.message}"


class IDataSource(ABC):

    def get_train_data(self):
        pass

    def get_next_batch(self):
        pass
