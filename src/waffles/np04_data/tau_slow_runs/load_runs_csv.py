import pandas as pd
import os

class ReaderCSV:

    def __init__(self, path_to_csv = None):
        self.__path_to_csv = None
        if not self.__path_to_csv:
            self.__path_to_csv = os.path.dirname(os.path.realpath(__file__))

        self.dataframes:dict = {}
        self.load_dataframes()
    def load_dataframes(self):
        dftypes = ["purity", "beam", "led"]
        for dt in dftypes:
            self.dataframes[dt] = pd.read_csv(f"{self.__path_to_csv}/{dt}_runs.csv")



