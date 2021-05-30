import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import normalize

# ----------------------------------------------------------------------------------------------------------------------
# Produces different data than /raw_code/preprocessing cause. Since nobody responsible fixed it, the results should be
# used in models. Instead, already existing final_data.csv is used.
# ----------------------------------------------------------------------------------------------------------------------


class Preprocessing:

    def __init__(self, dataframe2016, dataframe2020, index2016, index2020):
        self.df2016 = pd.merge(pd.read_csv(dataframe2016), pd.read_csv(index2016),
                               on="country", how="inner")
        self.df2020 = pd.merge(pd.read_excel(dataframe2020), pd.read_csv(index2020),
                               on="country", how="inner")
        self.df = self.df2016.append(self.df2020)
        self.df = pd.read_csv(dataframe2016)
        self.output_name = None
        self.countries = None

    def replace(self):
        self.df = self.df[self.df.isna().sum(axis=1) < 19]
        self.df.replace({'-': np.nan, '0': np.nan, '>99': 99})

    def drop(self):
        threshold = self.df.shape[0] * 0.55
        self.df.dropna(axis="columns", thresh=threshold, inplace=True)
        self.df.drop("Unnamed: 0", axis=1, inplace=True)

    def fill_na(self):
        imputer = KNNImputer(n_neighbors=8)
        length = len(self.df.columns)
        self.df.iloc[:, 2:length] = pd.DataFrame(imputer.fit_transform(self.df.iloc[:, 2:length]))

    def normalization(self):
        self.df.iloc[:, 1:len(self.df.columns)] = normalize(self.df.iloc[:, 1:len(self.df.columns)])

    def save(self, output_name, norm=True, write=True):
        self.replace()
        self.drop()
        self.fill_na()
        if norm:
            self.normalization()
        if write:
            self.output_name = "../../data/final data/" + output_name + ".csv"
            self.df.to_csv(self.output_name)

    def get_dataframe(self):
        return pd.read_csv(self.output_name)
