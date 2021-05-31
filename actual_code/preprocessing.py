import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer


# ----------------------------------------------------------------------------------------------------------------------
# Produces different data than /raw_code/preprocessing cause. Since nobody responsible fixed it, the results should be
# used in models. Instead, already existing final_data.csv is used.
# ----------------------------------------------------------------------------------------------------------------------


class Preprocessing:

    def __init__(self, dataframe2016, dataframe2020, index2016, index2020):
        self.df2016 = pd.read_csv(dataframe2016)
        self.df2020 = pd.read_csv(dataframe2020)
        self.df_coordinate = pd.read_csv("../data/raw data/coordinates.csv")

        self.df2016 = pd.merge(self.df2016, self.df_coordinate, how='left', on='country')
        self.df2020 = pd.merge(self.df2020, self.df_coordinate, how='left', on='country')

        self.df_index2016 = pd.read_csv(index2016)
        self.df_index2020 = pd.read_csv(index2020)

        self.output_name = None



    def replace(self):
        self.df2016 = self.df2016[self.df2016.isna().sum(axis=1) < 19]
        self.df2016.replace({'0': np.nan})

        self.df2020 = self.df2020[self.df2020.isna().sum(axis=1) < 19]
        self.df2020.replace({'0': np.nan})

    def drop(self):
         threshold2016 = self.df2016.shape[0] * 0.55
         self.df2016.dropna(axis="columns", thresh=threshold2016, inplace=True)
         self.df2016.drop("Unnamed: 0", axis=1, inplace=True)

         threshold2020 = self.df2020.shape[0] * 0.55
         self.df2020.dropna(axis="columns", thresh=threshold2020, inplace=True)
         self.df2020.drop("Unnamed: 0", axis=1, inplace=True)

    def fill_na(self):
        imputer = KNNImputer(n_neighbors=8)
        length = len(self.df2016.columns)
        self.df2016.iloc[:, 2:length] = imputer.fit_transform(self.df2016.iloc[:, 2:length])

        length = len(self.df2020.columns)
        self.df2020.iloc[:, 2:length] = imputer.fit_transform(self.df2020.iloc[:, 2:length])

    def normalization(self):
        length2016 = len(self.df2016.columns)
        self.df2016.iloc[:,1:length2016] = self.df2016.iloc[:,1:length2016].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

        length2020 = len(self.df2020.columns)
        self.df2020.iloc[:, 1:length2020] = self.df2020.iloc[:, 1:length2020].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    def merge_index(self):
         self.df2016 = pd.merge(self.df2016,self.df_index2016, on="country", how="inner")
         self.df2020 = pd.merge(self.df2020, self.df_index2020, on="country", how="inner")
         self.df = self.df2020.append(self.df2016)

         self.df.drop("country", axis=1, inplace=True)
         self.df.drop("Flood occurrence (WRI) (-)", axis=1, inplace=True)

    def write_2020data(self):
        self.output_name2020 = "../data/final data/" + "processed_data_2020" + ".csv"
        self.df2020.to_csv(self.output_name2020)

    def save(self, output_name, norm=True, write=True):
        self.replace()
        self.drop()
        self.fill_na()
        if norm:
            self.normalization()
        self.write_2020data()
        self.merge_index()
        if write:
            self.output_name = "../data/final data/" + output_name + ".csv"
            self.df.to_csv(self.output_name)



    def get_dataframe(self):
        return pd.read_csv(self.output_name)

