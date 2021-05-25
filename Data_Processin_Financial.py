# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

Financial_Data = pd.read_excel ("/home/basvanhaperen/MDA Project/FinancialData.xlsx")
Water_Stress = pd.read_excel ("/home/basvanhaperen/MDA Project/WaterStressData.xlsx")

GDP_data = pd.read_excel("/home/basvanhaperen/MDA Project/GDP_Data.xlsx")
GDP_data = GDP_data.loc[GDP_data['Unnamed: 2'] == "Gross Domestic Product (GDP)"]
GDP_data = GDP_data[["Unnamed: 1", "Unnamed: 48"]]
GDP_data.columns = ["Country", "GDP_2015"]


Data = pd.merge(Financial_Data, Water_Stress, on='Country', how='outer')
Data = pd.merge(Data, GDP_data, on='Country', how='outer')

Data = Data.dropna(axis = 0)
Data = Data.reset_index()
Data = Data.drop(['ISO', 'index', "PPP", "2030 Population", "2030 GDP", 
                  "Water Per Capita", "GDP/Capita"], axis = 1)

Data.to_csv("/home/basvanhaperen/MDA Project/Finances_Water.csv")

import matplotlib.pyplot as plt

Data['Percentage_GDP'] = Data['Total'] / Data['GDP_2015']

Data.plot(kind='scatter',x='Percentage_GDP',y='Water Stress',color='red')
plt.xscale("log")
plt.yscale("log")
plt.show()