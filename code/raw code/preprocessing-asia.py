import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

asia = pd.read_csv('wholedataAsia.csv')

pd.set_option('display.max_columns', None)  # showing all columns
pd.set_option('display.max_rows', None)  # showing all rows
# print(test1)


columnMissing = asia.isna().sum(axis=0)  # check missing value (column)
rowMissing = asia.isna().sum(axis=1)  # check missing value (row)
#print(columnMissing)
#print(rowMissing)      # 47 row - asia

reducedAsia1 = asia



def drop_col(df, col_name,cutoff=0.7):  #e.g. cutoff = 0.8, one column contains less than 80% non NA value to be droped
    n = len(df)
    cnt = df[col_name].count()

    if (float(cnt) / n) < cutoff:
        df.drop(col_name, axis=1, inplace=True)
        return df

columnNames = reducedAsia1.columns.values.tolist()
print(columnNames)

for i in columnNames:
    drop_col(reducedAsia1,i)


print(reducedAsia1.isna().sum())

reducedAsia2 = reducedAsia1[reducedAsia1.isna().sum(axis=1) < 10]   # drop row with more than 15 missing value
print(reducedAsia2.isna().sum(axis=1))
print(reducedAsia2.shape[1])                    #the number of row left


# fill all NA value with average
numeric = reducedAsia2.select_dtypes(include=np.number)
numeric_columns = numeric.columns
print(numeric_columns)
reducedAsia2[numeric_columns] = reducedAsia2[numeric_columns].fillna(reducedAsia2.mean())
print(reducedAsia2.isna().sum())



# examine data types of each column
# print(reducedAsia2.dtypes)

# Normalization

# remove the column of country name & water securty index
country_name = reducedAsia2.iloc[:,0]  #get the content of country name column
water_index = reducedAsia2.iloc[:,1]

norm_data_asia = reducedAsia2.drop('country',axis=1,inplace=False)
norm_data_asia = norm_data_asia.drop('water security index',axis=1,inplace=False)
# print(norm_data_asia.dtypes)

#normalization
norm_asia = norm_data_asia.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
# print(norm_asia.head(5))

#add the column of country name
norm_asia.insert(0,'country',country_name)
norm_asia.insert(1,'water security index',water_index)
#print(norm_asia.head(5))





# output the preprocessed dataframe
norm_asia.to_csv('average asia.csv')

