import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv('wholedata.csv')  # input data as data frame

#test1 = df.head(5)
pd.set_option('display.max_columns', None)  # showing all columns
pd.set_option('display.max_rows', None)  # showing all rows
# print(test1)

# data formatting
df.replace('-', np.nan, inplace=True)
df.replace(0, np.nan, inplace=True)
df.replace('>99', 99, inplace=True)
#print(df.head(10))

# formatting the data type
# print(df.dtypes)
df['RSA'] = df['RSA'].astype('float64')
df['rural water supply'] = df['rural water supply'].astype('float64')
df['urban water supply'] = df['urban water supply'].astype('float64')
df['ARA'] = df['ARA'].astype('float64')


columnMissing = df.isna().sum(axis=0)  # check missing value (column)
rowMissing = df.isna().sum(axis=1)  # check missing value (row)
#print(columnMissing)
#print(rowMissing)  # 47 row

reduced_df1 = df


reduced_df2 = reduced_df1[reduced_df1.isna().sum(axis=1) < 25]   # drop row with more than 25 missing value
# print(reduced_df2.isna().sum(axis=1))
print(reduced_df2.shape[1])                    #the number of column left
# print(reduced_df2.shape[0])                    #the number of row left


def drop_col(dataframe, col_name,cutoff=0.85):  #e.g. cutoff = 0.8, one column contains less than 80% non NA value to be droped
    n = len(dataframe)
    cnt = dataframe[col_name].count()

    if (float(cnt) / n) < cutoff:
        dataframe.drop(col_name, axis=1, inplace=True)
        return dataframe

columnNames = reduced_df2.columns.values.tolist()
print(columnNames)

for i in columnNames:
    drop_col(reduced_df2,i)

print(reduced_df2.shape[1])                    #the number of column left



# fill all NA value with average
numeric = reduced_df2.select_dtypes(include=np.number)
numeric_columns = numeric.columns
# print(numeric_columns)
reduced_df2[numeric_columns] = reduced_df2[numeric_columns].fillna(reduced_df2.mean())
print(reduced_df2.isna().sum())



# examine data types of each column
# print(reducedAsia2.dtypes)

# Normalization

# remove the column of country name & water securty index
country_name = reduced_df2.iloc[:,0]  #get the content of country name column

norm_data = reduced_df2.drop('country',axis=1,inplace=False)
# print(norm_data_asia.dtypes)

#normalization
norm_df = norm_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
#print(norm_asia.head(5))

#add the column of country name
norm_df.insert(0,'country',country_name)
#print(norm_df.head(5))


# 2. min-max, SKL

# create a scaler object
#scaler2 = MinMaxScaler()
# fit and transform the data
#asia_norm = pd.DataFrame(scaler.fit_transform(df_cars), columns=df_cars.columns)



# output the preprocessed dataframe
norm_df.to_csv('average whole.csv')


