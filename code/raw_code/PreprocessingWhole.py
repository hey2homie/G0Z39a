#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('5.23 2012 wholedata.csv')
#data.head()

data = data[data.isna().sum(axis=1) < 19]
coordinates=pd.read_csv("coordinates.csv",engine='python')
data= pd.merge(data, coordinates, how='left', on='country')
#data.shape

# data formatting
data=data.replace("-",np.nan)
data=data.replace("0",np.nan)
data=data.replace(">99",99)
#data.head()

#check the types of data
#data.dtypes

#Show missing data of every variable
df_missing=data.isnull().sum().reset_index()
df_missing.columns=["feature","missing count"]
df_missing["missing percentage"]=df_missing["missing count"]/data.shape[0]
#df_missing["missing percentage"]=df_missing["missing percentage"].map(lambda x:"{:0.2%}".format(x))
df_missing.sort_values(by="missing count",ascending=True,inplace=True)
#df_missing

#More missing data than Asia data, so 0.35 change to 0.65 and so on....
df=df_missing[df_missing["missing percentage"]<0.55]
data=data[df["feature"]]
data.head()

#Sort data by latitude
data=data.sort_values(by='latitude', ascending=True)

#To save the sequence of country for later, because that if directly insert country into the imputed data there will be false correspondence
country=data["country"]
df = pd.DataFrame(country,columns=['country'])
country_dict = df['country'].unique().tolist()
df['transformed']=df['country'].apply(lambda x : country_dict.index(x))
df.columns=["country1","transformed"]
data['transformed']=df['transformed']



#data.to_csv("Ratesmalerthan35.csv",encoding="utf-8")
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=8)
data=data.drop('Unnamed: 0',axis=1)


listcolumn=data.columns
data_new=imputer.fit_transform(data[listcolumn[1:]])
data_new=pd.DataFrame(data_new)
data_new.columns=listcolumn[1:]
df['transformed1']= df['transformed'].apply(lambda x : country_dict[x])
data_new=pd.merge(data_new,df,on="transformed")
data_new.insert(0,"country",data_new["country1"])
data_new=data_new.drop(['country1','transformed','transformed1'],axis=1)
#data_new
#data_new.to_csv("Imputed.csv",encoding="utf-8")


# In[263]:


# Normalization

# remove the column of country name & water securty index
country_name = data_new.iloc[:,0]  #get the content of country name column

norm_data = data_new.drop('country',axis=1,inplace=False)
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
norm_df.to_csv('AveragedWhole2012_55%_imp8.csv')






