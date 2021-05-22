#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


data=pd.read_csv("wholedataAsia.csv")
data.head()
data.shape


# In[16]:


# data formatting
data=data.replace("-",np.nan)
data=data.replace("0",np.nan)
data=data.replace(">99",99)
data.head()


# In[17]:


data.dtypes


# In[18]:


#Take a look at the percentage of missing data of every column, delete the ones higher than 0.35
df_missing=data.isnull().sum().reset_index()
df_missing.columns=["feature","missing count"]
df_missing["missing percentage"]=df_missing["missing count"]/data.shape[0]
#df_missing["missing percentage"]=df_missing["missing percentage"].map(lambda x:"{:0.2%}".format(x))
df_missing.sort_values(by="missing count",ascending=True,inplace=True)
df=df_missing[df_missing["missing percentage"]<0.35]
data=data[df["feature"]]
data.head()


# In[19]:


from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)


# In[13]:


listcolumn=data.columns
data_new=imputer.fit_transform(data[listcolumn[1:]])
data_new=pd.DataFrame(data_new)
data_new.columns=listcolumn[1:]
data_new.insert(0,"country",data["country"])
data_new
#data_new.to_csv("Imputed.csv",encoding="utf-8")


# In[20]:



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
norm_df.to_csv('averageAsia.csv')


# In[ ]:





# In[ ]:




