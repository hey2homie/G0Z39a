#!/usr/bin/env python
# coding: utf-8

# In[58]:


#Use the random forest model to calculate the index in 2025


import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import train_test_split


# In[59]:


whole2020=pd.read_csv("C://Users//oysdfx//Desktop//data//2020whole.csv")
AP2020=ace=pd.read_csv("C://Users//oysdfx//Desktop//data//2020APindex.csv")   
AP2020=AP2020.merge(pd.read_csv("C://Users//oysdfx//Desktop//data//2020whole.csv"),on="country",how="inner")   


# In[60]:


AP2016=ace=pd.read_csv("C://Users//oysdfx//Desktop//data//2016APindex.csv")   
AP2016=AP2016.merge(pd.read_csv("C://Users//oysdfx//Desktop//data//2016whole.csv"),on="country",how="inner")   
AP2020.append(AP2016)


# In[61]:


#check the cloumns of these two dataframe
main_list = np.setdiff1d(AP2016.columns,AP2020.columns)

print(main_list)


newdata=AP2020[AP2016.columns].append(AP2016)
print(newdata.columns)


newdata.to_csv('newdata.csv')
datas=newdata.values


y=datas[:,1]
y=y.astype('float')
X=datas[:,2:datas.shape[1]]


#split the data into traing and testing test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


#use random forest to build the model
rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
rfc.fit(X_train, y_train)
performance1=rfc.score(X_test, y_test)
print("performance1",performance1)#73%

#get the feature importances of each variables
importance=rfc.feature_importances_
featurename=newdata.columns[2:]
for i in range(0,len(featurename)):
	print(featurename[i],importance[i])
    


#rebuild the model based on the selected variables,select the varibales have the importance over 1%
srfc = SelectFromModel(rfc, threshold=0.03)
srfc.fit(X_train, y_train)
# Train the selected variables
X_important_train = srfc.transform(X_train)
X_important_test = srfc.transform(X_test)

rfc_important = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
rfc_important.fit(X_important_train, y_train)

y1_important_pred = rfc_important.predict(X_important_test)


# In[62]:


# rfc_sel_index the index of the selected variables
rfc_sel_index=srfc.get_support(indices=True)
print(newdata.columns[rfc_sel_index+1])


importance2=rfc_important.feature_importances_
featurename2=newdata.columns[rfc_sel_index+1]
for i in range(0,len(featurename2)):
	print(featurename2[i],importance2[i])


# In[63]:


#Use the model to calculate the Index in 2025
y1_important_pred = rfc_important.predict(X_important_test)
data2025=pd.read_csv("C://Users//oysdfx//Desktop//data//norm_2025.csv")
x=data2025.iloc[:,1:]
y_pred = rfc_important.predict(x)


# In[73]:


df_index2025=pd.DataFrame({'Country':data2025['Area'],'Index':y_pred})


# In[79]:


df_index2025.to_csv('C://Users//oysdfx//Desktop//data//2025Index.csv')

