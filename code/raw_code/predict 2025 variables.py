#!/usr/bin/env python
# coding: utf-8

# In[68]:


import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from statsmodels.tsa.arima_model import ARMA
import statsmodels.tsa.stattools as st
import statsmodels.api as sm


# In[69]:


#Nine of the eleven important variables are in this file
df=pd.read_csv('C:\\Users\\oysdfx\\Desktop\\Modern data\\aquastat.csv')


# In[70]:


#Drop the missing data and only use the existing data to do regression and prediction
df=df.dropna(axis=0)
#Read different variables in the dataframe and put them into separate dataframe
df_AWW=df.loc[df['Variable Name']=='Agricultural water withdrawal']
df_AVA=df.loc[df['Variable Name']=='Agriculture, value added (% GDP)']
df_GDP=df.loc[df['Variable Name']=='GDP per capita']
df_IWW=df.loc[df['Variable Name']=='Industrial water withdrawal']
df_NRI=df.loc[df['Variable Name']=='National Rainfall Index (NRI)']
df_PD=df.loc[df['Variable Name']=='Population density']
df_WS=df.loc[df['Variable Name']=='SDG 6.4.2. Water Stress']
df_TW=df.loc[df['Variable Name']=='Total renewable water resources per capita']
df_TP=df.loc[df['Variable Name']=='Total population with access to safe drinking-water (JMP)']
#Poplulation ages 65 and above is in another file
df_PA65=pd.read_csv('C:\\Users\\oysdfx\\Desktop\\Modern data\\Population ages 65 and above.csv',header=None)
df_PA65=df_PA65.dropna(axis=0)


# In[92]:


##Then do the polynomial regression of years and the value of the variable for every variable. 
#Use polynomial because don't what the model is between years and the variable and polynomial can find the proper model by adjusting the degree.

area=df_AWW['Area']
area=list(set(area))
df_AWW2025=pd.DataFrame(columns=('Area','Agricultural water withdrawal'))
for i in range(0,len(area)):
    df_areai=df_AWW.loc[df_AWW['Area']==area[i]]
    x=df_areai['Year']
    y=df_areai['Value']
    p=np.poly1d(np.polyfit(x,y,1))
    predict=p(2025)
    df_AWW2025=df_AWW2025.append(pd.DataFrame({'Area':[area[i]],'Agriculture, value added (% GDP)':[predict]}))
    #predict=p(2025)
    #Put the predicted result in a dataframe


# In[103]:


area=df_AVA['Area']
area=list(set(area))
df_AVA2025=pd.DataFrame(columns=('Area','Agriculture, value added (% GDP)'))
for i in range(0,len(area)):
    df_areai=df_AVA.loc[df_AVA['Area']==area[i]]
    x=df_areai['Year']
    y=df_areai['Value']
    p=np.poly1d(np.polyfit(x,y,1))
    predict=p(2025)
    df_AVA2025=df_AVA2025.append(pd.DataFrame({'Area':[area[i]],'Agriculture, value added (% GDP)':[predict]}))
#merge the predicted results of different variables into one final result dataframe
result=pd.merge(df_AWW2025,df_AVA2025,on=['Area'])


# In[104]:


area=df_GDP['Area']
area=list(set(area))
df_GDP2025=pd.DataFrame(columns=('Area','GDP per capita'))
for i in range(0,len(area)):
    df_areai=df_GDP.loc[df_GDP['Area']==area[i]]
    x=df_areai['Year']
    y=df_areai['Value']
    p=np.poly1d(np.polyfit(x,y,1))
    predict=p(2025)
    df_GDP2025=df_GDP2025.append(pd.DataFrame({'Area':[area[i]],'GDP per capita':[predict]}))
result=pd.merge(result,df_GDP2025,on=['Area'])


# In[105]:


area=df_IWW['Area']
area=list(set(area))
df_IWW2025=pd.DataFrame(columns=('Area','Industrial water withdrawal'))
for i in range(0,len(area)):
    df_areai=df_IWW.loc[df_IWW['Area']==area[i]]
    x=df_areai['Year']
    y=df_areai['Value']
    p=np.poly1d(np.polyfit(x,y,1))
    predict=p(2025)
    df_IWW2025=df_IWW2025.append(pd.DataFrame({'Area':[area[i]],'Industrial water withdrawal':[predict]}))
result=pd.merge(result,df_IWW2025,on=['Area'])


# In[106]:


area=df_NRI['Area']
area=list(set(area))
df_NRI2025=pd.DataFrame(columns=('Area','National Rainfall Index (NRI)'))
for i in range(0,len(area)):
    df_areai=df_NRI.loc[df_NRI['Area']==area[i]]
    x=df_areai['Year']
    y=df_areai['Value']
    p=np.poly1d(np.polyfit(x,y,1))
    predict=p(2025)
    df_NRI2025=df_NRI2025.append(pd.DataFrame({'Area':[area[i]],'National Rainfall Index (NRI)':[predict]}))
result=pd.merge(result,df_NRI2025,on=['Area'])


# In[107]:


area=df_PD['Area']
area=list(set(area))
df_PD2025=pd.DataFrame(columns=('Area','Population density'))
for i in range(0,len(area)):
    df_areai=df_PD.loc[df_PD['Area']==area[i]]
    x=df_areai['Year']
    y=df_areai['Value']
    p=np.poly1d(np.polyfit(x,y,1))
    predict=p(2025)
    df_PD2025=df_PD2025.append(pd.DataFrame({'Area':[area[i]],'Population density':[predict]}))
result=pd.merge(result,df_PD2025,on=['Area'])


# In[108]:


area=df_WS['Area']
area=list(set(area))
df_WS2025=pd.DataFrame(columns=('Area','SDG 6.4.2. Water Stress'))
for i in range(0,len(area)):
    df_areai=df_WS.loc[df_WS['Area']==area[i]]
    x=df_areai['Year']
    y=df_areai['Value']
    p=np.poly1d(np.polyfit(x,y,1))
    predict=p(2025)
    df_WS2025=df_WS2025.append(pd.DataFrame({'Area':[area[i]],'SDG 6.4.2. Water Stress':[predict]}))
result=pd.merge(result,df_WS2025,on=['Area'])


# In[109]:


area=df_TW['Area']
area=list(set(area))
df_TW2025=pd.DataFrame(columns=('Area','Total renewable water resources per capita'))
for i in range(0,len(area)):
    df_areai=df_TW.loc[df_TW['Area']==area[i]]
    x=df_areai['Year']
    y=df_areai['Value']
    p=np.poly1d(np.polyfit(x,y,1))
    predict=p(2025)
    df_TW2025=df_TW2025.append(pd.DataFrame({'Area':[area[i]],'Total renewable water resources per capita':[predict]}))
result=pd.merge(result,df_TW2025,on=['Area'])


# In[110]:


area=df_TP['Area']
area=list(set(area))
df_TP2025=pd.DataFrame(columns=('Area','Total population with access to safe drinking-water (JMP)'))
for i in range(0,len(area)):
    df_areai=df_TP.loc[df_TP['Area']==area[i]]
    x=df_areai['Year']
    y=df_areai['Value']
    p=np.poly1d(np.polyfit(x,y,1))
    predict=p(2025)
    df_TP2025=df_TP2025.append(pd.DataFrame({'Area':[area[i]],'Total population with access to safe drinking-water (JMP)':[predict]}))
result=pd.merge(result,df_TP2050,on=['Area'])


# In[111]:


x=pd.to_numeric(df_PA65.iloc[0,2:])
df_PA2025=pd.DataFrame(columns=('Area','Poplulation ages 65 and above'))
for i in range(1,df_PA65.shape[0]):
    y=pd.to_numeric(df_PA65.iloc[i,2:])
    p=np.poly1d(np.polyfit(x,y,4))
    predict=p(2025)
    df_PA2025=df_PA2025.append(pd.DataFrame({'Area':[df_PA65.iloc[i,0]],'Poplulation ages 65 and above':[predict]}))
result=pd.merge(result,df_PA2025,on=['Area'])


# In[113]:


#The latirude is in another file and it's not needed to be predicted. So just put insert it into the result data frame directly 
df_Coordinates=pd.read_csv('C:\\Users\\oysdfx\\Desktop\\Modern data\\coordinates.csv',encoding='latin1')
df_LA=df_Coordinates[['country','latitude']]
df_LA.columns=['Area','latitude']
result=pd.merge(result,df_LA,on=['Area'])


# In[114]:


#Write the results into csv file
result.to_csv('2025.csv')


# In[115]:


#Normalize the results for futher anaylyse
country_name = result.iloc[:,0]
norm_data = result.drop('Area',axis=1,inplace=False)
norm_result = norm_data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
norm_result.insert(0,'Area',country_name)


# In[116]:


#Write the normalized results in to another csv file
norm_result.to_csv('nrom_2025.csv')

