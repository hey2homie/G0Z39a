#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_theme(style="dark",palette=sns.color_palette("husl",2))


# In[3]:


data=pd.read_csv("AveragedWhole2017.csv",index_col=[0])
data.head()


# In[4]:


#delete null 

data.isnull().sum()


# In[5]:


data=data.dropna()


# In[6]:


# Group index into two classes

data['index']=data['index'].map({1:0,2:0,3:1,4:1})


# In[7]:


# To exclude the possibility of water stress to be the respond variable by looking at the correlation between it and the others.

data[['Rural population (1000 inhab)', 
           'Urban population (1000 inhab)',
           'Population density (inhab/km2)',
           'Urban population with access to safe drinking-water (JMP) (%)',
           'Rural population with access to safe drinking-water (JMP) (%)',
           'Total population with access to safe drinking-water (JMP) (%)',
           'population ages0-14',
           'Population ages 65 and above(-of total population)'
     ]].corrwith(data['SDG 6.4.2. Water Stress (%)'])


# In[8]:


# The mean value of 0 and 1 classes

data[['Rural population (1000 inhab)', 
           'Urban population (1000 inhab)',
           'Population density (inhab/km2)',
           'Urban population with access to safe drinking-water (JMP) (%)',
           'Rural population with access to safe drinking-water (JMP) (%)',
           'Total population with access to safe drinking-water (JMP) (%)',
           'population ages0-14',
           'Population ages 65 and above(-of total population)','index'
     ]].groupby('index').mean().T


# In[14]:


#Use 1.5  IQR to replace outliers
cols=['Rural population (1000 inhab)', 
           'Urban population (1000 inhab)',
           'Population density (inhab/km2)',
           'Urban population with access to safe drinking-water (JMP) (%)',
           'Rural population with access to safe drinking-water (JMP) (%)',
           'Total population with access to safe drinking-water (JMP) (%)',
           'population ages0-14',
           'Population ages 65 and above(-of total population)']
for col in cols:
    q1=data[col].quantile(0.25)
    q3=data[col].quantile(0.75)
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    
    data[col]=data[col].map(lambda x:x if x> lower else lower)
    data[col]=data[col].map(lambda x:x if x< upper else upper)
for col in ['Rural population (1000 inhab)', 
           'Urban population (1000 inhab)',
           'Population density (inhab/km2)',
           'Urban population with access to safe drinking-water (JMP) (%)',
           'Rural population with access to safe drinking-water (JMP) (%)',
           'Total population with access to safe drinking-water (JMP) (%)',
           'population ages0-14',
           'Population ages 65 and above(-of total population)']:
    plt.figure(figsize=(9,6))
    sns.histplot(x=col,data=data)
    plt.title("Historgram of {}".format(col),fontsize=12)
    
    plt.grid()
    plt.show()


# In[ ]:





# In[15]:


# See the difference of 0 and 1 in each variable

for x in ['Rural population (1000 inhab)', 
           'Urban population (1000 inhab)',
           'Population density (inhab/km2)',
           'Urban population with access to safe drinking-water (JMP) (%)',
           'Rural population with access to safe drinking-water (JMP) (%)',
           'Total population with access to safe drinking-water (JMP) (%)',
           'population ages0-14',
           'Population ages 65 and above(-of total population)'
     ]:
    plt.figure(figsize=(9,6))
    g1=data[data["index"]==0][x].values
    g2=data[data["index"]==1][x].values
    
    plt.hist(g1,bins=50,label="index=0",color='green',alpha=0.5)
    plt.hist(g2,bins=50,label="index=1",color='orange',alpha=0.5)
    
    # 画两根竖线，表示两组数据的均值
    g1_mean=g1.mean()
    g2_mean=g2.mean()
    plt.axvline(g1_mean,ls='--',c='green',linewidth=1)
    plt.axvline(g2_mean,ls='--',c='orange',linewidth=1)
    
    plt.legend()
    plt.title("Historgram by Different Index Group\n{}".format(x),fontsize=14)
    plt.show()


# In[16]:


# Use t.test to see the significance of variables

for col in ['Rural population (1000 inhab)', 
           'Urban population (1000 inhab)',
           'Population density (inhab/km2)',
           'Urban population with access to safe drinking-water (JMP) (%)',
           'Rural population with access to safe drinking-water (JMP) (%)',
           'Total population with access to safe drinking-water (JMP) (%)',
           'population ages0-14',
           'Population ages 65 and above(-of total population)'
     ]:
    
    # 分别取出index为0 1 时的数据
    g1=data[data["index"]==0][col].values
    g2=data[data["index"]==1][col].values
    
    # 先检查两列数据是否具有方差齐性
    p_levene=stats.levene(g1, g2)[1]
    if p_levene>0.05:
        statistic,pvalue_ttest=stats.ttest_ind(g1, g2, equal_var = True)
    else:
        statistic,pvalue_ttest=stats.ttest_ind(g1, g2, equal_var = False)
        
    print("{:<70}: ttest- statistic={:>10.5f};pvalue={:>10.5f}".format(col,statistic,pvalue_ttest))


# + + The Urban population together with the last four variables has significant influence on index classification

# In[ ]:





# In[18]:


# Use the feature of random forest model to output feature importance
# So here use random forest to build a classification model where x is the features and y is index

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(random_state=1)

x_cols=['Rural population (1000 inhab)', 
           'Urban population (1000 inhab)',
           'Population density (inhab/km2)',
           'Urban population with access to safe drinking-water (JMP) (%)',
           'Rural population with access to safe drinking-water (JMP) (%)',
           'Total population with access to safe drinking-water (JMP) (%)',
           'population ages0-14',
           'Population ages 65 and above(-of total population)']
        
x=data[x_cols]
y=data['index']

classifier.fit(x,y)
feature_importances=classifier.feature_importances_

df_feature_importances=pd.DataFrame(zip(x_cols,feature_importances),columns=["features","importance"])
df_feature_importances


# In[12]:


# Visualize the importance

plt.figure(figsize=(16,9))
sns.barplot(x="features",y="importance",data=df_feature_importances)
plt.xticks(rotation=-90)


# In[ ]:




