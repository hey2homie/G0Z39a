#!/usr/bin/env python
# coding: utf-8

#This analysis is to use the final output to gain some knowledge about the relationship with population related variables and index.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
sns.set_theme(style="dark",palette=sns.color_palette("husl",2))




data=pd.read_csv("../final_data.csv",index_col=[0])
data.isnull().sum()
data=data.dropna()


# Group index into two classes

data['index']=data['index'].map({1:0,2:0,3:1,4:1})


# To exclude the possibility of water stress to be the respond variable by looking at the correlation between it and the others.
listpop=['Rural population (1000 inhab)', 
           'Urban population (1000 inhab)',
           'Population density (inhab/km2)',
           'Urban population with access to safe drinking-water (JMP) (%)',
           'Rural population with access to safe drinking-water (JMP) (%)',
           'Total population with access to safe drinking-water (JMP) (%)',
           'population ages0-14',
           'Population ages 65 and above(-of total population)']
data[listpop].corrwith(data['SDG 6.4.2. Water Stress (%)'])


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



# See the difference of 0 and 1 in each variable

for x in listpop:
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



# Use t.test to see the significance of variables

for col in listpop:
    
    # Get the values of 0 and 1 seperately
    g1=data[data["index"]==0][col].values
    g2=data[data["index"]==1][col].values
    
   
    p_levene=stats.levene(g1, g2)[1]
    if p_levene>0.05:
        statistic,pvalue_ttest=stats.ttest_ind(g1, g2, equal_var = True)
    else:
        statistic,pvalue_ttest=stats.ttest_ind(g1, g2, equal_var = False)
        
    print("{:<70}: ttest- statistic={:>10.5f};pvalue={:>10.5f}".format(col,statistic,pvalue_ttest))


# The Urban population together with the last four variables has significant influence on index classification



# Use the feature of random forest model to output feature importance
# So here use random forest to build a classification model where x is the features and y is index

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(random_state=1)

x_cols=listpop
        
x=data[x_cols]
y=data['index']

classifier.fit(x,y)
feature_importances=classifier.feature_importances_

df_feature_importances=pd.DataFrame(zip(x_cols,feature_importances),columns=["features","importance"])
df_feature_importances



# Visualize the importance

plt.figure(figsize=(16,9))
sns.barplot(x="features",y="importance",data=df_feature_importances)
plt.xticks(rotation=-90)

