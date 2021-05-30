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


# index 里面有一个缺失值，后面把它删除了

data.isnull().sum()


# In[5]:


data=data.dropna()


# In[6]:


# 把index 分组

data['index']=data['index'].map({1:0,2:0,3:1,4:1})


# In[7]:


# 这是各个变量 跟 'SDG 6.4.2. Water Stress (%)' 的相关系数，看起来是基本不相关，后面就不看他了哈

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


# 当index 分别取 0 1 情况下各个特征的均值

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


# 以index为目标，看起来有的变量的分布 当index分布取 0 1 的时候，分布还是有差异的

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


# 使用独立样本的t检验

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


# + 这么看起来 urban + 后面四个变量，在index 分别取 0 1 时，均值是有显著差异的(p值小于 0.05)

# In[ ]:





# In[18]:


# 下面我再给你搞个模型来看特征重要性大小的吧
# 因为随机森林模型有个特点 就是可以输出特征重要性
# 所以这里使用随机森林构建x为你前面设置的特征 y为index的分类模型

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


# 使用柱状图看每个变量的重要性大小

plt.figure(figsize=(16,9))
sns.barplot(x="features",y="importance",data=df_feature_importances)
plt.xticks(rotation=-90)


# In[ ]:




