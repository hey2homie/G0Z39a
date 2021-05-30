#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[35]:


usefull_cols=[
    "country",
    "Population density (inhab/km2)",
    "GDP per capita (current US$/inhab)",
    "Agriculture, value added (% GDP) (%)",
    "Total renewable water resources per capita (m3/inhab/yr)",
    "Total population with access to safe drinking-water (JMP) (%)",
    "SDG 6.4.2. Water Stress (%)",
    "Agricultural water withdrawal as % of total water withdrawal (%)",
    "Industrial water withdrawal (km<sup>3</sup>/year or 10<sup>9</sup>m<sup>3</sup>/year)",
    "National Rainfall Index (NRI) (mm/yr)",
    "latitude",
    "Population ages 65 and above(-of total population)"
]

new_col_names=[
    "country",
     "Population density",
    "GDP per capita",
    "Agriculture(% GDP)",
    "Total renewable water resources per capita",
    "Total population with access to safe drinking-water",
    "Water Stress",
    "Agricultural water withdrawal of total water withdrawal",
    "Industrial water withdrawal ",
    "National Rainfall Index ",
    "latitude",
    "Population ages 65 and above"
]


data=pd.read_csv("AveragedWhole2017.csv",usecols=usefull_cols)
data=data[usefull_cols]
data.columns=new_col_names


# In[36]:


data=data.set_index("country")


# In[37]:


col1=data.columns
col1
data=data[col1].copy()
data.head()
# See the distribution of everyone

for col in [
     "Population density",
    "GDP per capita",
    "Agriculture(% GDP)",
    "Total renewable water resources per capita",
    "Total population with access to safe drinking-water",
    "Water Stress",
    "Agricultural water withdrawal of total water withdrawal",
    "Industrial water withdrawal ",
    "National Rainfall Index ",
    "latitude",
    "Population ages 65 and above"]:
    fig,axs=plt.subplots(1,2,figsize=(15,5))
    sns.histplot(data=data,x=col,ax=axs[0])
    axs[0].set_title("Historgram of {}".format(col),fontsize=12)
        
    sns.histplot(np.log1p(data[col]),ax=axs[1])
    axs[1].set_title("Historgram of log-{}".format(col),fontsize=12)
    plt.show()


# In[38]:

cols=[   "Population density",
    "GDP per capita",
    "Agriculture(% GDP)",
    "Total renewable water resources per capita",
    "Total population with access to safe drinking-water",
    "Water Stress",
    "Agricultural water withdrawal of total water withdrawal",
    "Industrial water withdrawal ",
    "National Rainfall Index ",
    "latitude",
    "Population ages 65 and above"]
fig,axs=plt.subplots(6,2,figsize=(15,20))
axs=axs.flatten()
for i in range(11):
    sns.boxplot(x=cols[i],data=data,ax=axs[i])
plt.show()


# In[39]:


cols=[  "Population density",
    "GDP per capita",
    "Agriculture(% GDP)",
    "Total renewable water resources per capita",
    "Total population with access to safe drinking-water",
    "Water Stress",
    "Agricultural water withdrawal of total water withdrawal",
    "Industrial water withdrawal ",
    "National Rainfall Index ",
    "latitude",
    "Population ages 65 and above"]
for col in cols:
    q1=data[col].quantile(0.25)
    q3=data[col].quantile(0.75)
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    
    data[col]=data[col].map(lambda x:x if x> lower else lower)
    data[col]=data[col].map(lambda x:x if x< upper else upper)


# In[40]:


for col in [ "Population density",
    "GDP per capita",
    "Agriculture(% GDP)",
    "Total renewable water resources per capita",
    "Total population with access to safe drinking-water",
    "Water Stress",
    "Agricultural water withdrawal of total water withdrawal",
    "Industrial water withdrawal ",
    "National Rainfall Index ",
    "latitude",
    "Population ages 65 and above"]:
    plt.figure(figsize=(9,6))
    sns.histplot(x=col,data=data)
    plt.title("Historgram of {}".format(col),fontsize=12)
    
    plt.grid()
    plt.show()


# In[41]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score




X=data[["Population density",
    "GDP per capita",
    "Agriculture(% GDP)",
    "Total renewable water resources per capita",
    "Total population with access to safe drinking-water",
    "Water Stress",
    "Agricultural water withdrawal of total water withdrawal",
    "Industrial water withdrawal ",
    "National Rainfall Index ",
    "latitude",
    "Population ages 65 and above"]]

sse_list = []  # Store the sum of SSE

for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=1)
    km.fit(X)
    
    # Collect sse
    sse_list.append(km.inertia_)

    # Get silhouette_score
    cluster_labels = km.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    print('k={}时，Silhouette coefficient：{:0.2f}'.format(k,silhouette_avg))
    #Here we can see that when the coefficient equals 2, it's highest, but 6,7,8,9 are equally the second good




# In[50]:


plt.figure(figsize=(9,6))

plt.plot(range(2,11),sse_list,'o-')
plt.xlabel('K',fontsize=12)
plt.ylabel('SSE',fontsize=12)
plt.title("Cluster SSE by Different K",fontsize=14)
plt.grid()
plt.show()
#To quantify the quality of clustering, intrinsic metrics - such as the within-cluster (SSE) distortion - to compare the performance of different k-means clusterings.
#Intuitively, if k increases, the distortion will decrease. This is because the samples will be closer to the centroids they are assigned to. The idea behind the elbow 
#method is to identify the value of k where the distortion begins to increase most rapidly, which will be clearer if the distortion for different values k is depicted.
#Here it shows that the k is around 5


# In[51]:


km = KMeans(n_clusters=5, random_state=1)
km.fit(X)
data["labels"]=km.labels_
data.head()


# In[52]:


#how many countries in every cluster

data["labels"].value_counts().sort_index()


# In[53]:


#See the difference between every groups
cluster_stats=data.groupby("labels")[["Population density",
    "GDP per capita",
    "Agriculture(% GDP)",
    "Total renewable water resources per capita",
    "Total population with access to safe drinking-water",
    "Water Stress",
    "Agricultural water withdrawal of total water withdrawal",
    "Industrial water withdrawal ",
    "National Rainfall Index ",
    "latitude",
    "Population ages 65 and above"]].mean()
cluster_stats


# In[ ]:





# In[58]:


plt.figure(figsize=(21,11))
sns.heatmap(cluster_stats, annot=True, fmt="0.2f",linewidths=.5,cmap="YlGnBu", cbar=True)
plt.xticks(rotation=-90,fontsize=12)
plt.show()


# In[60]:


#data.to_csv("result.csv",encoding="utf-8")

#Compare the index of Mianyong's model with the clustering results
# In[ ]:
data1=pd.read_csv("final_data.csv")
data=data.reset_index("country")

datamerged=pd.merge(data[["country","labels"]],data1[["country","water security index"]],on="country")
datamerged

table1=datamerged.pivot_table(index="labels",columns=["water security index"],values="country",aggfunc="count")
plt.figure(figsize=(10,10))
sns.heatmap(table1, annot=True, fmt="0.2f",linewidths=.5,cmap="YlGnBu", cbar=False)
plt.xticks(rotation=0,fontsize=12)
plt.show()

