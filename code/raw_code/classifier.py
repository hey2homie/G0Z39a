# -*- coding: utf-8 -*-
"""


@author: mianyong
"""


#build the classifier model


import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
#building model with the different methods and comparing the performance



"""AP2020=ace=pd.read_csv("2020APindex.csv")   
AP2020=AP2020.merge(pd.read_csv("2020whole.csv"),on="country",how="inner")   

#extract the 2020 Asian-pacific index and variables

AP2016=ace=pd.read_csv("2016APindex.csv")   
AP2016=AP2016.merge(pd.read_csv("2016whole.csv"),on="country",how="inner")   

AP2020.append(AP2016)

print(AP2020.shape)
print(AP2016.shape)

#check the cloumns of these two dataframe
main_list = np.setdiff1d(AP2016.columns,AP2020.columns)

print(main_list)


newdata=AP2020[AP2016.columns].append(AP2016)
print(newdata.columns)
"""

# newdata.to_csv('final_data.csv')
newdata = pd.read_csv("../../data/final_data/final_data.csv")
datas = newdata.values


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
    
#build a plot
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()



#rebuild the model based on the selected variables,select the varibales have the importance over 1%
srfc = SelectFromModel(rfc, threshold=0.05)
srfc.fit(X_train, y_train)
# Train the selected variables
X_important_train = srfc.transform(X_train)
X_important_test = srfc.transform(X_test)

rfc_important = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
rfc_important.fit(X_important_train, y_train)

y1_important_pred = rfc_important.predict(X_important_test)

performance_select=accuracy_score(y_test, y1_important_pred)
print("performance_select",performance_select)#70%

# rfc_sel_index the index of the selected variables
rfc_sel_index=srfc.get_support(indices=True)
print(newdata.columns[rfc_sel_index+1])


importance2=rfc_important.feature_importances_
featurename2=newdata.columns[rfc_sel_index+1]
for i in range(0,len(featurename2)):
	print(featurename2[i],importance2[i])
    

#use the gradient boost to build the model
gbc= GradientBoostingClassifier(n_estimators=1000, learning_rate=1,
        max_depth=1, random_state=0).fit(X_train, y_train)

perform_gbc=gbc.score(X_test, y_test)
print(perform_gbc)#70%


gbc_sel = SelectFromModel(gbc, threshold=0.01)
gbc_sel.fit(X_train, y_train)

#rebuild based on the selected variables
X_important_train = gbc_sel.transform(X_train)
X_important_test = gbc_sel.transform(X_test)
gbc_important = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0,max_depth=1, random_state=0)
gbc_important.fit(X_important_train, y_train)
y1_important_pred = gbc_important.predict(X_important_test)
perform_gbc_sel=accuracy_score(y_test, y1_important_pred)
print(perform_gbc_sel)#60%
