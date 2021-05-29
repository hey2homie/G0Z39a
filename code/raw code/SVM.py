#!/usr/bin/env python
# coding: utf-8

# In[156]:


from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import validation_curve
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline


# In[157]:


data=pd.read_csv('../../data/final_data/newdata.csv')
y=data['water security index']
x=data.iloc[:,2:]
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=0)


# In[158]:


clf = svm.SVC(decision_function_shape='ovo')
model_svm=clf.fit(X_train, Y_train)
Accuracy_test=model_svm.score(X_test, Y_test)
Accuracy_train=model_svm.score(X_train, Y_train)
print(Accuracy_test)#59.1%
print(Accuracy_train)#65.1%


# In[159]:



clist_v = [2**(-5),2**(-4),2**(-3),2**(-2),0.5,1,2,4,8,16,32,64,128,256,512,1024,2048,2**12]
gammalist_v=[2**(-12),2**(-11),2**(-10),2**(-9),2**(-8),2**(-7),2**(-6),2**(-5),2**(-4),2**(-3),2**(-2),0.5,1,2,4,8]
pipe = Pipeline([
            ('clf', model_svm)
                ])
grid_params  = [{
            'clf__C' : clist_v,
            'clf__gamma':gammalist_v
        },]
f1 = make_scorer(f1_score , average='weighted')
grid_search = GridSearchCV(pipe, param_grid=grid_params, cv=3, scoring=f1, n_jobs=2,return_train_score=True,verbose=3)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
clf=svm.SVC(decision_function_shape='ovo',C=64,gamma=0.03125)
model_svm=clf.fit(X_train, Y_train)
Accuracy_test=model_svm.score(X_test, Y_test)
Accuracy_train=model_svm.score(X_train, Y_train)
print(Accuracy_test)#86.4%
print(Accuracy_train)#76%

