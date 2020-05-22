
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:

from sklearn.datasets import load_breast_cancer


# In[3]:

cancer = load_breast_cancer()


# In[4]:

cancer.keys()


# In[6]:

print(cancer['DESCR'])


# In[8]:

df_feats = pd.DataFrame(cancer['data'], columns =cancer['feature_names'])


# In[10]:

df_feats.head(2)


# In[11]:

df_feats.info()


# In[12]:

cancer['target']


# In[13]:

cancer['target_names']


# In[14]:

from sklearn.cross_validation import train_test_split


# In[16]:

X = df_feats
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[17]:

from sklearn.svm import SVC


# In[18]:

model = SVC()


# In[19]:

model.fit(X_train, y_train)


# In[21]:

predictions = model.predict(X_test)


# In[22]:

from sklearn.metrics import classification_report, confusion_matrix


# In[24]:

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))


# In[25]:

# as we can see our model predicted evrything is of class 1 type, and it classified nothing in class 0
# this is visible from the warning message too "F-Score is ill defined"

# our model needs to have its parameters adjusted
# it may also help to mormalize the data as well, when you are passing it into support vector Machine
# we can go ahead and search for the best parameters using a Grid Search
# Grid search allows you to find the best parameters , such as the C or Gamma values



# In[26]:

from sklearn.grid_search import GridSearchCV


# In[27]:

param_grid = {'C':[0.1,1,10,100, 1000], 'gamma' : [1,0.1, 0.01, 0.001, 0.0001]}


# In[28]:

grid = GridSearchCV(SVC(), param_grid, verbose=3)


# In[29]:

grid.fit(X_train, y_train)


# In[30]:

grid.best_params_


# In[31]:

grid.best_estimator_


# In[32]:

grid.best_score_


# In[33]:

grid_predictions = grid.predict(X_test)


# In[34]:

print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))


# In[ ]:

# now we can see the the recults are good

