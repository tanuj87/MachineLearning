
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:

train = pd.read_csv('titanic_train.csv')


# In[4]:

train.head()


# In[5]:

# to cehck if we have NULL data
train.isnull()
# with this sort of DB we can actually make a heatmap


# In[9]:

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
# there is missing data in AGE column that we will have to fill in
# there is too much missing data in Cabin Column and we will have to probably drop this column


# In[11]:

# some more exploratory data analysis at visual level


# In[12]:

sns.set_style('whitegrid')


# In[16]:

sns.countplot(x='Survived', data = train)
plt.show()


# In[20]:

sns.countplot(x='Survived', data = train, hue='Sex')
plt.show()
# we can tell the trend that males are much less likely to have survived
# however there are about double females who survived as compared to who ahve died


# In[21]:

sns.countplot(x='Survived', data = train, hue='Pclass', )
plt.show()
# the prople who did not survive are overwhelmingly from class 3 (cheap tickets)
# the people who did survive, were leaning a litle more towards higher classes


# In[22]:

sns.distplot(train['Age'].dropna(), kde=False, bins = 30)
plt.show()


# In[24]:

train['Age'].plot.hist(bins = 35)
plt.show()


# In[25]:

train.info()


# In[27]:

sns.countplot(x='SibSp', data = train)
plt.show()
# this shows lot of single people on board, 
# 1 either sibling or spouse


# In[30]:

train['Fare'].hist(bins= 40, figsize=(10,4))
plt.show()
# most of the prices re distributed between 0 and 50


# In[31]:

# Data Cleansing


# In[32]:

plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y = 'Age', data=train)
plt.show()


# In[33]:

# we can use the average age values based on classes to impute age where it is null, 
# this is little better than just putting in average age


# In[35]:

#Function to compute average age in case of null based on Passanger Class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[36]:

#Lets apply this
train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
#axis = 1 because we want o aply this across the columns


# In[42]:

# lets check that heatmap again
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()


# In[38]:

# lets drop cabin column
train.drop('Cabin', axis=1, inplace = True)


# In[39]:

train.head()


# In[41]:

# remove other duoplicare values
train.dropna(inplace=True)


# In[43]:

# converting categorical features into dummy variables using pandas


# In[46]:

sex = pd.get_dummies(train['Sex'], drop_first=True)


# In[47]:

embarked = pd.get_dummies(train['Embarked'], drop_first=True)


# In[48]:

embarked.head()


# In[49]:

train = pd.concat([train, sex, embarked],axis = 1)


# In[50]:

train.head()


# In[51]:

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace=True)


# In[ ]:




# In[52]:

train.head()


# In[53]:

# passangerId is just an index, even though numerical it just and index and of no use


# In[56]:

train.drop(['PassengerId'], axis = 1, inplace=True)


# In[57]:

train.head()


# In[58]:

X = train.drop('Survived', axis=1)
y = train['Survived']


# In[59]:

from sklearn.model_selection import train_test_split


# In[60]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# In[61]:

from sklearn.linear_model import LogisticRegression


# In[62]:

logmodel = LogisticRegression()


# In[63]:

logmodel.fit(X_train, y_train)


# In[64]:

predictions = logmodel.predict(X_test)


# In[65]:

from sklearn.metrics import classification_report


# In[66]:

# this will tell us our Precisiom Recall, F1 score etc/ 


# In[67]:

print(classification_report(y_test, predictions))


# In[68]:

from sklearn.metrics import confusion_matrix


# In[77]:

confusion_matrix(y_test, predictions)


# In[90]:

# comparing results
comparison = pd.DataFrame({"Predicted_value" : predictions, "Actual_values" : y_test})


# In[91]:

comparison


# In[94]:

# Now configuring class column as Dummy Value
Class = pd.get_dummies(train['Pclass'], drop_first=True)


# In[96]:

train = pd.concat([train, Class],axis = 1)


# In[97]:

train.head()


# In[98]:

train.drop('Pclass', axis = 1, inplace=True)


# In[99]:

train.head()


# In[100]:

X_new = train.drop('Survived', axis=1)
y_new = train['Survived']


# In[101]:

X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, y_new, test_size = 0.3, random_state = 101)


# In[102]:

logmodel_new = LogisticRegression()


# In[104]:

logmodel_new.fit(X_new_train, y_new_train)


# In[105]:

predict_new = logmodel_new.predict(X_new_test)


# In[106]:

print(classification_report(y_new_test, predict_new))


# In[107]:

confusion_matrix(y_new_test, predict_new)


# In[ ]:



