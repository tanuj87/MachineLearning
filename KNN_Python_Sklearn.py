
# coding: utf-8

# In[1]:

# K Nearest Neighbours with Python


# In[2]:

import pandas as pd
import numpy as np


# In[4]:

import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:

df = pd.read_csv('Classified Data', index_col=0)


# In[10]:

df.head()


# In[11]:

# standardize everything to a same scale


# In[12]:

from sklearn.preprocessing import StandardScaler


# In[13]:

scaler = StandardScaler()


# In[15]:

scaler.fit(df.drop('TARGET CLASS', axis=1))
# we dont want of it it to our target class


# In[16]:

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))


# In[18]:

# scaled version of our actual values
scaled_features


# In[21]:

df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])


# In[22]:

df_feat.head()


# In[23]:

#Train test split


# In[24]:

from sklearn.cross_validation import train_test_split


# In[25]:

X = df_feat
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 101, test_size = 0.3)


# In[26]:

# elbow method to choose a K value


# In[27]:

from sklearn.neighbors import KNeighborsClassifier


# In[28]:

knn = KNeighborsClassifier(n_neighbors=1)


# In[29]:

knn.fit(X_train, y_train)


# In[30]:

pred = knn.predict(X_test)


# In[31]:

pred


# In[32]:

# Evaluation of our KNN method


# In[33]:

from sklearn.metrics import classification_report, confusion_matrix


# In[34]:

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# In[35]:

# We will try to choose an even better K method, using the elbow method


# In[37]:

error_rate = []

for i in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error_rate.append(np.mean(pred_i != y_test))


# In[45]:

plt.figure(figsize = (10,6))
plt.plot(range(1,40), error_rate, color='blue', linestyle = 'dashed', marker='o', markerfacecolor = 'red', markersize = 10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('error_rate')
plt.show()


# In[46]:

# We can see here that the error rate is lower at 18, 34 etc
# lets go ahead and choose a higher K value for lower error rate


# In[47]:

knn = KNeighborsClassifier(n_neighbors=17)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# In[ ]:

# we were able to classify a couple of pore points correctly and our score also improved

