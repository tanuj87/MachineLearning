
# coding: utf-8

# ___
# 
# ___
# # Support Vector Machines Project 
# 
# Welcome to your Support Vector Machine Project! Just follow along with the notebook and instructions below. We will be analyzing the famous iris data set!
# 
# ## The Data
# For this series of lectures, we will be using the famous [Iris flower data set](http://en.wikipedia.org/wiki/Iris_flower_data_set). 
# 
# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
# 
# Here's a picture of the three different Iris types:

# In[17]:

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[18]:

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[19]:

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
#     Iris-setosa (n=50)
#     Iris-versicolor (n=50)
#     Iris-virginica (n=50)
# 
# The four features of the Iris dataset:
# 
#     sepal length in cm
#     sepal width in cm
#     petal length in cm
#     petal width in cm
# 
# ## Get the data
# 
# **Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') **

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
iris = sns.load_dataset('iris')


# Let's visualize the data and get you started!
# 
# ## Exploratory Data Analysis
# 
# Time to put your data viz skills to the test! Try to recreate the following plots, make sure to import the libraries you'll need!
# 
# **Import some libraries you think you'll need.**

# In[12]:

iris.head()


# ** Create a pairplot of the data set. Which flower species seems to be the most separable?**

# In[3]:

sns.pairplot(iris, hue = 'species')
plt.show()


# In[ ]:




# **Create a kde plot of sepal_length versus sepal width for setosa species of flower.**

# In[38]:

iris_setosa = iris[iris.species == 'setosa']
sns.kdeplot(iris_setosa['sepal_width'], iris_setosa['sepal_length'], cmap='plasma', shade=True, shade_lowest = False)
plt.show()


# In[ ]:




# # Train Test Split
# 
# ** Split your data into a training set and a testing set.**

# In[22]:

from sklearn.cross_validation import train_test_split
X = iris.drop('species', axis = 1)
y = iris['species']


# In[23]:

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 101, test_size = 0.3)


# # Train a Model
# 
# Now its time to train a Support Vector Machine Classifier. 
# 
# **Call the SVC() model from sklearn and fit the model to the training data.**

# In[24]:

from sklearn.svm import SVC


# In[25]:

svc_model = SVC()


# In[26]:

svc_model.fit(X_train, y_train)


# ## Model Evaluation
# 
# **Now get predictions from the model and create a confusion matrix and a classification report.**

# In[27]:

predictions = svc_model.predict(X_test)


# In[28]:

from sklearn.metrics import classification_report, confusion_matrix


# In[29]:

print(confusion_matrix(y_test, predictions))


# In[ ]:




# In[30]:

print(classification_report(y_test, predictions))


# In[ ]:




# Wow! You should have noticed that your model was pretty good! Let's see if we can tune the parameters to try to get even better (unlikely, and you probably would be satisfied with these results in real like because the data set is quite small, but I just want you to practice using GridSearch.

# ## Gridsearch Practice
# 
# ** Import GridsearchCV from SciKit Learn.**

# In[31]:

from sklearn.grid_search import GridSearchCV


# **Create a dictionary called param_grid and fill out some parameters for C and gamma.**

# In[32]:

param_grid = {'C':[0.1,1,10,100, 1000], 'gamma' : [1,0.1, 0.01, 0.001, 0.0001]}


# ** Create a GridSearchCV object and fit it to the training data.**

# In[33]:

grid = GridSearchCV(SVC(), param_grid, verbose =3)


# In[34]:

grid.fit(X_train, y_train)


# In[35]:

grid_predictions = grid.predict(X_test)


# ** Now take that grid model and create some predictions using the test set and create classification reports and confusion matrices for them. Were you able to improve?**

# In[36]:

print(confusion_matrix(y_test, grid_predictions))


# In[ ]:




# In[37]:

print(classification_report(y_test, grid_predictions))


# In[ ]:




# You should have done about the same or exactly the same, this makes sense, there is basically just one point that is too noisey to grab, which makes sense, we don't want to have an overfit model that would be able to grab that.

# 

# In[ ]:



