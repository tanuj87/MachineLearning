
# coding: utf-8

# # Natural Language Processing Project
# 
# Welcome to the NLP Project for this section of the course. In this NLP project you will be attempting to classify Yelp Reviews into 1 star or 5 star categories based off the text content in the reviews. This will be a simpler procedure than the lecture, since we will utilize the pipeline methods for more complex tasks.
# 
# We will use the [Yelp Review Data Set from Kaggle](https://www.kaggle.com/c/yelp-recsys-2013).
# 
# Each observation in this dataset is a review of a particular business by a particular user.
# 
# The "stars" column is the number of stars (1 through 5) assigned by the reviewer to the business. (Higher stars is better.) In other words, it is the rating of the business by the person who wrote the review.
# 
# The "cool" column is the number of "cool" votes this review received from other Yelp users. 
# 
# All reviews start with 0 "cool" votes, and there is no limit to how many "cool" votes a review can receive. In other words, it is a rating of the review itself, not a rating of the business.
# 
# The "useful" and "funny" columns are similar to the "cool" column.
# 
# Let's get started! Just follow the directions below!

# ## Imports
#  **Import the usual suspects. :) **

# In[42]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## The Data
# 
# **Read the yelp.csv file and set it as a dataframe called yelp.**

# In[4]:

yelp = pd.read_csv('yelp.csv')


# ** Check the head, info , and describe methods on yelp.**

# In[6]:

yelp.head()


# In[7]:

yelp.info()


# In[8]:

yelp.describe()


# **Create a new column called "text length" which is the number of words in the text column.**

# In[11]:

yelp['text length'] = yelp['text'].apply(len)


# In[12]:

yelp.head()


# # EDA
# 
# Let's explore the data
# 
# ## Imports
# 
# **Import the data visualization libraries if you haven't done so already.**

# In[101]:




# **Use FacetGrid from the seaborn library to create a grid of 5 histograms of text length based off of the star ratings. Reference the seaborn documentation for hints on this**

# In[13]:

g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length')
plt.show()


# **Create a boxplot of text length for each star category.**

# In[14]:

sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
plt.show()


# **Create a countplot of the number of occurrences for each type of star rating.**

# In[15]:

sns.countplot(x = 'stars', data=yelp)
plt.show()


# ** Use groupby to get the mean values of the numerical columns, you should be able to create this dataframe with the operation:**

# In[26]:

stars = yelp.groupby('stars').mean()
print(stars)


# **Use the corr() method on that groupby dataframe to produce this dataframe:**

# In[22]:

stars.corr()


# **Then use seaborn to create a heatmap based off that .corr() dataframe:**

# In[24]:

sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)
plt.show()


# ## NLP Classification Task
# 
# Let's move on to the actual task. To make things a little easier, go ahead and only grab reviews that were either 1 star or 5 stars.
# 
# **Create a dataframe called yelp_class that contains the columns of yelp dataframe but for only the 1 or 5 star reviews.**

# In[32]:

yelp_class = yelp[(yelp.stars == 1) | (yelp.stars ==5)]


# ** Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class. (Your features and target/labels)**

# In[33]:

X = yelp_class.text
y = yelp_class.stars


# **Import CountVectorizer and create a CountVectorizer object.**

# In[34]:

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer()


# ** Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.**

# In[35]:

X = bow_transformer.fit_transform(X)


# In[38]:

print(X)


# ## Train Test Split
# 
# Let's split our data into training and testing data.
# 
# ** Use train_test_split to split up the data into X_train, X_test, y_train, y_test. Use test_size=0.3 and random_state=101 **

# In[36]:

from sklearn.model_selection import train_test_split


# In[39]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =101)


# ## Training a Model
# 
# Time to train a model!
# 
# ** Import MultinomialNB and create an instance of the estimator and call is nb **

# In[40]:

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# **Now fit nb using the training data.**

# In[41]:

nb.fit(X_train,y_train)


# ## Predictions and Evaluations
# 
# Time to see how our model did!
# 
# **Use the predict method off of nb to predict labels from X_test.**

# In[43]:

predictions = nb.predict(X_test)


# ** Create a confusion matrix and classification report using these predictions and y_test **

# In[44]:

from sklearn.metrics import classification_report, confusion_matrix


# In[45]:

print(confusion_matrix(y_test, predictions))
print("\n")
print(classification_report(y_test, predictions))


# In[125]:

print(confusion_matrix(y_test, predictions))
print("\n")
print(classification_report(y_test, predictions))


# **Great! Let's see what happens if we try to include TF-IDF to this process using a pipeline.**

# # Using Text Processing
# 
# ** Import TfidfTransformer from sklearn. **

# In[46]:

from sklearn.feature_extraction.text import TfidfTransformer


# ** Import Pipeline from sklearn. **

# In[48]:

from sklearn.pipeline import Pipeline


# ** Now create a pipeline with the following steps:CountVectorizer(), TfidfTransformer(),MultinomialNB()**

# In[49]:

pipeline = Pipeline([
    ('bow', CountVectorizer()),# strings to token integer counts
    ('tfidf', TfidfTransformer()),# integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB())# train on TF-IDF vectors w/ Naive Bayes classifier
])


# ## Using the Pipeline
# 
# **Time to use the pipeline! Remember this pipeline has all your pre-process steps in it already, meaning we'll need to re-split the original data (Remember that we overwrote X as the CountVectorized version. What we need is just the text**

# ### Train Test Split
# 
# **Redo the train test split on the yelp_class object.**

# In[50]:

X = yelp_class.text
y = yelp_class.stars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)


# **Now fit the pipeline to the training data. Remember you can't use the same training data as last time because that data has already been vectorized. We need to pass in just the text and labels**

# In[51]:

pipeline.fit(X_train, y_train)


# ### Predictions and Evaluation
# 
# ** Now use the pipeline to predict from the X_test and create a classification report and confusion matrix. You should notice strange results.**

# In[52]:

pipe_predictions = pipeline.predict(X_test)


# In[54]:

print(confusion_matrix(y_test, pipe_predictions))
print("\n")
print(classification_report(y_test, pipe_predictions))

