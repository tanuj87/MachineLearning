
# coding: utf-8

# In[9]:

# project to implement recommender system in Python using correlations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:

columns_names = ['user_id', 'item_id', 'rating', 'timestamp']


# In[3]:

df = pd.read_csv('u.data', sep='\t', names = columns_names)


# In[4]:

df.head()


# In[5]:

movie_titles = pd.read_csv('Movie_Id_Titles')


# In[6]:

movie_titles.head()


# In[7]:

df = pd.merge(df, movie_titles, on='item_id')


# In[8]:

df.head()


# In[10]:

sns.set_style('white')


# In[12]:

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[13]:

df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[14]:

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[15]:

ratings.head()


# In[16]:

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[17]:

ratings.head()


# In[18]:

ratings['num of ratings'].hist(bins = 70)


# In[19]:

plt.show()


# In[20]:

ratings['rating'].hist(bins = 70)
plt.show()


# In[21]:

sns.jointplot(x='rating', y='num of ratings', data = ratings, alpha = 0.5)
plt.show()


# In[22]:

# convert this into matrix with movie rating on one side and user id on one side


# In[23]:

moviemat = df.pivot_table(index = 'user_id', columns='title', values = 'rating')


# In[24]:

moviemat


# In[25]:

# lots of NaNs because most of the prople habve not seen most of the movies


# In[26]:

ratings.sort_values('num of ratings', ascending=False).head(10)


# In[27]:

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']


# In[28]:

starwars_user_ratings


# In[30]:

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)


# In[31]:

similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)


# In[32]:

corr_starwars  = pd.DataFrame(similar_to_starwars, columns = ['Correlation'])
corr_starwars.dropna(inplace = True)


# In[33]:

corr_starwars.head()


# In[34]:

corr_starwars.sort_values('Correlation', ascending=False).head(10)


# In[35]:

corr_starwars = corr_starwars.join(ratings['num of ratings'])


# In[36]:

corr_starwars.head()


# In[37]:

corr_starwars[corr_starwars['num of ratings'] > 100].sort_values('Correlation', ascending=False).head(10)


# In[38]:

corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])


# In[40]:

corr_liarliar.dropna(inplace=True)


# In[41]:

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])


# In[42]:

corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values('Correlation', ascending = False).head()


# In[ ]:



