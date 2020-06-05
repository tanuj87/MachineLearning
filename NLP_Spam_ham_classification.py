
# coding: utf-8

# In[1]:

import nltk


# In[2]:

nltk.download_shell()


# In[3]:

# we will be using a dataset frpom the UCI dataset


# In[3]:

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[4]:

print(len(messages))


# In[5]:

messages[0]


# In[6]:

import pandas as pd


# In[7]:

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep = '\t', names=['label', 'message'])


# In[8]:

messages.head() 


# In[9]:

messages.describe()


# In[10]:

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')


# In[26]:

messages[0]


# In[33]:

# calling describe on dataframe


# In[34]:

messages.describe()


# In[35]:

# we will use group by to geta higher level view of the dataset


# In[11]:

messages.groupby('label').describe()


# In[37]:

# Most popular ham message is "Sorry, I'll call later "
# Most popular spam message is "Please call our customer service representativ"


# In[38]:

# we need to think about the features we are going to be using
# A large part of NLP is going to be feature_engineering
# the more your knowledge/domain knowledge about data, the better your abimity to engineer more features form it
# Very important for spam detection


# In[39]:

# Creating a new column to detect hoe large a text message are


# In[12]:

messages['length'] = messages['message'].apply(len)


# In[13]:

messages.head()


# In[14]:

# lets visualize the length of the message


# In[15]:

import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:

messages['length'].plot.hist(bins=150)
plt.show()


# In[49]:

# We can see there are some text messages with length more than 800, lets see them


# In[17]:

messages['length'].describe()


# In[18]:

# there iis messge with max length of 910
# let go ahead an view that message


# In[21]:

messages[messages['length'] == 910]['message'].iloc[0]


# In[22]:

# this si kind of a wierd love letter


# In[23]:

# lets explore outliers


# In[24]:

# histogram


# In[25]:

messages.hist(column='length', by='label', bins=60, figsize=(12,4))
plt.show()


# In[66]:

# on the X axis we have length of the messages
# on the y axis we have count of those messages
# as we can see here is a trend, most of the ham messages have low length, around 50
# most fo the spam messages have higher length , around 150


# In[26]:

# now we will covert raw messages which are a sequence of characters into a vectors : a sequence of numbers


# In[27]:

# split the message into individual words and then return a list


# In[28]:

import string
# first thing we want to do is remove punctuations


# In[29]:

mess = 'Sample Message! Notice: it has punctuation.'


# In[30]:

string.punctuation


# In[31]:

# Now I can use list comprehension in order to pass on for every character and check 
# if it is not in the string punctuation


# In[33]:

nopunc = [c for c in mess if c not in string.punctuation]


# In[34]:

nopunc


# In[35]:

# punctuations removed


# In[36]:

from nltk.corpus import stopwords


# In[37]:

stopwords.words('english')


# In[38]:

nopunc = ''.join(nopunc)


# In[39]:

nopunc


# In[40]:

nopunc.split()


# In[41]:

clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[42]:

clean_mess


# In[43]:

# lets put this remving punctuations and removing stop words into a nice fucntion 
# so that we can apply this to our entire dataset


# In[44]:

def text_process(mess):
    """
    1. remove punctuations
    2. remove stopwords
    3. return list of clean text words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[45]:

# we will tokenize these messages
# cleaned  version of the words


# In[46]:

messages['message'].head(5).apply(text_process)


# In[47]:

# these are tokenized version


# In[48]:

# this is a very simple version of tokenization
# NLTK library has very great tools to performt this
# for example STEMMING is a very common way to continue processing data


# In[49]:

# Stemming : Running, ran , run
# stemming will return Run
# some of these text processing methods have trouble with short hand
# and we have a lot of short hand in our data like Nah, U, C etc


# In[50]:

# vectorization!!


# In[51]:

# we will be using bag of words methods to calculate
# 1. term frequency
# 2. TFIDF
# we will be using sklearn's CountVectorizer
# this will give us a matrix with all the words in vocab as rows and 
# all the messages a columns and counts will be the data
# this will outout a sparse matrix : lot of zero values


# In[52]:

from sklearn.feature_extraction.text import CountVectorizer


# In[53]:

bow_transformer = CountVectorizer(analyzer = text_process).fit(messages['message'])


# In[54]:

# lets check how many words we have in our vocablary


# In[55]:

print(len(bow_transformer.vocabulary_))


# In[56]:

mess4 = messages['message'][3]


# In[57]:

print(mess4)


# In[58]:

bow4 = bow_transformer.transform([mess4])


# In[59]:

print(bow4)


# In[62]:

print(bow4.shape)


# In[63]:

bow_transformer.get_feature_names()[4068]


# In[64]:

bow_transformer.get_feature_names()[9554]


# In[65]:

# transform the entire dataset


# In[67]:

messages_bow = bow_transformer.transform(messages['message'])


# In[68]:

print('Shape of matrix : ', messages_bow.shape)


# In[70]:

messages_bow.nnz # non zero occurances


# In[71]:

# we can also check the sparcity


# In[74]:

# TFIDF transformer from sklearn


# In[75]:

from sklearn.feature_extraction.text import TfidfTransformer


# In[76]:

tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[79]:

tfidf4 = tfidf_transformer.transform(bow4)


# In[80]:

print(tfidf4) # these are weight values for each of these words vs the actual document


# In[81]:

# to check TFIDF value for a particulat word:


# In[82]:

tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]


# In[83]:

# now lets convert entire bow corpus into tfidf at once


# In[84]:

messages_tfidf = tfidf_transformer.transform(messages_bow)


# In[85]:

# we will be using Naive Bayes classifier


# In[86]:

from sklearn.naive_bayes import MultinomialNB


# In[87]:

spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# In[89]:

spam_detect_model.predict(tfidf4)[0]


# In[90]:

all_pred = spam_detect_model.predict(messages_tfidf)


# In[91]:

# we just used all our data for training, amnd we should not do that
# we should always do train test split
# lets go ahead and do that


# In[92]:

from sklearn.cross_validation import train_test_split


# In[93]:

msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size = 0.3)


# In[94]:

#  the above data contains plain text messages without any processing that we just did above
# no bow, no tf, no tfidf
# so one way is to do all this again
# or we can use sklearn;s pipeline feature


# In[95]:

from sklearn.pipeline import Pipeline


# In[96]:

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer = text_process)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


# In[98]:

pipeline.fit(msg_train, label_train)


# In[99]:

predictions = pipeline.predict(msg_test)


# In[100]:

from sklearn.metrics import classification_report


# In[101]:

print(classification_report(label_test, predictions))


# In[ ]:



