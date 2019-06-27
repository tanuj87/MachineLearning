# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/Users/tanuj1024805/Documents/Machine Learning/Data_Preprocessing/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Taking care of missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)#Strategy can be "mean", "median", "most_frequent"
imputer = imputer.fit(X[:,1:3])# we need to repalce missing values from Age and Salary column, so index 1 and 2(in python index is excluded), 3 is written because upper bound is excluded in python
#imputer object is now fitted to Matrix X
X[:,1:3] = imputer.transform(X[:,1:3])# this is the method that is going to replace the missing data

#Taking care of categorical Data

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:,0])
