#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning: Getting Started
# ###### Iris Flower Dataset

# In[6]:


import sys
import scipy
import numpy as np
import pandas as pd
import matplotlib
import sklearn


# In[7]:


import pandas
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[10]:


# Loading the data
irs=pd.read_csv(r"C:\Users\Sneha\AppData\Local\Temp\Rar$DI00.135\IRIS.csv")


# In[12]:


# dimensions of the dataset
print(irs.shape)


# In[13]:


# take a peak at the data
irs.head(10)


# In[14]:


# statistical summary
print(irs.describe())


# In[15]:


# class distribution: show us the instances i.e. the no:of rows that belongs to each class
irs.groupby('species').size()


# ###### in this, we can see that each class/species has the same no:of instances

# ### Visualisation

# ###### 2 types of plot: 
# ######  i) Uni-variate plot: which helps us understand each attribute : box and whisker plots
# ######  ii) Multi-variate plot: which helps us understand the relationship b/w attributes

# In[17]:


# Univariate plots
irs.plot(kind='box',subplots=True ,layout=(2,2), sharex=False, sharey=False)


# In[18]:


# histogram of the variable
irs.hist()


# In[19]:


# Multivariate plots
## Scatter plots : to understand the structured relationships between the variables
scatter_matrix(irs)


# ### Evaluation of Algorithms
# ###### Create some model of the data and estimate their accuracy on unseen data.
# ###### Step 1: how to seperate out a validation set
# ###### Step 2: 10 fold cross validation test : to estimate the skill of machine learning model on unseen data
# ###### Step 3: To build multiple different models to predict species from flower measurements
# ###### Step 4: Select the best model out of the different models

# In[21]:


# Step 1: Creating a validation dataset
### splitting dataset
array= irs.values
X=array[:,0:4]
Y=array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size= 0.2, random_state=1)


# In[23]:


# Logistic regression
# linear discriminant analysis
# K- Nearest neighbors
# Classification and regression trees
# Gaussian Naive Bayes
# Support vector machines

# Building models
models =[]
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma= 'auto')))


# In[25]:


# Evaluate the created models
results=[]
names=[]
for name, model in models:
    kfold= StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s:%f (%f)'% (name, cv_results.mean(), cv_results.std()))
    


# ###### from this evaluation, SVM seems to have the greatest accuracy of 98%
# 

# In[26]:


# Comparing and checking for the best model
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# In[27]:


# Make predictions on SVM
model= SVC( gamma='auto')
model.fit(X_train, Y_train)
predictions= model.predict(X_validation)


# #### Evaluating the Prediction:
# ###### We can evaluate the prediction by comparing them to the expected results in the validation set and then calculate the classification accuracy as well as the confusion matrix to check the classification report

# In[28]:


# Evaluate our prediction
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




