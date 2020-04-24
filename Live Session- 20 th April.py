#!/usr/bin/env python
# coding: utf-8

# ### Continuation of Titanic.csv

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"C:\Users\Sneha\Downloads\Titanicdf_train.csv")


# In[3]:


df[df.Age.isnull()]


# In[4]:


mean_age = df["Age"].mean()
std_age = df["Age"].std()
is_null_age = df["Age"].isnull().sum()

# to compute random numbers between the mean, std and is_null
# random values that fall between the min value i.e. (mean-std) and max value i.e.(mean+std)
rand_age = np.random.randint(mean-std, mean+std, size=is_null)

# fill NaN values in Age column with randon values generated
age_slice = df["Age"].copy()
age_slice[np.isnan(age_slice)]= rand_age
df["Age"] = age_slice
df["Age"] = df["Age"].astype(int)


# ### Working with Outliers
# ###### to ensure that the outliers are not affecting thr values of the entire population

# In[7]:


marks=[85,75,72,65,70]


# In[8]:


np.mean(marks)


# In[9]:


nmarks=[85,75,72,65,70,20,20]
np.mean(nmarks)


# ###### when 2 students secured very low marks of 20, the mean marks of the whole lot was brought down from 73.4 to 58.14. Therefore the 2 students with 20 are outliers which affect the whole population

# In[10]:


df.Age.mean()


# In[5]:


# use histogram to understand the distribution
df.Age.plot(kind='hist', bins=20, color='c');


# In[11]:


df.loc[df.Age>70]


# In[19]:


df.Fare.plot(kind='hist',title='histogram for fare',bins=20,color='c')


# In[13]:


df.Fare.mean()


# In[15]:


df.Fare.plot(kind='box')


# ###### all the values outside the box are outliers

# In[16]:


# t look into the outliers
df.loc[df.Fare==df.Fare.max()]


# In[17]:


# try some transformations to reduce the skewness
LogFare= np.log(df.Fare+1.0)   # Adding 1 to accomodate zero fares: log(0) is not defined


# In[18]:


# histogram of LogFare
LogFare.plot(kind='hist',color='c',bins=20)


# ###### now, the curve is more flattened or distributed

# In[20]:


# to create 4 diff classes
pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high']).value_counts().plot(kind='bar',color='c',rot=0)


# In[21]:


pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high']).value_counts()


# In[22]:


df['Fare_bin']=pd.qcut(df.Fare,4,labels=['very_low','low','high','very_high'])
df.head(10)


# ### Feature Engineering
# ###### combining two or more features to create a new feature

# In[23]:


# creating a new column: Agestate based on Age
df['AgeState']= np.where(df['Age']>=18, 'Adult', 'Child')


# In[24]:


# check for the counts
df.AgeState.value_counts()


# In[25]:


pd.crosstab(df.Survived,df.AgeState)


# In[26]:


# creating the family size: addind parents with children
df['FamilySize']=df.Parch+df.SibSp +1  # 1 for self


# In[27]:


# plotting the family feature
df['FamilySize'].plot(kind='hist',color='c')


# ###### people travelling alone were the maximum whereas people with families were less

# In[28]:


# further exploring the family size feature: finding family with max fmly membs
df.loc[df.FamilySize==df.FamilySize.max()]


# In[29]:


pd.crosstab(df.Survived,df.FamilySize)


# ###### the survival rate of bigger families is less than those with smaller families

# In[31]:


import seaborn as sns


# In[32]:


sns.pyplot(df)


# ### Creating a Model
# ##### Before creating a model, all the categorical values must be converted to num values in order for the machine to be able to read

# In[34]:


# labelEncoder(): takes the input column and convert the unique categorical value into coded numbers
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()


# In[35]:


df.info()


# ###### convert all object into num values

# In[36]:


df.drop('Cabin',inplace=True,axis=1)


# In[37]:


df.info()


# In[40]:


#converting the object values
df.iloc[:,4]=lb.fit_transform(df.iloc[:,4])


# In[41]:


df.head()


# In[45]:


df.iloc[:,11]=lb.fit_transform(df.iloc[:,11])
df.iloc[:,12]=lb.fit_transform(df.iloc[:,12])


# In[46]:


df.head()


# In[47]:


df.Embarked.fillna('C',inplace=True)


# In[48]:


df[df.Embarked.isnull()]


# In[49]:


df.iloc[:,10]=lb.fit_transform(df.iloc[:,10])


# In[50]:


df.head()


# ### Model building

# In[51]:


from sklearn.model_selection import train_test_split


# In[81]:


df.drop('Age',inplace=True,axis=1)


# ###### splitting the dataset into two categories: 80% for training and 20% for testing

# In[52]:


# 1st Step: creating two variables: dependent and independent variable
df.columns


# In[67]:


# y is dependent variable
y=df.loc[:,'Survived']
x=df.loc[:,['Pclass','Age','Sex', 'Embarked', 'Fare_bin', 'AgeState','FamilySize']]


# In[68]:


x


# In[69]:


y


# In[71]:


# splitting the values for testing and training
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=101)
# random_state is ggiven because each time we run the test train split, it could be generating random values which could affect the ouput
# the train_test_split( returns 4 outputs: x_train,x_test,y_train,y_test)


# ###### Why are we splitting: we need to train our data and we also need to test the data to check the accuracy of the prediction and to compare with the predicted output

# In[72]:


x_train


# In[73]:


x_test


# ###### all columns except the Age column is at a std scale. Therefore we have to use the StandardScaler from thr prepocessing module of sklearn library to bring all coumns toa std scale

# In[74]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[75]:


# giving the training values to the scaler so that the function can understand the range of values
sc.fit(x_train)
# creating the scaled up train value and test value : X_tr and X_ts
X_tr =sc.transform(x_train)
X_ts =sc.transform(x_test)


# In[76]:


X_tr


# In[ ]:





# In[82]:


# Model 1: Logistic Regression 
from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()

# Fit our model to the training data
model1.fit(X_tr,y_train)


# In[ ]:




