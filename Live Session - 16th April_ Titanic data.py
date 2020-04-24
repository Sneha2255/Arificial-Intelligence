#!/usr/bin/env python
# coding: utf-8

# #### 1. EDA - Exploratory data analysis
# #### 2. Data munging
# #### 3. Feature Engineering
# #### 4. Visulaisation
# 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


# importing dataset
df= pd.read_csv(r"C:\Users\Sneha\Downloads\Titanicdf_train.csv")


# In[6]:


type(df)


# In[10]:


type(df)


# In[11]:


df


# In[13]:


df.info()


# In[14]:


# first 5 observations
df.head()


# In[15]:


# last 5 observations
df.tail()


# In[16]:


# first 10 observations
df.head(10)


# In[17]:


# to know the no of rows and columns
df.shape


# In[18]:


# to print the columns
col=df.columns
col


# In[19]:


rw=df.rows
rw


# In[20]:


# to give the statistical parameters
df.describe()


# In[21]:


#to print the column contents
df.iloc[:,3]


# In[22]:


df.Name


# In[23]:


df.loc[:,"Name"]


# In[24]:


# to print names from 5th to 50th row
df.loc[5:50,"Name"]


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
# this is given to show the output


# In[26]:


df.plot(kind='box')
plt.show()


# ######  the middle green line is the median or the 50th quartile
# ######  the lowest line represents th min
# ######  the line ablove that represents the 25th quartile
# ######  the highest line represents the max
# ######  the line below that represents the 75th quartile
# ######  the length of the rectangle is called the IQR (Inter Quartile range: diff b/w the 25th and the 75th)
#  

# In[27]:


# to show all the statistical values of all
df.describe(include="all")


# In[28]:


# to analyse the dataset based on the gender 
# value_counts(): gives the count of categorical values
df.Sex.value_counts()


# In[29]:


# to analyse data in %
df.Sex.value_counts(normalize= True)


# In[30]:


df.Survived.value_counts(normalize=True)


# In[31]:


df.Pclass.value_counts()


# In[32]:


# to plot a bar graph on the above data
df.Pclass.value_counts().plot(kind="bar")


# In[33]:


# To group according to sex and show the median ages of both
df.groupby(['Sex']).Age.median()


# In[34]:


df.groupby(['Sex']).Age.mean()


# In[35]:


df.groupby(['Pclass']).Fare.mean()


# In[36]:


# coorelating it with value_counts
df.Pclass.value_counts()


# In[37]:


# to show the mean/median of 2 other columns or data with respect to a particular column
# to group Fare and Age according to the Pclass and find their mean
df.groupby(['Pclass'])["Fare","Age"].mean()


# In[38]:


# crosstab(): to analyse data from two different categorical columns
pd.crosstab(df.Sex,df.Pclass)


# In[39]:


pd.crosstab(df.Pclass,df.Sex)


# In[41]:


pd.crosstab(df.Sex,df.Pclass).plot(kind='bar')


# In[42]:


# pivot_table(index="",cou=lumns="",values="",aggfunc="") : is an extension of the crosstab
df.pivot_table(index="Sex",columns="Pclass",values="Age",aggfunc="mean")


# ###### this represents the mean age pmale and female passengers in each Pclass

# In[43]:


df.pivot_table(index="Sex",columns="Pclass",values="Fare",aggfunc="mean")


# ###### this represents the mean fares of male and female passngers in each pclass

# In[44]:


df.groupby(['Sex','Pclass']).Age.mean()


# ##### here, Sex, Pclass and mean Age is shown using grroupby()
# ##### in In[42], Sex, Pclass and mean Age is shown using pivot_table()

# In[45]:


df.groupby(['Sex','Pclass']).Age.mean().unstack()


# In[47]:


# to find the survival pattern based on ages
df.pivot_table(index="Survived",columns="Pclass",values="Age",aggfunc="mean")


# #### Data munging

# In[48]:


df.info()


# ##### here, for Age,cabin, Embarked, some valuues are missing

# In[49]:


#to display the missing values in Embarked
df[df.Embarked.isnull()]


# In[51]:


df.groupby(['Survived','Pclass','Embarked']).Fare.median()


# ##### When the median fares are grouped according to survived and Pclass, it is highly evident that the 2 passengers have embarked from C as the fare is 79.2 which is closest to 80

# In[52]:


df.Embarked.fillna('C',inplace=True)


# In[53]:


df[df.Embarked.isnull()]


# In[54]:


df.info()


# In[55]:


df.loc[[61,829],:]


# In[56]:


204/891


# ##### the actual values are less than 40%, that particular column should be droped

# In[58]:


df.drop('Cabin',inplace=True,axis=1)


# In[59]:


df.info()


# ##### since Cabin has less than 40% actual data, it has been dropped

# In[1]:


df[df.Age.isnull()]


# In[2]:


# importing dataset
df= pd.read_csv(r"C:\Users\Sneha\Downloads\Titanicdf_train.csv")


# In[5]:


import numpy as np
import pandas as pd
import matplotlib as plt


# In[7]:


# importing dataset
df= pd.read_csv(r"C:\Users\Sneha\Downloads\Titanicdf_train.csv")


# In[8]:


df[df.Age.isnull()]


# In[9]:


891-714


# In[16]:


df.groupby(['Survived','Sex','Pclass']).Age.median()


# In[12]:


df.iloc[29:858,:]


# In[ ]:




