#!/usr/bin/env python
# coding: utf-8

# # Introduction to the DataSet
# 
# The Dataset contain 3 classes of 50 instances each. Where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are not linearly separable from each other.
# 
# Attribute information:
#    1. Sepal length in cm
#    2. Sepal width in cm
#    3. Petal length in cm
#    4. Petal width in cm
#    5. classes; 
#         a) Iris Setosa
#         b) Iris Versicolour
#         c) Iris Virginica

# ## Import Modules
# 

# In[7]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# ## Loading the Dataset

# In[8]:


df = pd.read_csv('Iris.csv')
df.head()


# In[9]:


df = df.drop(columns = ['Id'])
df.head()
        


# In[10]:


df.describe()


# In[12]:


df.info()


# In[18]:


df['Species'].value_counts()


# In[19]:


df.isnull().sum() # data has no null so no need to pre process the data its ready already


# ## EDA

# In[27]:


df['SepalLengthCm'].hist(bins = 10)


# In[28]:


df['SepalWidthCm'].hist(bins = 10)


# In[29]:


df['PetalLengthCm'].hist(bins = 10)


# In[30]:


df['PetalWidthCm'].hist(bins = 10)


# In[31]:


# Scatter plot
colours = ['Red','Green','Blue']
Species = ['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[36]:


for i in range (3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c =colours[i], label = Species[i])
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend()
    


# In[37]:


for i in range (3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c =colours[i], label = Species[i])
plt.xlabel('PetalLength')
plt.ylabel('PetalWidth')
plt.legend()


# In[38]:


for i in range (3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'],c =colours[i], label = Species[i])
plt.xlabel('SepalLength')
plt.ylabel('PetalLength')
plt.legend()
    


# In[39]:


for i in range (3):
    x = df[df['Species'] == Species[i]]
    plt.scatter(x['SepalWidthCm'],x['PetalWidthCm'],c =colours[i], label = Species[i])
plt.xlabel('SepalWidth')
plt.ylabel('PetalWidth')
plt.legend()
    


# ## Correlation Matrix 
# 
# A correlation matrix is a table representing a correlation between two variables. The value of a cell ranges between -1 to 1. 
# 1. strong correlation = 1
# 2. moderate correlation = 0 
# 3. weak correlation = -1

# In[40]:


df.corr()


# In[47]:


corr = df.corr()
fig , ax = plt.subplots(figsize=(5,5))
sns.heatmap(corr , annot = True , ax=ax) #correlation of 0.96 between petal width and petal length


# ## Label encoding
# 
# We use this to convert the labels present in the data to a numeric form for machine to understand it.

# In[50]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[53]:


df['Species'] = le.fit_transform(df['Species'])


# # Classification Model Training

# In[59]:


# Train - 65
# Test - 35
from sklearn.model_selection import train_test_split
x = df.drop(columns = 'Species')
y = df['Species']

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.35)


# In[62]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[64]:


model.fit(x_train,y_train)


# In[70]:


print('Accuracy: ', model.score(x_test,y_test)* 100)


# In[74]:


from sklearn.neighbors import KNeighborsClassifier
model1 = KNeighborsClassifier()


# In[75]:


model1.fit(x_train,y_train)


# In[76]:


print('Accuracy: ', model1.score(x_test,y_test)* 100)


# In[78]:


from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier()


# In[79]:


model2.fit(x_train,y_train)


# In[80]:


print('Accuracy: ', model2.score(x_test,y_test)* 100)


# # Now Predict the class

# In[88]:


y.shape


# In[81]:


from sklearn import metrics


# In[89]:


expected = y
predicted = model2.predict(x)


# Now summarize the fit of the model

# In[90]:


print(metrics.classification_report(expected,predicted))


# In[91]:


print(metrics.confusion_matrix(expected,predicted))

