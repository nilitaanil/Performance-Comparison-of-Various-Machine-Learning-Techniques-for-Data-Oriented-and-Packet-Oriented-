#!/usr/bin/env python
# coding: utf-8

# In[128]:


from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics 
import skfuzzy as fuzz
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# In[129]:


import numpy as np, pandas as pd, os
import matplotlib
import itertools
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import skfuzzy as fuzz


# In[130]:


data = pd.read_csv('Final_data_MP.csv')


# In[131]:


data.head(20)


# In[132]:


data=data.dropna()


# In[133]:


data['Label'].unique()


# In[134]:


data.groupby('Label')['Label'].count()


# In[135]:


x=data[['temperature','light','voltage','humidity']]


# In[136]:


x.head()


# In[137]:


y=data['Label']


# In[138]:


y


# In[139]:


cate=set(data['Label'])
id=0
maps={}
for proto in cate:
    if proto == 'Normal':
        maps[proto]=1
    else:
        maps[proto]=0
    
maps


# In[140]:


y=data['Label'].map(maps)


# In[141]:


scaler = StandardScaler()
X_std = scaler.fit_transform(x)


# In[142]:


X_std


# In[143]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_std, y, test_size=0.30, random_state=42)


# In[144]:


lsa = TruncatedSVD(2, algorithm = 'arpack')
dtm_lsa = lsa.fit_transform(x_train)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
a= pd.DataFrame(dtm_lsa, columns = ["component_1","component_2"])
a['targets']=y_train
fig1, axes1 = plt.subplots(1, 1, figsize=(8, 8))
alldata = np.vstack((a['component_1'], a['component_2']))
fpcs = []
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen'] 
ax = axes1
ncenters = 2
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # # Plot assigned clusters, for each data point in training set
cluster_membership = np.argmax(u, axis=0)
    # print(cluster_membership)
for j in range(ncenters):
     ax.plot(a['component_1'][cluster_membership == j], a['component_2'][cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
for pt in cntr:
    ax.plot(pt[0], pt[1], 'rs')
    
ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
ax.axis('off')
fig1.tight_layout()
fig1.savefig('dataset__fuzzy_clusters_train.png')


# In[145]:


accuracy = sklearn.metrics.accuracy_score(y_train, cluster_membership)
print("Accuracy = ", accuracy * 100)


# In[148]:


lsa = TruncatedSVD(2, algorithm = 'arpack')
dtm_lsa = lsa.fit_transform(x_test)
dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
a= pd.DataFrame(dtm_lsa, columns = ["component_1","component_2"])
a['targets']=y_test
fig1, axes1 = plt.subplots(1, 1, figsize=(8, 8))
alldata = np.vstack((a['component_1'], a['component_2']))
fpcs = []
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen'] 
ax = axes1
ncenters = 2
# cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans.predict(alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(alldata, cntr, 2, error=0.005, maxiter=1000)

    # # Plot assigned clusters, for each data point in training set
cluster_membership = np.argmax(u, axis=0)
    # print(cluster_membership)
for j in range(ncenters):
     ax.plot(a['component_1'][cluster_membership == j],
             a['component_2'][cluster_membership == j], '.', color=colors[j])

        
    # Mark the center of each fuzzy cluster
for pt in cntr:
    ax.plot(pt[0], pt[1], 'rs')
ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
ax.axis('off')


# In[147]:


accuracy = sklearn.metrics.accuracy_score(y_test, cluster_membership)
print("Accuracy = ", accuracy * 100)


# In[ ]:


# scaler = StandardScaler()
# X_std = scaler.fit_transform(x)
# sm = SMOTE(random_state=34)
# X_std, y_smote = sm.fit_resample(X_std, y)

