#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


data = pd.read_csv('Final_data_MP_changed.csv')
data=data.dropna()


# In[3]:


data.head()


# In[4]:


# data=data.sample(n=100000)


# In[5]:


data.groupby('Label').count()


# In[6]:


data.columns


# In[7]:


X=data[['temperature', 'humidity', 'light', 'voltage']]


# In[8]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
# Create a PCA instance: pca
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(X_std)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='blue')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.savefig("PCA.png")
plt.xticks(features)


# In[9]:


from sklearn.decomposition import PCA
pca = PCA()
x_pca = pca.fit_transform(X)
x_pca = pd.DataFrame(x_pca)
x_pca.head()
explained_variance = pca.explained_variance_ratio_
explained_variance


# In[10]:


y=data['Label']


# In[11]:


classes=data['Label'].unique()


# In[12]:


classes


# In[13]:


# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()
# le.fit(classes)
# encoded_y=le.transform(y)
encoded_y=y


# # Up Sampled Data 

# In[14]:


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, encoded_y)


# In[15]:


X.shape,X_res.shape


# In[ ]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_res)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig("Elbow.png")
plt.show()


# In[ ]:


# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42,shuffle=True)


# In[ ]:


# print('Train Dataset Size: ',X_train.shape)
# print('Test Dataset Size: ',X_test.shape)


# In[ ]:


kmeans = kmeans = KMeans(n_clusters=3, max_iter=300, algorithm = 'auto')
kmeans.fit(X_res)


# In[ ]:


labels=kmeans.labels_


# In[ ]:


labels_true=y_res


# In[ ]:


from sklearn import metrics
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X_res, labels))


# In[ ]:


y_km = kmeans.fit_predict(X_res)


# In[ ]:


y_res,y_km


# In[ ]:


# from sklearn.metrics import confusion_matrix 
# from sklearn.metrics import accuracy_score 
# from sklearn.metrics import classification_report 
# actual=y_train

# predicted=y_km
# results = confusion_matrix(actual, predicted) 
# print ('Confusion Matrix :')
# print(results) 
# print ('Accuracy Score :',accuracy_score(actual, predicted) )
# print('Report : ') 
# print (classification_report(actual, predicted) )


# In[ ]:


# N_CLUSTERS=3

# for cluster in range(N_CLUSTERS):
#     print('cluster: ', cluster)
#     print(y_train[np.where(y_km == cluster)])


# In[27]:


# import seaborn as sn

# df_cm = pd.DataFrame(results)
# plt.figure(figsize = (10,7))
# sn.heatmap(df_cm, annot=True,fmt='')


# # Normal Data

# In[28]:


kmeans = kmeans = KMeans(n_clusters=3, max_iter=300, algorithm = 'auto')
kmeans.fit(X)


# In[29]:


labels=kmeans.labels_
labels_true= y


# In[30]:


from sklearn import metrics
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))


# # Down scaled Data 

# In[31]:


from imblearn.under_sampling import RandomUnderSampler


# In[32]:


rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)


# In[33]:


X_res.shape,y_res.shape


# In[34]:


kmeans = kmeans = KMeans(n_clusters=3, max_iter=300, algorithm = 'auto')
kmeans.fit(X_res)


# In[35]:


labels=kmeans.labels_

labels_true=y_res


# In[36]:


from sklearn import metrics
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels,
                                           average_method='arithmetic'))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X_res, labels))

