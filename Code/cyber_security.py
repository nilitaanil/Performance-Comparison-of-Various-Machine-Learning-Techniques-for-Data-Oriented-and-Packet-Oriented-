#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data1 = pd.read_csv('Final_data_MP.csv')


# In[2]:


data1 = data1.dropna()


# In[3]:


labels = data1['Label'].values


# In[4]:


data1 = data1[['temperature','humidity','light','voltage']]


# In[5]:


data1


# In[6]:


feat_w2v = data1.values


# In[7]:


feat_w2v


# In[8]:


feat_w2v.shape


# In[9]:


#NB
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
gnb = GaussianNB()
import numpy as np
from sklearn.metrics import accuracy_score
acc_gnb = []
acc_train_gnb =[]
for train, test in kf.split(feat_w2v, labels):
    X_train, X_test = feat_w2v[train], feat_w2v[test]
    Y_train, Y_test = labels[train], labels[test]
    
    gnb.fit(X_train, Y_train)
    pred = gnb.predict(X_test)
    pred_train = gnb.predict(X_train)
    acc_train_gnb.append(accuracy_score(pred_train, Y_train))
    acc_gnb.append(accuracy_score(pred, Y_test))

print ('\nTrain Fold Accuracies: ', acc_train_gnb)
print ('\nTrain Accuracy NB CF: ', np.mean(acc_train_gnb))
print ('\nTest Fold Accuracies: ', acc_gnb)
print ('\nTest Accuracy NB CF: ', np.mean(acc_gnb))


# In[ ]:


#SVM
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
a = SVC(kernel='rbf', C=4)
import numpy as np
from sklearn.metrics import accuracy_score
acc_svm = []
acc_train_svm =[]
for train, test in kf.split(feat_w2v, labels):
    X_train, X_test = feat_w2v[train], feat_w2v[test]
    Y_train, Y_test = labels[train], labels[test]
    
    a.fit(X_train, Y_train)
    pred_train = a.predict(X_train)
    pred = a.predict(X_test)
    acc_svm.append(accuracy_score(pred, Y_test))
    acc_train_svm.append(accuracy_score(pred_train, Y_train))

print ('\nTrain Fold Accuracies: ', acc_train_svm)
print ('\nTrain Accuracy SVM CF: ', np.mean(acc_train_svm))
print ('\nTest Fold Accuracies: ', acc_svm)
print ('\nTest Accuracy SVM CF: ', np.mean(acc_svm))


# In[9]:


#LR
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
5
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
lr = LogisticRegression()
acc_lr = []
acc_train_lr =[]
for train, test in kf.split(feat_w2v, labels):
    X_train, X_test = feat_w2v[train], feat_w2v[test]
    Y_train, Y_test = labels[train], labels[test]
    
    lr.fit(X_train, Y_train)
    pred = lr.predict(X_test)
    pred_train = lr.predict(X_train)
    acc_lr.append(accuracy_score(pred, Y_test))
    acc_train_lr.append(accuracy_score(pred_train, Y_train))

print ('\nTrain Fold Accuracies: ', acc_train_lr)
print ('\nTrain Accuracy SVM CF: ', np.mean(acc_train_lr))
print ('\nTest Fold Accuracies: ', acc_lr)
print ('\nTest Accuracy SVM CF: ', np.mean(acc_lr))


# In[ ]:


#GBM
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(learning_rate=0.01, max_depth=3)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_gb = []
acc_train_gb =[]
for train, test in kf.split(feat_w2v, labels):
    X_train, X_test = feat_w2v[train], feat_w2v[test]
    Y_train, Y_test = labels[train], labels[test]
    gb.fit(X_train, Y_train)
    pred = gb.predict(X_test)
    pred_train = gb.predict(X_train)
    acc_gb.append(accuracy_score(pred, Y_test))
    acc_train_gb.append(accuracy_score(pred_train, Y_train))
    
    
print ('\nTrain Fold Accuracies: ', acc_train_gb)
print ('\nTrain Accuracy SVM CF: ', np.mean(acc_train_gb))
print ('\nTest Fold Accuracies: ', acc_gb)
print ('\nTest Accuracy SVM CF: ', np.mean(acc_gb))


# In[ ]:


#MLP
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(16, 2), max_iter=1000, learning_rate='adaptive')
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_mlp = []
acc_train_mlp =[]
for train, test in kf.split(feat_w2v, labels):
    X_train, X_test = feat_w2v[train], feat_w2v[test]
    Y_train, Y_test = labels[train], labels[test]
    mlp.fit(X_train, Y_train)
    pred = mlp.predict(X_test)
    pred_train = mlp.predict(X_train)
    acc_train_mlp.append(accuracy_score(pred_train, Y_train))
    acc_mlp.append(accuracy_score(pred, Y_test))
    
    
print ('\nTrain Fold Accuracies: ', acc_train_mlp)
print ('\nTrainAccuracy SVM CF: ', np.mean(acc_train_mlp))
print ('\nTest Fold Accuracies: ', acc_mlp)
print ('\nTest Accuracy SVM CF: ', np.mean(acc_mlp))


# In[ ]:


#KNN
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_knn = []
acc_train_knn =[]
for train, test in kf.split(feat_w2v, labels):
    X_train, X_test = feat_w2v[train], feat_w2v[test]
    Y_train, Y_test = labels[train], labels[test]
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, Y_train)
    pred = knn.predict(X_test)
    pred_train = knn.predict(X_train)
    acc_train_knn.append(accuracy_score(pred_train, Y_train))
    acc_knn.append(accuracy_score(pred, Y_test))

    
print ('\nTrain Fold Accuracies: ', acc_train_knn)
print ('\nTrain Accuracy SVM CF: ', np.mean(acc_train_knn))
print ('\nTest Fold Accuracies: ', acc_knn)
print ('\nTest  Accuracy SVM CF: ', np.mean(acc_knn))


# In[ ]:


#RF
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=360)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc_rf = []
acc_train_rf = []
for train, test in kf.split(feat_w2v, labels):
    X_train, X_test = feat_w2v[train], feat_w2v[test]
    Y_train, Y_test = labels[train], labels[test]
    rf.fit(X_train, Y_train)
    pred = rf.predict(X_test)
    pred_train = rf.predict(X_train)
#     print (classification_report(pred, Y_test))
    acc_rf.append(accuracy_score(pred, Y_test))
    acc_train_rf.append(accuracy_score(pred_train, Y_train))
    #conf_rf.append(classification_report(pred, Y_test))

print ('\nTrain Fold Accuracies: ', acc_train_rf)
print ('\nTrain Accuracy SVM CF: ', np.mean(acc_train_rf))
print ('\nTest Fold Accuracies: ', acc_rf)
print ('\nTest  Accuracy SVM CF: ', np.mean(acc_rf))


# In[ ]:


from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE # doctest: +NORMALIZE_WHITESPACE
X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)


# In[ ]:


y


# In[ ]:


import pandas as pd
data1 = pd.read_csv('csv_result-Training Dataset.csv')
labels = data1['Result'].values
feat_w2v = data1.drop(['id','Result'], axis=1).values


# In[ ]:


print('Original dataset shape %s' % Counter(labels))


# In[ ]:


sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(feat_w2v, labels)
print('Resampled dataset shape %s' % Counter(y_res))


# In[ ]:




