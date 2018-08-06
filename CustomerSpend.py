
# coding: utf-8

# ## Lous' Modeling Problem
# 
# 
# #### Assignment
# The company is interested in doing a marketing campaign for a product they wish to sell to existing customers. They have asked you to give them some insights into what **characteristics** of the customers are important to determine whether they will buy this product. 
# 
# 
# #### Data
# The data in data.csv has 100,000 observations on customers. The columns are R and V1 – V30.
# -	V1-V30: customer characteristics in February 2018
# -	R: money spent on product if customer bought the product in April 2018 and 0 otherwise. 
# 
# 
# #### Approaches
# 
# ###### I. Econometrics Approach
# 
# In estimating consumer spend, a traditional econometrics approach would utilize a two-step regression approach where:
# 1. The first stage will be a probit model for a binary spend vs. no spend behavior will be estimated and 
# 2. The second step will be linear regression model for a customer spend given that they spent.
# 
# There are other models that econometricians may consider. For instance, we can consider a Tobit model, a model that I've considered for my dissertation:
# 
# > Abdul-Rahman, M. (2008). The demand for physical activity: an application of Grossman's health demand model to the elderly population. (Electronic Thesis or Dissertation). Retrieved from https://etd.ohiolink.edu/pg_10?0::NO:10:P10_ACCESSION_NUM:osu1199127215 
# 
# However, ideally, these approach require that econometricians to have elaborate background on research issues and available variables. So, we will approach the posed research question from a Machine Learning approach.
# 
# ###### II. Machine Learning Approach
# 
# We will be focussing on the posed research question, which is 
# > "what **characteristics** of the customers are important to determine whether they will buy this product"?
# 
# The posed research question is not concerned about how much a buyer spend.  Instead, we are concerned about consumers buy or not. 
# 
# 
# 

# In[18]:

# Base Packages

import sys
import pickle
import matplotlib.pyplot

import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression


# In[19]:

### Data Reading
spend_data = pd.read_csv('data.csv')

print "Original Dataset Dimension:",spend_data.shape
spend_data.head()


# In[22]:

##### Data Cleaning

### Converting consumer spend into buy (1) vs. not buy (0)
spend_data['R'] = (spend_data['R']>0).astype(int)

### We assume that the data are cleaned. 


# In[23]:

##### Univariate Statistics

print "\nUNIVARIATE STATISTICS"
spend_data.describe().transpose()


# In[17]:

# Crosstab breakdown for spend vs. no spend 
# is consistent with 0.05535 or 5.535%
r_count = pd.crosstab(index = spend_data["R"],columns="count") 
r_count


# **Missing Data**
# 
# There are various ways to deal with missing data. For this analysis, I removed observations with missing data. A quick look into univariate statistics, the statistics did not seem to change much. 

# In[37]:

df1 = spend_data.dropna()
# df1.describe().transpose()


# **Data Splitting**
# 
# There are various ways to deal with missing data. For this analysis, I removed observations with missing data. A quick look into univariate statistics, the statistics did not seem to change much. 

# In[44]:

# X Features
features_list = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
 'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
 'V21','V22','V23','V24','V25','V26','V27','V28','V29','V30']

# All rows and the feature_list' columns
X = df1.loc[:, features_list]
X.shape


# In[46]:

# Y Features; response vector
y = df1.R
y.shape


# In[47]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                stratify=y, 
                                                test_size=0.25)


# **Ideal Number of Features**
# 
# We utilized SelectKBest and GridSearchCV to arrive at an ideal number of selected features for this Machine Learning analysis. Based on this analysis, the ideal number of features is 15. If we follow this recommendation strictly, we should go with features with top 15 SelectKBest scores.

# In[50]:

from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

kbest = SelectKBest(f_classif)
pipeline = Pipeline([('kbest', kbest), ('lr', LogisticRegression())])
grid_search = GridSearchCV(pipeline, {'kbest__k': [11,12,13,14,15,16,17,18,19,20]})
#grid_search.fit(features, labels)
grid_search.fit(X, y)
print "Number of Ideal Features:", grid_search.best_params_


# In[52]:

# Univariate Feature Selection
import sklearn.feature_selection

knum=15

k_best = sklearn.feature_selection.SelectKBest(k=knum)
k_best.fit(X, y)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
k_best_features = dict(sorted_pairs[:knum])
print knum,"best features: {0}\n".format(k_best_features.keys())
k_best_features

import operator
sorted(k_best_features.items(), key=operator.itemgetter(1), reverse=True)


# The ideal number of predictors is 15 and they are listed above. Using appropiate Machine Learning and Statistical techniques, we can utilize these 15 variables to arrive at the best prediction model. 
# 
# In the next section, we shared two Machine Learning examples with an analysis on how modifying parameters of the same Machine Learning technique (i.e. Support Vector Machine or SVM) arrived at different prediction performance. 
# 
# **Estimating with a Variety of Classifier**

# In[56]:

### Example 1. Naive Bayes

features_list = ['V23','V20','V17','V29','V11','V8','V2','V5','V26','V28','V22','V19','V25','V21','V9']

features_train, features_test, labels_train, labels_test = train_test_split(X, y,
                                                stratify=y, 
                                                test_size=0.25)

clf = GaussianNB()
clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
accuracy = accuracy_score(predict, labels_test)
acc = accuracy_score(labels_test, predict)
prec = precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print "Prediction Totals:", sum(predict) 
print "Acc =",("{0:.3f}".format(acc))
print "Prec =",("{0:.3f}".format(prec))
print "Rec =",("{0:.3f}".format(recall))

print "\nConfusion Matrix [0 v. 1]: \n",confusion_matrix(labels_test, predict)


# In[65]:

### Example 2. Support Vector Machine (SVM)

features_list = ['V23','V20','V17','V29','V11','V8','V2','V5','V26','V28','V22','V19','V25','V21','V9']

# Test Size
features_train, features_test, labels_train, labels_test = train_test_split(X, y,
                                                stratify=y, 
                                                test_size=0.25)
ccc = 10 # C
ggg = 0.001  # gamma_list 
svmpar = 'sigmoid' # svm_param = ['rbf', 'sigmoid']; kernel; rbf the fastest

clf = SVC(kernel=svmpar, C = ccc, gamma=ggg)
clf.fit(features_train, labels_train)
predict = clf.predict(features_test)
acc = accuracy_score(labels_test, predict)
prec = precision_score(labels_test, predict)
recall = recall_score(labels_test, predict)

print ccc, ggg, svmpar,sum(predict),("{0:.3f}".format(acc)), ("{0:.3f}".format(prec)), ("{0:.3f}".format(recall)),"\n"
print "\nConfusion Matrix [0 v. 1]: \n",confusion_matrix(labels_test, predict)


# In[62]:

### SVM with its Various Model Parameters

features_list = ['V23','V20','V17','V29','V11','V8','V2','V5','V26','V28','V22','V19','V25','V21','V9']

C = [0.01, 0.1, 1, 10, 100]
gamma_list = [1, 0.1, 0.01, 0.001, 0.0001]
svm_param = ['rbf', 'sigmoid'] # kernel; rbf the fastest
# n_sample = [0.3, 0.35, 0.4, 0.5] # kernel; rbf the fastest

# for ccc in C
print "C | gamma | svmpar | n_pred | acc | prec | recall "
print "---------------------------------------------------------" 

# a more hands on approach for my understanding 
# especially on which parameters having the largest impact
for ccc in C:
    for ggg in gamma_list:
        for svmpar in svm_param:
            # for nnn in n_sample:
                # Test Size
            features_train, features_test, labels_train, labels_test = train_test_split(X, y,
                                                stratify=y, 
                                                test_size=0.25)

            clf = SVC(kernel=svmpar, C = ccc, gamma=ggg)
            clf.fit(features_train, labels_train)
            predict = clf.predict(features_test)
            acc = accuracy_score(labels_test, predict)
            prec = precision_score(labels_test, predict)
            recall = recall_score(labels_test, predict)

            print ccc, ggg, svmpar,sum(predict),("{0:.3f}".format(acc)),             ("{0:.3f}".format(prec)), ("{0:.3f}".format(recall)),"\n"


# ### Conclusion
# 
# Based on our results, characteristics of the customers that are important to determine whether they will buy this product are: 
# 
# > 'V23','V20','V17','V29','V11','V8','V2','V5','V26','V28','V22','V19','V25','V21','V9'
# 
# We based this on SelectKBest algorithm, which select the top k features that have maximum relvance score with y (purchase). Selection of best variables does not guarantee great prediction. We also showed examples on how different Machine Learning models and different sets of algorithm parameters arrived at different model performce.  
