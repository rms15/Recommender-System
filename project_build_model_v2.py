# Code for data pre-processing - Training and Test Dataset

# Feature creation for training dataset.
# Compute Pearson's correlation coefficient between users to depict user-user similarity.


"""
Created on Thu Sep 24 13:55:29 2015

@author: rshaik2
"""
import os
import pandas as pd
from multiprocessing import Pool
import numpy as np
import scipy as scp
import sys
from scipy import sparse
from scipy.stats.stats import pearsonr 
from decimal import getcontext, Decimal
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
## To save the models to disk
from sklearn.externals import joblib
## To get time in different format
import time
## For saving the model
import pickle
## To calculate area under the ROC curve as a measure to tune the model. 
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pylab import *

getcontext().prec = 3

os.getcwd()
if sys.platform in ('linux', 'linux2'):
    os.chdir('/home/auto/rshaik2/code')
else:
    os.chdir('/Users/riazm_shaik/Rice/03 Fall 2015/STAT 640 - Data Mining/kaggle')
 #   os.chdir('C:\\Users\\rshaik2\\Documents\\Rice\stat640\kaggle')
    
ratings_ext_input_sim2 = pd.read_csv("ratings_ext_input_sim2_prof.csv")

X_features = ['top1_rating','top2_rating','top3_rating','top1_sim','top2_sim','top3_sim',
              'gender_num','avg_profile_ratings_users','avg_profile_ratings_all_users',
              'avg_user_rating_k_sim_profiles']

X_features = ['top1_rating',
              'gender_num','avg_profile_ratings_users','avg_profile_ratings_all_users',
              'avg_user_rating_k_sim_profiles']
# ridgecv : 0.653
              
X_features = ['gender_num','avg_profile_ratings_users','avg_profile_ratings_all_users',
              'avg_user_rating_k_sim_profiles']
        
ratings_ext_input_sim2['avg_profile_ratings_users'][1:10]      
plt.plot([1, 2, 3, 4], [1, 4, 9, 16],'ro')
       
plt.plot(ratings_ext_input_sim2['avg_profile_ratings_users'][1:1000000],ratings_ext_input_sim2['rating'][1:1000000],'ro')
# good. for rating 10, few outliers present.

plt.plot(ratings_ext_input_sim2['avg_profile_ratings_all_users'][1:100000],ratings_ext_input_sim2['rating'][1:100000],'ro')
# not good correlation. feature used in benchmark model ?

plt.plot(ratings_ext_input_sim2['avg_user_rating_k_sim_profiles'][1:1000000],ratings_ext_input_sim2['rating'][1:1000000],'ro')
# good.

plt.plot(ratings_ext_input_sim2['top1_rating'][1:1000],ratings_ext_input_sim2['rating'][1:1000],'ro')
# not good

plt.plot(ratings_ext_input_sim2['top2_rating'][1:1000],ratings_ext_input_sim2['rating'][1:1000],'ro')
# not good

plt.plot(ratings_ext_input_sim2['top3_rating'][1:1000],ratings_ext_input_sim2['rating'][1:1000],'ro')



d = 4
f = 2
s = 1024
param_n_jobs_value = -1
model = RandomForestClassifier(n_estimators=50, criterion="entropy", \
                                               max_features=f, max_depth=d, \
                                               min_samples_split=s,n_jobs=param_n_jobs_value,
                                               verbose=1,\
                                               oob_score=False)
sample = ratings_ext_input_sim[X_features].head(10)
weighted_rating = (ratings_ext_input_sim['top1_rating']*ratings_ext_input_sim['top1_sim'] + 
ratings_ext_input_sim['top2_rating']*ratings_ext_input_sim['top2_sim'] + 
ratings_ext_input_sim['top3_rating']*ratings_ext_input_sim['top3_sim'] )/(ratings_ext_input_sim['top1_sim'] + ratings_ext_input_sim['top2_sim'] + 
ratings_ext_input_sim['top3_sim'] )
weighted_rating.fillna(5,inplace=True)



#Predicting purely based on average or max values
mean_rating = ratings_ext_input_sim[['top1_rating','top2_rating','top3_rating']].apply(np.mean,1)
error = (ratings_ext_input_sim['top3_rating'] - ratings_ext_input_sim['rating'])
error = (weighted_rating - ratings_ext_input_sim['rating'])
np.mean(error*error) # very high 4.04

## how about products of top similarities with their ratings
ratings_ext_input_sim2['top1_sim_times_rating'] = ratings_ext_input_sim2['top1_sim'] * ratings_ext_input_sim2['top1_rating']
ratings_ext_input_sim2['top2_sim_times_rating'] = ratings_ext_input_sim2['top2_sim'] * ratings_ext_input_sim2['top2_rating']
ratings_ext_input_sim2['top3_sim_times_rating'] = ratings_ext_input_sim2['top3_sim'] * ratings_ext_input_sim2['top3_rating']


# Random Forests. Can use oob_score for model tuning
#joblib.dump(model_rf, 'model_rf.pkl') 
rating_predicted = model_rf.predict(ratings_ext_input_sim[X_features])
error = (ramodel_rf = model.fit(ratings_ext_input_sim[X_features], ratings_ext_input_sim['rating'])
ting_predicted - ratings_ext_input_sim['rating'])
np.mean(error*error) # very high 4.04

# Linear regression
wt_rating = pd.DataFrame(data=weighted_rating)
model = linear_model.LinearRegression()
model_lr = model.fit(wt_rating, ratings_ext_input_sim['rating'])
rating_predicted = model_lr.predict(wt_rating)
error = (rating_predicted - ratings_ext_input_sim['rating'])
np.mean(error*error) # very high 4.04

# Linear regression 
model = linear_model.LinearRegression()
model_lr = model.fit(ratings_ext_input_sim2[X_features],  ratings_ext_input_sim2['rating'])
rating_predicted = model_lr.predict(ratings_ext_input_sim2[X_features])
error = (rating_predicted - ratings_ext_input_sim2['rating'])
np.mean(error*error) # very high 4.77


# LDA
model = LDA()
model_lda = model.fit(ratings_ext_input_sim2[X_features], ratings_ext_input_sim2['rating'])
rating_predicted = model_lda.predict(ratings_ext_input_sim2[X_features])
error = (rating_predicted - ratings_ext_input_sim2['rating'])
np.mean(error*error) # very high 7.13

# out of 3.2 million, for 1.03 million ratings are not matching. Model is not fitting one-third of the data
# QDA
from sklearn.qda import QDA
model = QDA()
model_qda = model.fit(ratings_ext_input_sim2[X_features], ratings_ext_input_sim2['rating'])
rating_predicted = model_qda.predict(ratings_ext_input_sim2[X_features])
error = (rating_predicted - ratings_ext_input_sim2['rating'])
np.mean(error*error) # very high 5.03

# Ridge
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model_ridge = model.fit(ratings_ext_input_sim2[X_features], ratings_ext_input_sim2['rating'])
rating_predicted = model_ridge.predict(ratings_ext_input_sim2[X_features])
error = (rating_predicted - ratings_ext_input_sim2['rating'])
np.mean(error*error) #  high 4.04


X_new_features = ['UserID', 'ProfileID',
'avg_profile_ratings_all_users_x', 'avg_profile_rating_k_sim_users','wt_avg_profile_rating_k_sim_users',
                'gender_num_x',
'avg_user_rating_k_sim_profiles_x','wt_avg_user_rating_k_sim_profiles',
'top1_rating','top2_rating','top3_rating','top1_sim','top2_sim','top3_sim']

nan_idx = np.where(np.isnan(ratings_ALL_ftrs['wt_avg_user_rating_k_sim_profiles']))
ratings_ALL_ftrs.loc[nan_idx[0],'wt_avg_user_rating_k_sim_profiles'] = ratings_ALL_ftrs['avg_profile_ratings_all_users_x'][nan_idx[0]]

nan_idx = np.where(np.isnan(ratings_ALL_ftrs['avg_user_rating_k_sim_profiles_x']))
fill_values = ratings_ALL_ftrs.loc[nan_idx[0]]['avg_profile_ratings_all_users_x']
### tip when using .loc use both row and column index in [i,j] form
ratings_ALL_ftrs.loc[nan_idx[0],['avg_user_rating_k_sim_profiles_x'] ]= fill_values

# Ridge on collab.filtering all features
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model_ridge = model.fit(ratings_ALL_ftrs[X_new_features], ratings_ALL_ftrs['rating'])
rating_predicted = model_ridge.predict(ratings_ALL_ftrs[X_new_features])
error = (rating_predicted - ratings_ALL_ftrs['rating'])
np.mean(error*error) #  high 4.04
#-- visualize regularization paths for Ridge
# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)


#RidgeCV on collab.filtering all features
from sklearn.linear_model import RidgeCV
model = RidgeCV(cv=10)
model_ridge = model.fit(ratings_ALL_ftrs[X_new_features], ratings_ALL_ftrs['rating'])
rating_predicted = model_ridge.predict(ratings_ALL_ftrs[X_new_features])
error = (rating_predicted - ratings_ALL_ftrs['rating'])
sqrt(np.mean(error*error)) #  4.77 (0.633 good?)
score=model_ridge.score(ratings_ALL_ftrs[X_new_features], ratings_ALL_ftrs['rating'])
model_ridge.coef_
###############################################################################
# Compute paths

n_alphas = 20
alphas = np.logspace(-10, 1, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(ratings_ext_input_sim2[X_features], ratings_ext_input_sim2['rating'])
    coefs.append(clf.coef_)

###############################################################################
# Display results

ax = plt.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

#RidgeCV
from sklearn.linear_model import RidgeCV
model = RidgeCV(cv=20)
model_ridge = model.fit(ratings_ext_input_sim2[X_features], ratings_ext_input_sim2['rating'])
rating_predicted = model_ridge.predict(ratings_ext_input_sim2[X_features])
error = (rating_predicted - ratings_ext_input_sim2['rating'])
np.mean(error*error) #  4.77 (0.633 good?)
score=model_ridge.score(ratings_ext_input_sim2[X_features], ratings_ext_input_sim2['rating'])
model_ridge.coef_

# Elastic Net
from sklearn.linear_model import ElasticNetCV
enet = ElasticNetCV(l1_ratio=0.5,cv = 10) # 1 for LASSO
model_enet = enet.fit(ratings_ext_input_sim2[X_features], ratings_ext_input_sim2['rating'])
rating_predicted = model_enet.predict(ratings_ext_input_sim2[X_features])
error = (rating_predicted - ratings_ext_input_sim2['rating'])
np.mean(error*error)  # 4.168
# alpha = 1, l1_ration = 0: very high 4.67
# alpha = 0.1, l1_ration = 0: very high 4.57
# alpha = 0.5, l1_ration = 0: very high 4.64
# alpha = 0.7, l1_ration = 0: very high 4.65
from sklearn.linear_model import lasso_path, enet_path
model_enet.mse_path_
plt.figure(1)
ax = plt.gca()
ax.set_color_cycle(2 * ['b', 'r', 'g', 'c', 'k'])
#l1 = plt.plot(-np.log10(alphas_lasso), coefs_lasso.T)
l1 = plt.plot(-np.log10(model_enet.alphas_), model_enet.coef_, linestyle='--')

# Elastic Net CV
from sklearn.linear_model import ElasticNetCV
enet = ElasticNetCV(l1_ratio=0.7,cv=10)
model_enet = enet.fit(ratings_ext_input_sim2[X_features], ratings_ext_input_sim2['rating'])
rating_predicted = model_enet.predict(ratings_ext_input_sim2[X_features])
error = (rating_predicted - ratings_ext_input_sim2['rating'])
np.mean(error*error)  # 4.168
#1.179944 - l1_ration = 1
# 1.18 - l1:0.5 or 0.7



# Stochastic Gradient Descent
SGD = linear_model.SGDClassifier(loss='hinge',l1_ratio=0.5)
model_SGD = SGD.fit(ratings_ext_input_sim2[X_features], ratings_ext_input_sim2['rating'])
rating_predicted = model_SGD.predict(ratings_ext_input_sim2[X_features])
error = (rating_predicted - ratings_ext_input_sim2['rating'])
np.mean(error*error) 
# best modified_huber: 7.59 very high

# Logistic Regression
logreg = linear_model.LogisticRegression(C=1e5)
model_logreg = logreg.fit(ratings_ext_input_sim[X_features], ratings_ext_input_sim['rating'])
rating_predicted = model_logreg.predict(ratings_ext_input_sim[X_features])
error = (rating_predicted - ratings_ext_input_sim['rating'])
np.mean(error*error) # 7.757 Too lot of time for model building    