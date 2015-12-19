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
from sklearn.decomposition import NMF
import scipy.sparse as spr
import nimfa
from scipy.sparse import csr_matrix as csr
import scipy as scp
import sklearn


getcontext().prec = 3
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.getcwd()
if sys.platform in ('linux', 'linux2'):
#    os.chdir('/home/auto/rshaik2/code/')
    os.chdir('/home/rms15/kaggle/dataFiles_ver2')
else:
        os.chdir('/Users/riazm_shaik/Rice/03 Fall 2015/STAT 640 - Data Mining/kaggle/dataFiles_ver2')

 #   os.chdir('C:\\Users\\rshaik2\\Documents\\Rice\stat640\kaggle')
#######################Read input files#################################################
ratings = pd.read_csv("ratings.csv")
ratings = pd.read_csv("ratings_X_user_profile_features.csv")
ratings_topR = pd.read_csv("ratings_ext_input_sim2_prof.csv")

idmap =  pd.read_csv("IDMap.csv",index_col=False)
idmap = pd.read_csv("idmap_X_user_features_test_prof.csv",index_col=False)
idmap_topR = pd.read_csv("idmap_ext_sim2_prof.csv",index_col=False)
gender =  pd.read_csv("gender.csv")
################## Get the top ratings and sim with those groups from previous data file version

ratings_ALL_ftrs = pd.merge(ratings,ratings_topR ,
                                  left_on=['UserID','ProfileID'],right_on=['userid','profileid'],
      how='inner')
      
idmap_ALL_ftrs = pd.merge(idmap,idmap_topR ,
                                  left_on=['UserID','ProfileID'],right_on=['userid','profileid'],
      how='inner')
      
X_new_features = ['UserID', 'ProfileID',
'avg_profile_ratings_all_users_x', 'avg_profile_rating_k_sim_users','wt_avg_profile_rating_k_sim_users',
                'gender_num_x',
'avg_user_rating_k_sim_profiles_x','wt_avg_user_rating_k_sim_profiles',
'top1_rating','top2_rating','top3_rating','top1_sim','top2_sim','top3_sim']

idmap_ALL_ftrs_renamed =  idmap_ALL_ftrs[['UserID', 'ProfileID','KaggleID',
'avg_profile_ratings_all_users', 'avg_profile_rating_k_sim_users','wt_avg_profile_rating_k_sim_users',
                'gender_num',
'avg_user_rating_k_sim_profiles_x','wt_avg_user_rating_k_sim_profiles',
'top1_rating','top2_rating','top3_rating','top1_sim','top2_sim','top3_sim']

idmap_ALL_ftrs_renamed = idmap_ALL_ftrs_renamed.rename(columns={'avg_profile_ratings_all_users': 'avg_profile_ratings_all_users_x', 'gender_num': 'gender_num_x'})


nan_idx = np.where(np.isnan(idmap_ALL_ftrs_renamed['wt_avg_user_rating_k_sim_profiles']))
idmap_ALL_ftrs_renamed.loc[nan_idx[0],'wt_avg_user_rating_k_sim_profiles'] = idmap_ALL_ftrs_renamed['avg_profile_ratings_all_users_x'][nan_idx[0]]

nan_idx = np.where(np.isnan(idmap_ALL_ftrs_renamed['avg_user_rating_k_sim_profiles_x']))
fill_values = idmap_ALL_ftrs_renamed.loc[nan_idx[0]]['avg_profile_ratings_all_users_x']
### tip when using .loc use both row and column index in [i,j] form
idmap_ALL_ftrs_renamed.loc[nan_idx[0],['avg_user_rating_k_sim_profiles_x'] ]= fill_values

 
nan_idx = np.where(np.isnan(idmap_ALL_ftrs_renamed['avg_profile_rating_k_sim_users']))
fill_values = idmap_ALL_ftrs_renamed.loc[nan_idx[0]]['avg_profile_ratings_all_users_x']
idmap_ALL_ftrs_renamed.loc[nan_idx[0],['avg_profile_rating_k_sim_users'] ]= fill_values

 
nan_idx = np.where(np.isnan(idmap_ALL_ftrs_renamed['wt_avg_profile_rating_k_sim_users']))
fill_values = idmap_ALL_ftrs_renamed.loc[nan_idx[0]]['avg_profile_ratings_all_users_x']
idmap_ALL_ftrs_renamed.loc[nan_idx[0],['wt_avg_profile_rating_k_sim_users'] ]= fill_values



###############Build a sparse matrix ####
ratings_sample = ratings.head(10000)
#ratings_spr = csr((ratings_sample.Rating, (ratings_sample.UserID-1, ratings_sample.ProfileID-1)))
ratings_spr = csr((ratings.Rating, (ratings.UserID-1, ratings.ProfileID-1)))
print('Target:\n%s' % ratings_spr.todense())
ratings_dense = ratings_spr.todense()
########## ALS ####


nz_idx = nonzero(ratings_spr)
ratings_dense = ratings_spr.todense(0)
W = np.zeros(shape=(10000,10000))
W[nz_idx] = 1
lambda_ = 0.1
n_factors = 20
m, n = ratings_spr.shape
n_iterations = 5
X = 5 * np.random.rand(m, n_factors) 
Y = 5 * np.random.rand(n_factors, n)

avg_prof = csr.sum(ratings_spr,axis=0)/(csr.getnnz(ratings_spr,axis=0)+1.0)
avg_user = transpose(csr.sum(ratings_spr,axis=1))/((csr.getnnz(ratings_spr,axis=1))+1.0)
avg_mat = csr.sum(ratings_spr)/(csr.getnnz(ratings_spr)*1.0) ## row means

def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y) - avg_prof - avg_user - avg_mat))**2)
    
weighted_errors = []
for ii in range(n_iterations):
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), ratings_dense[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), ratings_dense[:, i])))
    weighted_errors.append(get_error(ratings_dense, X, Y, W))
    print('{}th iteration is completed'.format(ii))
weighted_Q_hat = np.dot(X,Y)

########### SVD ####

#a = np.matrix('0,0,5;0,1,6;0,5,10;1,3,9;2,4,2;3,1,3;4,4,9;5,1,5')
#a_df = pd.DataFrame(a,columns = ['UserID','ProfileID','Rating'])
#a_spr = csr((a_df.Rating,(a_df.UserID,a_df.ProfileID)))
#a_spr.todense()

# Compute average profile ratings
# compute average user ratings
# compute overall mean
# compute low-rank SVD and then compute W and H
# Optimize (actual rating - avg_u - avg_prof - avg_all - W*H for that user-profile)


avg_prof = csr.sum(ratings_spr,axis=0)/(csr.getnnz(ratings_spr,axis=0)+1.0)
avg_user = transpose(csr.sum(ratings_spr,axis=1))/((csr.getnnz(ratings_spr,axis=1))+1.0)
avg_mat = csr.sum(ratings_spr)/(csr.getnnz(ratings_spr)*1.0) ## row means

from sklearn import linear_model
avg_user_array = np.array(avg_user)
avg_prof_array = np.array(avg_prof)


bias_mat  = np.zeros(shape=(len(ratings),3))
bias_mat[:,0] = avg_user_array[0][ratings.UserID-1]
bias_mat[:,1] = avg_prof_array[0][ratings.ProfileID-1]
bias_mat[:,2] = avg_mat
#fun = ratings_spr - avg_user - avg_prof - ratings_app


LinReg = linear_model.LinearRegression()
model_lr = LinReg.fit(bias_mat,  ratings.Rating)
model_lr.coef_
#  array([ 0.50667756,  0.94059636, -0.04218541])
rating_predicted = model_lr.predict(bias_mat)
rating_unbiased = ratings.Rating - rating_predicted

ratings_unb_spr = csr((rating_unbiased, (ratings.UserID-1, ratings.ProfileID-1)))
#ratings_spr  = ratings_spr *1.0
ratings_svd = spr.linalg.svds(ratings_unb_spr,k=60,which='LM')
sing_values = ratings_svd[1]
var_prop = sing_values/sum(sing_values)
U = matrix(ratings_svd[0])
D = diag(ratings_svd[1])
V = matrix(transpose(ratings_svd[2]))

D_sort = sorted(diag(D),reverse=True)
D_sort_sq_sum = sum(diag(D) * diag(D))
D_sort_sq_isum = 0
pct_var = zeros(len(diag(D)))
pct_cum_var = zeros(len(diag(D)))

for i in range(len(diag(D))):
    D_sort_sq_isum = (D_sort[i]*D_sort[i]) + D_sort_sq_isum
    pct_cum_var[i] = D_sort_sq_isum/D_sort_sq_sum
    pct_var[i] = (D_sort[i]*D_sort[i])/D_sort_sq_sum
    
# k = 60 is good
    
#### user * weights for profile features
W = U * transpose(sqrt(D))
#### profile_features * profiles ( how good are the profiles w.r.t the profile features)
H = (sqrt(D)) * transpose(V)
ratings_app = W*H
frob_loss = np.linalg.norm(ratings_unb_spr[csr.nonzero(ratings_unb_spr)] - ratings_app[csr.nonzero(ratings_unb_spr)], ord='fro')
print(frob_loss)

ratings_est_svd = ratings_app[csr.nonzero(ratings_unb_spr)]  +  np.matrix(bias_mat) * transpose(np.matrix(model_lr.coef_))
# estimate for training dataset


#scp.optimize.minimize(fun, method=)

### Can get the weights from W and use them as values for new features (K)
### Or try to reduce the Frob loss with regulariation
############ NMF Tuning k no of components ###
error_train = np.zeros(30);    
row_idx = range(10000)
nz_idx = nonzero(ratings_spr)
error_CV_avg = np.zeros(30)
from sklearn.decomposition import NMF
import random
for nbr_comp in range(1,30):
    # regularization; 0 means no reg
# 0 for L2, 1 for L2
    model = NMF(n_components=nbr_comp, init='random', random_state=0,l1_ratio=0.5,alpha=0.5)
    model.fit(ratings_spr) 
    H = model.components_;
    W = model.fit_transform(ratings_spr);
    ratings_app = matrix(W) * matrix(H);
    error_train[nbr_comp-1] = np.linalg.norm(ratings_spr - ratings_app, ord ='fro')
    print('error_train:', error_train[nbr_comp-1])
    error_CV = np.zeros(5);
    for r in range(5):
        row_idx_train, row_idx_test= train_test_split( row_idx, test_size=0.33, random_state=42)        
        nz_idx_train = tuple([np.array(nz_idx[0][row_idx_train]),np.array(nz_idx[1][row_idx_train])])
        nz_idx_test = tuple([np.array(nz_idx[0][row_idx_test]),np.array(nz_idx[1][row_idx_test])])    
        error_CV[r] = np.linalg.norm(ratings_spr[nz_idx_test] - ratings_app[nz_idx_test], ord ='fro')
        print('error_CV:', error_CV[r])
    error_CV_avg[nbr_comp-1] = np.mean(error_CV)
    print('error_CV_avg:', error_CV_avg[nbr_comp-1])

############ l1_ratio = 0   ###

model = NMF(n_components=30, init='random', random_state=0,l1_ratio=0)
model.fit(ratings_spr) 
H = model.components_;
W = model.fit_transform(ratings_spr);
ratings_app = matrix(W) * matrix(H);
error_k30 = np.linalg.norm(ratings_spr - ratings_app, ord ='fro')

############ NMF Tuning lambda  ###

error_train = np.zeros(10);    
row_idx = range(10000)
nz_idx = nonzero(ratings_spr)
error_CV_avg = np.zeros(10)
i = 0;
for alpha_iter in np.arange(1,50,5):
    # regularization; 0 means no reg
# 0 for L2, 1 for L2
    print('alpha_iter:', alpha_iter)
    model = NMF(n_components=30, init='random', random_state=0,l1_ratio=0,alpha=alpha_iter)
    row_idx_train, row_idx_test= train_test_split( row_idx, test_size=0.33, random_state=42)        
    nz_idx_train = tuple([np.array(nz_idx[0][row_idx_train]),np.array(nz_idx[1][row_idx_train])])
    nz_idx_test = tuple([np.array(nz_idx[0][row_idx_test]),np.array(nz_idx[1][row_idx_test])]) 
    model.fit(ratings_spr[row_idx_train]) 
    H = model.components_;
    W = model.fit_transform(ratings_spr[row_idx_train]);
    ratings_app = matrix(W) * matrix(H);
    error_train[i] = np.linalg.norm(ratings_spr[row_idx_train] - ratings_app, ord ='fro')
    print('error_train:', error_train[i])
    error_CV = np.zeros(5);
    for r in range(5):
        model.fit(ratings_spr[nz_idx_test]) 
        H = model.components_;
        W = model.fit_transform(ratings_spr[nz_idx_test]);
        ratings_app = matrix(W) * matrix(H);   
        error_CV[r] = np.linalg.norm(ratings_spr[nz_idx_test] - ratings_app, ord ='fro')
        print('error_CV:', error_CV[r])
    error_CV_avg[i] = np.mean(error_CV)
    print('error_CV_avg:', error_CV_avg[i]) 
    i = i+1

    # for validationg, select any random portion of non zero ratings and find the Frob norm
    # Find the l1_ration parameter and n_components

################ NMF ########

#snmf = nimfa.models.Nmf(ratings_spr, max_iter=200, rank=2, update='euclidean', objective='fro')
# split matrix into train and test


#nz_idx[row_idx_train]
#nz_idx = list(nonzero(ratings_spr)[0], nonzero(ratings_spr)[1])
## 
#nz_idx = {'UserID': ratings.UserID-1, 'ProfileID': ratings.ProfileID-1}
#nz_idx_df = pd.DataFrame(nz_idx, columns = ['UserID','ProfileID'])
#nz_idx_mat = matrix(nz_idx_df)


#nz_idx = (ratings.UserID-1, ratings.ProfileID-1)

#nz_train_idx = tuple(nz_idx_mat[row_idx_train,:])
#nz_test_idx = nz_idx_mat[row_idx_test,:]


#idx_train, idx_test= train_test_split( idx[0], test_size=0.33, random_state=42)

error_train = np.zeros(30);
error_test = np.zeros(30);


for nbr_comp in range(1,30):
    model = NMF(n_components=nbr_comp, init='random', random_state=0)
    model.fit(ratings_spr[row_idx_train]) 
    H = model.components_;
    W = model.fit_transform(ratings_spr[row_idx_train]);
    ratings_train_reconst = matrix(W) * matrix(H);
    temp1 = ratings_spr[nz_idx_train]
    temp2 = ratings_train_reconst
    error_train[nbr_comp-1] = np.linalg.norm(temp1 - temp2, ord ='fro')
    
    model.fit(ratings_spr[row_idx_test]) 
    H = model.components_;
    W = model.fit_transform(ratings_spr[row_idx_test]);
    ratings_test_reconst = matrix(W) * matrix(H);
    temp1 = ratings_spr[nz_idx_test]
    temp2 = ratings_test_reconst
    error_test[nbr_comp-1] = np.linalg.norm(temp1 - temp2, ord ='fro')
    
    
## nbr_comp = 14 has min Frob norm error
    
############ sample NMF ###
    
X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
nbr_comp = 2
model = NMF(n_components=nbr_comp, init='random', random_state=0)
model.fit(X) 
H = model.components_;
W = model.fit_transform(X);
X_reconst = matrix(W) * matrix(H);
print(np.linalg.norm(X - X_reconst, ord ='fro'))

#######################Build a dense matrix from the ratings file#######################
rmat = np.empty((10000,10000,))
rmat[:] = np.NAN
rmat.shape
prev_row_UserID = 0
for row in ratings.iterrows():
#    if row[1]['UserID'] != prev_row_UserID :
#        prev_row_UserID = row[1]['UserID']
#        print("user index",row[1]['UserID']-1,"profile index:",row[1]['ProfileID']-1,"value:", row[1]['Rating'])
    rmat[row[1]['UserID']-1,row[1]['ProfileID']-1] = row[1]['Rating']

#######################Verify #######################
#userid_tocheck = 10000
#
#for userid_tocheck in xrange(1000,1050):
#    id_profiles = ratings[(ratings['UserID'] == userid_tocheck ) ]['ProfileID'] -1 
#    id_ratings = np.array(ratings[(ratings['UserID'] == userid_tocheck  ) ]['Rating'])
#    print(sum(rmat[userid_tocheck -1 ,id_profiles] - id_ratings))


####This code needs to be active for computing the correlation matrix.
#corr_read = np.zeros((rmat.shape[0],rmat.shape[1],))
#corr_read[:] = np.NAN
#
#for user1 in range(rmat.shape[0]):
##for user1 in range(5):
#    print("Processing for user1:",user1)
#    for user2 in range(user1,rmat.shape[0]):
##    for user2 in range(user1,5):        
#        user1_profiles = np.where(~np.isnan(rmat[user1,:]))
#        user2_profiles = np.where(~np.isnan(rmat[user2,:]))
#        user1_profiles_list = list(user1_profiles[0])
#        user2_profiles_list = list(user2_profiles[0])
#        common_profiles = list(set(user1_profiles_list).intersection(user2_profiles_list))
##        corr_mat2[user1,user2] = round(np.corrcoef(rmat[user1,common_profiles], rmat[user2,common_profiles],rowvar=1)[0][1],2)
#        if len(common_profiles) != 0:
#             x = pearsonr(rmat[user1,common_profiles], rmat[user2,common_profiles])[0]
#             corr_read[user1,user2] = str.format("{0:.3f}", x)
#        else:
#            corr_read[user1,user2] = 0
#
#for j in range(corr_read.shape[1]):
#    for i in range(j+1,corr_read.shape[0]):
#        corr_read[i,j] = corr_read[j,i]

#corr_read - np.transpose(corr_read)

#np.savetxt("corr_mat_new2.csv",corr_read,delimiter=",",fmt=("%1.3f"))
#print("Saved the new correlation matrix to disk")

corr_read = np.asmatrix(pd.read_csv("corr_mat_new2.csv",header=None))
print("Loaded the user-user correlation matrix from disk")

#######################Verify #######################
#userid1_tocheck = 1100
#userid2_tocheck = 3300
#
#id1_profiles = ratings[(ratings['UserID'] == userid1_tocheck ) ]['ProfileID'] -1
#id2_profiles = ratings[(ratings['UserID'] == userid2_tocheck ) ]['ProfileID'] -1
#id_common_profiles = list(set(id1_profiles).intersection(id2_profiles))
#id_common_profiles2 = pd.Series(id_common_profiles)
##id1_ratings = np.array(ratings[(ratings['UserID'] == userid1_tocheck )  & (ratings['ProfileID'] == id_common_profiles2.any) ]['Rating'])
##id2_ratings = np.array(ratings[(ratings['UserID'] == userid2_tocheck )  & (ratings['ProfileID'] == id_common_profiles2.any) ]['Rating'])
#
#corr_check = pearsonr(rmat[userid1_tocheck -1 ,id_common_profiles2],rmat[userid2_tocheck -1 ,id_common_profiles2])[0]
#corr_check - corr_read[userid1_tocheck-1, userid2_tocheck-1]
###########


#x = rmat[0,:]
#
#y = rmat[1,:]
#xy = ((x - np.nanmean(x))*(y - np.nanmean(y)))
#n = np.nansum((x - np.nanmean(x))*(y - np.nanmean(y)))
#
#d1 = math.sqrt(np.nansum((x - np.nanmean(x))*(x - np.nanmean(x))))
#d2 = math.sqrt(np.nansum((y - np.nanmean(y))*(y - np.nanmean(y))))
#
#corr = n/(d1*d2)
#
#
#rmat_n = np.sum((rmat[user1,common_profiles] - np.mean(rmat[user1,common_profiles]))*(rmat[user2,common_profiles] - np.mean(rmat[user2,common_profiles])))
#rmat_d1 = math.sqrt(np.sum((rmat[user1,common_profiles] - np.mean(rmat[user1,common_profiles]))*(rmat[user1,common_profiles] - np.mean(rmat[user1,common_profiles]))))
#rmat_d2 = math.sqrt(np.sum((rmat[user2,common_profiles] - np.mean(rmat[user2,common_profiles]))*(rmat[user2,common_profiles] - np.mean(rmat[user2,common_profiles]))))
#rmat_n/(rmat_d1*rmat_d2)



####This code needs to be active for computing the correlation matrix for profiles.
## Run parallel threads 10.
#corr_prof = np.zeros((rmat.shape[0],rmat.shape[1],))
#corr_prof[:] = np.NAN
#
#
#for profile1 in range(rmat.shape[1]):
##for user1 in range(5):
#    print("Profile1:",profile1)
#    for profile2 in range(profile1,rmat.shape[1]):
##    for user2 in range(user1,5):        
#        profile1_users_list = list(np.where(~np.isnan(rmat[:,profile1]))[0])
#        profile2_users_list = list(np.where(~np.isnan(rmat[:,profile2]))[0])
#        common_users = list(set(profile1_users_list).intersection(profile2_users_list))
##        corr_mat2[user1,user2] = round(np.corrcoef(rmat[user1,common_profiles], rmat[user2,common_profiles],rowvar=1)[0][1],2)
#        if len(common_users) != 0:
#             x = pearsonr(rmat[common_users,profile1], rmat[common_users,profile2])[0]
#             corr_prof[profile1,profile2] = str.format("{0:.3f}", x)
#        else:
#             corr_prof[profile1,profile2] = 0
#
#for j in range(corr_prof.shape[1]):
#    for i in range(j+1,corr_prof.shape[0]):
#        corr_prof[i,j] = corr_prof[j,i]


#np.savetxt("corr_prof_new2.csv",corr_prof,delimiter=",",fmt=("%1.3f"))
#print("Saved the new correlation matrix to disk")

corr_prof = np.asmatrix(pd.read_csv("corr_prof_new2.csv",header=None))
print("Loaded the profile-profile correlation matrix from disk")

#######################Verify #######################
#profile1_tocheck = 3500
#profile2_tocheck = 2300
#
#profile1_users = ratings[(ratings['ProfileID'] == profile1_tocheck ) ]['UserID'] -1
#profile2_users = ratings[(ratings['ProfileID'] == profile2_tocheck ) ]['UserID'] -1
#profile_common_users = list(set(profile1_users).intersection(profile2_users))
#profile_common_users2 = pd.Series(profile_common_users)
#
#corr_check = pearsonr(rmat[profile_common_users2,profile1_tocheck-1 ],rmat[profile_common_users2 ,profile2_tocheck-1])[0]
#corr_check - corr_prof[profile1_tocheck-1, profile2_tocheck-1]
###########
#sys.exit()

##############Create features for similarity of active user with user groups grouped based #####
##############on 1 to 10 ratings for that profile########
#ratings_ext = np.empty((len(ratings),13,))
#ratings_ext[:] = np.NAN
### rmat and corr_read have users from 0 to 9999
### ratings file has users from 1 to 10000
#for row in ratings.iterrows():
#    print("row number:", row[0],"userid:",row[1]['UserID'])
#    users_who_rated_this_profile = (np.where(~np.isnan(rmat[:,row[1]['ProfileID']-1]))[0].tolist())
## Remove the user for which we are analysing from other users list
#    users_who_rated_this_profile.remove(row[1]['UserID']-1)
#    users_rating_for_this_profile = rmat[users_who_rated_this_profile,row[1]['ProfileID']-1]
#    user_user_similarity = np.squeeze(np.array(corr_read[row[1]['UserID']-1,tuple(users_who_rated_this_profile)]))
#    data = { 'users_rating_for_this_profile':users_rating_for_this_profile, 'user_user_similarity':user_user_similarity}
#    df_users_ratings_profile=pd.DataFrame(data)
#    ratings_summary = df_users_ratings_profile.groupby(by='users_rating_for_this_profile',as_index=False).mean()
#    ratings_ext[row[0],0] = row[1]['UserID']
#    ratings_ext[row[0],1] = row[1]['ProfileID']   
#    ratings_ext[row[0],12] = row[1]['Rating'] 
#    k = 2
#    for rating in range(1,11):
#        if rating not in list(ratings_summary['users_rating_for_this_profile']):
#            ratings_ext[row[0],k] = np.NAN
#        else:
#            x = ratings_summary.loc[ratings_summary['users_rating_for_this_profile'] ==rating,'user_user_similarity'].values[0]
#            ratings_ext[row[0],k] = str.format("{0:.3f}",x)
#        k = k+1
#
#np.savetxt("ratings_ext.csv",ratings_ext,delimiter=",",fmt=("%1.3f"))
#
#############################Obtain top 3 similarity groups and their ratings####
#ratings_ext_input = pd.read_csv("ratings_ext.csv",names=[
#                            'userid','profileid',
#                            'sim_rating1',
#                            'sim_rating2','sim_rating3','sim_rating4',
#                            'sim_rating5','sim_rating6','sim_rating7','sim_rating8',
#                            'sim_rating9','sim_rating10','rating'])
#                                
#X_features = [w for w in list(ratings_ext_input)]
#X_features.remove('rating')
#X_features.remove('userid')
#X_features.remove('profileid')
##np.where(np.isnan(ratings_ext_input[1:4]))
#
#ratings_ext_input.fillna(0,inplace=True)
#
##temp = ratings_ext_input.head()
#
#top1_sim = np.empty(len(ratings_ext_input))
#top1_sim[:] = np.NaN
#
#top2_sim = np.empty(len(ratings_ext_input))
#top2_sim[:] = np.NaN
#
#top3_sim = np.empty(len(ratings_ext_input))
#top3_sim[:] = np.NaN
#
#
#top1_rating = np.empty(len(ratings_ext_input))
#top1_rating[:] = np.NaN
#
#top2_rating = np.empty(len(ratings_ext_input))
#top2_rating[:] = np.NaN
#
#top3_rating = np.empty(len(ratings_ext_input))
#top3_rating[:] = np.NaN
#
#for row in ratings_ext_input.iterrows():
#    print("row:",row[0])
#    sim_val_row = list(row[1][X_features])
#    sim_val_row2 = list(sim_val_row)
#    top1_sim[row[0]] = max(sim_val_row)
#    top1_rating[row[0]] = sim_val_row2.index(top1_sim[row[0]])+1
#
#    sim_val_row.remove(top1_sim[row[0]])
#    top2_sim[row[0]] = max(sim_val_row)
#    top2_rating[row[0]] = sim_val_row2.index(top2_sim[row[0]])+1
#
#    sim_val_row.remove(top2_sim[row[0]])
#    top3_sim[row[0]] = max(sim_val_row)
#    top3_rating[row[0]] = sim_val_row2.index(top3_sim[row[0]])+1
#
#sim_cols = pd.DataFrame({'top1_sim':top1_sim,'top2_sim':top2_sim,
#                         'top3_sim':top3_sim,'top1_rating':top1_rating,
#                         'top2_rating':top2_rating,'top3_rating':top3_rating})
#ratings_ext_input_sim = pd.concat([ratings_ext_input,sim_cols],axis=1)
#
#ratings_ext_input_sim.to_csv("ratings_ext_input_sim.csv")
#ratings_ext_input_sim = pd.read_csv("ratings_ext_input_sim.csv")

################ Choose k-number of nearest neighbors for the active user Ua#########
################# Use the average of the ratings given by them ###############
#Options: Pick top n similar users; Pick all similar users above a similarity threshold (current implementation); Pick top n similar users;

threshold_sim = 0.8

topn_sim = 20
topn_sim_lwlimit = 0
sample = ratings.head(3)
avg_profile_rating_k_sim_users = np.empty(len(ratings))
avg_profile_rating_topn_sim_users = np.empty(len(ratings))

for row in ratings.iterrows():
   print("row:",row[0])
   active_user = row[1]['UserID']-1
# get similar users to the active user
   sim_array_temp = np.array(corr_read[active_user,:])[0]
   sim_array_temp[np.where(np.isnan(sim_array_temp))] = -2
# get indices of top k similarities   - userids
#   sim_array_sorted = np.sort(sim_array_temp)[::-1][0:topk]
#   top_k_similar_users = np.argsort(sim_array_temp)[::-1][0:topk]
#   profile_ratings_k_sim_users = rmat[top_k_similar_users,row[1]['profileid']-1]
   sim_array_above_trshld = sim_array_temp[sim_array_temp >= threshold_sim]
   users_above_sim_trshld = (np.where([sim_array_temp >= threshold_sim])[1])
   profile_ratings_sim_trshld_users = rmat[users_above_sim_trshld,row[1]['ProfileID']-1]

# Pick top n similar users 
   topn_sim_lwlimit = 0
   topn_sim_users =   np.argsort(sim_array_temp)[::-1][:topn_sim]
   profile_ratings_topn_sim_users = rmat[topn_sim_users,row[1]['ProfileID']-1]
   while (  sum(~np.isnan(profile_ratings_topn_sim_users)) == 0):
       topn_sim_lwlimit = topn_sim_lwlimit + topn_sim
       topn_sim_uplimit = topn_sim_lwlimit + topn_sim
     #  print(topn_sim_uplimit)
       topn_sim_users =   np.argsort(sim_array_temp)[::-1][topn_sim_lwlimit:topn_sim_uplimit]
       profile_ratings_topn_sim_users = rmat[topn_sim_users,row[1]['ProfileID']-1]   
   
#   profile_ratings_sim_trshld_users[np.isnan(profile_ratings_sim_trshld_users)] = 0
#   np.dot(profile_ratings_sim_trshld_users,sim_array_above_trshld)/sum(sim_array_above_trshld[~np.isnan(sim_array_above_trshld)])
   avg_profile_rating_k_sim_users[row[0]] = np.nanmean(profile_ratings_sim_trshld_users)
   avg_profile_rating_topn_sim_users[row[0]] = np.nanmean(profile_ratings_topn_sim_users)


# From rmat matrix, get ratings of these k users for the profile in question.
# Compute the average or weighted average and collect this as a series   

####### check ########
#from collections import Counter 
#trshld_users_check = profile_ratings_sim_trshld_users[np.where(~np.isnan(profile_ratings_sim_trshld_users))]
#Counter(trshld_users_check)
#sim_array_temp.sort(reverse=True)
#
#np.argsort(sim_array_temp)[::-1][:10]
######################################################



################ For profiles: Choose k-number of nearest neighbors for the active user Ua#########
################# Use the average of the ratings given by them ###############

threshold_sim = 0.8
topn_sim = 20
topn_sim_lwlimit = 0
sample = ratings_ext_input_sim2.head()
avg_user_rating_k_sim_profiles = np.empty(len(ratings))
for row in ratings.iterrows():
   print("row:",row[0])
   active_profile = row[1]['ProfileID']-1
# get similar users to the active user
   sim_array_temp = np.array(corr_prof[active_profile,:])[0]
   sim_array_temp[np.where(np.isnan(sim_array_temp))] = 0
# get indices of top k similarities   - userids
#   sim_array_sorted = np.sort(sim_array_temp)[::-1][0:topk]
#   top_k_similar_users = np.argsort(sim_array_temp)[::-1][0:topk]
#   profile_ratings_k_sim_users = rmat[top_k_similar_users,row[1]['profileid']-1]
   sim_array_above_trshld = sim_array_temp[sim_array_temp >= threshold_sim]
   profiles_above_sim_trshld = (np.where([sim_array_temp >= threshold_sim])[1])
   user_ratings_sim_trshld_profiles = rmat[row[1]['UserID']-1,profiles_above_sim_trshld]


# Pick top n similar profiles 
   topn_sim_lwlimit = 0  
   topn_sim_profiles =   np.argsort(sim_array_temp)[::-1][:topn_sim]
   user_ratings_topn_sim_profiles = rmat[row[1]['UserID']-1, topn_sim_profiles]
   while (  sum(~np.isnan(user_ratings_topn_sim_profiles)) == 0):
       topn_sim_lwlimit = topn_sim_lwlimit + topn_sim
       topn_sim_uplimit = topn_sim_lwlimit + topn_sim
       print(topn_sim_uplimit)
       topn_sim_profiles =   np.argsort(sim_array_temp)[::-1][topn_sim_lwlimit:topn_sim_uplimit]
       user_ratings_topn_sim_profiles = rmat[row[1]['UserID']-1, topn_sim_profiles]   


#   profile_ratings_sim_trshld_users[np.isnan(profile_ratings_sim_trshld_users)] = 0
#   np.dot(profile_ratings_sim_trshld_users,sim_array_above_trshld)/sum(sim_array_above_trshld[~np.isnan(sim_array_above_trshld)])
   avg_user_rating_k_sim_profiles[row[0]] = np.nanmean(user_ratings_sim_trshld_profiles)
   avg_user_rating_topn_sim_profiles[row[0]] = np.nanmean(user_ratings_topn_sim_profiles)

#####
###
avg_profile_rating = np.nanmean(rmat,axis=0)
data = { 'ProfileID': range(1,10001),'avg_profile_ratings_all_users':avg_profile_rating}   
avg_profile_ratings_all_users_df=pd.DataFrame(data)
ratings = pd.merge(ratings,avg_profile_ratings_all_users_df ,
                                  left_on=['ProfileID'],right_on=['ProfileID'],
      how='inner')
      
#del ratings_ext_input_sim2['UserID']
data = { 'avg_profile_ratings_users':avg_profile_rating_k_sim_users}   
new_cols=pd.DataFrame(data=data)
ratings = pd.concat([ratings,new_cols],axis=1)



# not required since we have a sliding window for top n until at least one user rating is found
#import copy
#avg_user_rating_k_sim_profiles_cp = copy.copy(avg_user_rating_k_sim_profiles)
#nan_indx = np.array(np.where(np.isnan(avg_user_rating_k_sim_profiles_cp))).tolist()
#avg_user_rating_k_sim_profiles_cp[nan_indx] = avg_profile_ratings_all_users_df.loc[ratings_ext_input_sim2.loc[nan_indx[0]]['profileid']-1,'avg_profile_ratings_all_users']

#data = { 'avg_user_rating_k_sim_profiles':avg_user_rating_k_sim_profiles_cp}   
data = { 'avg_user_rating_k_sim_profiles':avg_user_rating_k_sim_profiles}   
new_cols=pd.DataFrame(data=data)

ratings = pd.concat([ratings,new_cols],axis=1)

#### gender and average profile rating
ratings = pd.merge(ratings, gender,left_on=['UserID'],right_on=['UserID'],
      how='inner')
gender_dict = dict([('M', 1), ('F', 2), ('U', 3)])
gender_data = pd.DataFrame(data={'gender_num':[gender_dict[w] for w in ratings['Gender']]})
ratings_ext_input_sim2 = pd.concat([ratings,gender_data],axis=1)

X_features = ['top1_rating','top2_rating','top3_rating','top1_sim','top2_sim','top3_sim',
              'gender_num','avg_profile_ratings_users','avg_profile_ratings_all_users']
   
ratings_ext_input_sim2.to_csv("ratings_ext_input_sim2_prof.csv")
###====================================================================##
### PRE-PROCESSING FOR TEST DATASET
###====================================================================##
#
###############Create features for similarity of active user with user groups grouped based #####
###############on 1 to 10 ratings for that profile########
#
#idmap_ext = np.empty((len(idmap),13,))
#idmap_ext[:] = np.NAN
### rmat and corr_read have users from 0 to 9999
### ratings file has users from 1 to 10000
#for row in idmap.iterrows():
#    print("row number:", row[0],"userid:",row[1]['UserID'])
#    users_who_rated_this_profile = (np.where(~np.isnan(rmat[:,row[1]['ProfileID']-1]))[0].tolist())
## Remove the user for which we are analysing from other users list
##    users_who_rated_this_profile.remove(row[1]['UserID']-1)
#    users_rating_for_this_profile = rmat[users_who_rated_this_profile,row[1]['ProfileID']-1]
#    user_user_similarity = np.squeeze(np.array(corr_read[row[1]['UserID']-1,tuple(users_who_rated_this_profile)]))
#    data = { 'users_rating_for_this_profile':users_rating_for_this_profile, 'user_user_similarity':user_user_similarity}
#    df_users_ratings_profile=pd.DataFrame(data)
#    test_ratings_summary = df_users_ratings_profile.groupby(by='users_rating_for_this_profile',as_index=False).mean()
##    row_new = [row[1]['UserID'],row[1]['ProfileID'],row[1]['Rating']]
##    row_new.extend(list(ratings_summary['user_user_similarity']))
#    idmap_ext[row[0],0] = row[1]['UserID']
#    idmap_ext[row[0],1] = row[1]['ProfileID']   
#    idmap_ext[row[0],12] = row[1]['KaggleID'] 
#    k = 2
#    for rating in range(1,11):
#        if rating not in list(test_ratings_summary['users_rating_for_this_profile']):
#            idmap_ext[row[0],k] = np.NAN
#        else:
#            x = test_ratings_summary.loc[test_ratings_summary['users_rating_for_this_profile'] ==rating,'user_user_similarity'].values[0]
#            idmap_ext[row[0],k] = str.format("{0:.3f}",x)
#        k = k+1
#
#
#np.savetxt("idmap_ext.csv",idmap_ext,delimiter=",",fmt=("%1.3f"))
#
#idmap_ext=pd.read_csv("idmap_ext.csv",names=['userid','profileid',
#                                             'sim_rating1','sim_rating2','sim_rating3','sim_rating4',
#                                             'sim_rating5','sim_rating6','sim_rating7','sim_rating8',
#                                             'sim_rating9','sim_rating10','kaggleid'])
#
#############################Obtain top 3 similarity groups and their ratings####
#
#
#idmap_ext.fillna(0,inplace=True)
#
##temp = idmap_ext.head()
#
#top1_sim = np.empty(len(idmap_ext))
#top1_sim[:] = np.NaN
#
#top2_sim = np.empty(len(idmap_ext))
#top2_sim[:] = np.NaN
#
#top3_sim = np.empty(len(idmap_ext))
#top3_sim[:] = np.NaN
#
#
#top1_rating = np.empty(len(idmap_ext))
#top1_rating[:] = np.NaN
#
#top2_rating = np.empty(len(idmap_ext))
#top2_rating[:] = np.NaN
#
#top3_rating = np.empty(len(idmap_ext))
#top3_rating[:] = np.NaN
#
#for row in idmap_ext.iterrows():
#    print("row:",row[0])
#    sim_val_row = list(row[1][X_features_test])
#    sim_val_row2 = list(sim_val_row)
#    top1_sim[row[0]] = max(sim_val_row)
#    top1_rating[row[0]] = sim_val_row2.index(top1_sim[row[0]])+1
#
#    sim_val_row.remove(top1_sim[row[0]])
#    top2_sim[row[0]] = max(sim_val_row)
#    top2_rating[row[0]] = sim_val_row2.index(top2_sim[row[0]])+1
#
#    sim_val_row.remove(top2_sim[row[0]])
#    top3_sim[row[0]] = max(sim_val_row)
#    top3_rating[row[0]] = sim_val_row2.index(top3_sim[row[0]])+1
#
#sim_cols = pd.DataFrame({'top1_sim':top1_sim,'top2_sim':top2_sim,
#                         'top3_sim':top3_sim,'top1_rating':top1_rating,
#                         'top2_rating':top2_rating,'top3_rating':top3_rating})
#idmap_ext_sim = pd.concat([idmap_ext,sim_cols],axis=1)
#
# 
#X_features_test = ['top1_rating','top2_rating','top3_rating','top1_sim','top2_sim','top3_sim']
#    
#idmap_ext_sim.to_csv("idmap_ext_sim.csv")
#idmap_ext_sim = pd.read_csv("idmap_ext_sim.csv")    
#
################# Choose k-number of nearest neighbors for the active user Ua#########
################## Use the average of the ratings given by them ###############
#
#threshold_sim = 0.7
#sample = idmap_ext_sim.head()
#avg_profile_rating_k_sim_users_test = np.empty(len(idmap_ext_sim))
#for row in idmap_ext_sim.iterrows():
#   print("row:",row[0])
#   active_user = row[1]['userid']-1
## get similar users to the active user
#   sim_array_temp = np.array(corr_read[active_user,:])[0]
#   sim_array_temp[np.where(np.isnan(sim_array_temp))] = 0
## get indices of top k similarities   - userids
##   sim_array_sorted = np.sort(sim_array_temp)[::-1][0:topk]
##   top_k_similar_users = np.argsort(sim_array_temp)[::-1][0:topk]
##   profile_ratings_k_sim_users = rmat[top_k_similar_users,row[1]['profileid']-1]
#   sim_array_above_trshld = sim_array_temp[sim_array_temp >= threshold_sim]
#   users_above_sim_trshld = (np.where([sim_array_temp >= threshold_sim])[1])
#   profile_ratings_sim_trshld_users = rmat[users_above_sim_trshld,row[1]['profileid']-1]
##   profile_ratings_sim_trshld_users[np.isnan(profile_ratings_sim_trshld_users)] = 0
##   np.dot(profile_ratings_sim_trshld_users,sim_array_above_trshld)/sum(sim_array_above_trshld[~np.isnan(sim_array_above_trshld)])
#   avg_profile_rating_k_sim_users_test[row[0]] = np.nanmean(profile_ratings_sim_trshld_users)
#    
#       
## From rmat matrix, get ratings of these k users for the profile in question.
## Compute the average or weighted average and collect this as a series   
#sum(np.isnan(avg_profile_rating_k_sim_users_test)   )
## 19769 (reduced to 2776 for theshold = 0.7)
#
#import copy
#avg_profile_rating_k_sim_users_testcp = copy.copy(avg_profile_rating_k_sim_users_test)
#nan_indx = np.array(np.where(np.isnan(avg_profile_rating_k_sim_users_testcp))).tolist()
#avg_profile_rating_k_sim_users_testcp[nan_indx] = avg_profile_ratings_all_users_df.loc[idmap_ext_sim.loc[nan_indx[0]]['profileid']-1,'avg_profile_ratings_all_users']
#
################# For profiles: Choose k-number of nearest neighbors for the active user Ua#########
################## Use the average of the ratings given by them ###############
#
#threshold_sim = 0.7
#sample = idmap_ext_sim2.head()
#avg_user_rating_k_sim_profiles_test = np.empty(len(idmap_ext_sim2))
#for row in idmap_ext_sim2.iterrows():
#   print("row:",row[0])
#   active_profile = row[1]['profileid']-1
## get similar users to the active user
#   sim_array_temp = np.array(corr_prof[active_profile,:])[0]
#   sim_array_temp[np.where(np.isnan(sim_array_temp))] = 0
## get indices of top k similarities   - userids
##   sim_array_sorted = np.sort(sim_array_temp)[::-1][0:topk]
##   top_k_similar_users = np.argsort(sim_array_temp)[::-1][0:topk]
##   profile_ratings_k_sim_users = rmat[top_k_similar_users,row[1]['profileid']-1]
#   sim_array_above_trshld = sim_array_temp[sim_array_temp >= threshold_sim]
#   profiles_above_sim_trshld = (np.where([sim_array_temp >= threshold_sim])[1])
#   user_ratings_sim_trshld_profiles = rmat[row[1]['userid']-1,profiles_above_sim_trshld]
##   profile_ratings_sim_trshld_users[np.isnan(profile_ratings_sim_trshld_users)] = 0
##   np.dot(profile_ratings_sim_trshld_users,sim_array_above_trshld)/sum(sim_array_above_trshld[~np.isnan(sim_array_above_trshld)])
#   avg_user_rating_k_sim_profiles_test[row[0]] = np.nanmean(user_ratings_sim_trshld_profiles)
#
#
#import copy
#avg_user_rating_k_sim_profiles_testcp = copy.copy(avg_user_rating_k_sim_profiles_test)
#nan_indx = np.array(np.where(np.isnan(avg_user_rating_k_sim_profiles_testcp))).tolist()
#avg_user_rating_k_sim_profiles_testcp[nan_indx] = avg_profile_ratings_all_users_df.loc[idmap_ext_sim2.loc[nan_indx[0]]['profileid']-1,'avg_profile_ratings_all_users']
#
#
#
#data = { 'avg_user_rating_k_sim_profiles':avg_user_rating_k_sim_profiles_testcp}   
#new_cols=pd.DataFrame(data=data)
#
#idmap_ext_sim2 = pd.concat([idmap_ext_sim2,new_cols],axis=1)
############################
#
#idmap_ext_sim2 = pd.merge(idmap_ext_sim, gender,left_on=['userid'],right_on=['UserID'],
#      how='inner')
#
#del idmap_ext_sim2['UserID']
#
#data1 = { 'avg_profile_ratings_users':avg_profile_rating_k_sim_users_testcp}   
#new_cols=pd.DataFrame(data=data1)
##idmap_ext_sim2['avg_profile_ratings_users']
#idmap_ext_sim2 = pd.concat([idmap_ext_sim2,new_cols],axis=1)
#sum(np.isnan(idmap_ext_sim2['avg_profile_ratings_users'])   )
#
#avg_profile_rating = np.nanmean(rmat,axis=0)
#data2 = { 'profileid': range(1,10001),'avg_profile_ratings_all_users':avg_profile_rating}   
#avg_profile_ratings_all_users_df=pd.DataFrame(data=data2)
#idmap_ext_sim2 = pd.merge(idmap_ext_sim2,avg_profile_ratings_all_users_df ,
#                                  left_on=['profileid'],right_on=['profileid'],
#      how='inner')
#
#gender_dict = dict([('M', 1), ('F', 2), ('U', 3)])
#
#gender_data = pd.DataFrame(data={'gender_num':[gender_dict[w] for w in
#     idmap_ext_sim2['Gender']]})
#
#idmap_ext_sim2= pd.concat([idmap_ext_sim2,gender_data],axis=1)
#X_features = ['top1_rating','top2_rating','top3_rating','top1_sim','top2_sim','top3_sim',
#              'gender_num','avg_profile_ratings_users','avg_profile_ratings_all_users']
#    
#idmap_ext_sim2.to_csv("idmap_ext_sim2_prof.csv")
##idmap_ext_sim2 = pd.read_csv("idmap_ext_sim2.csv")
