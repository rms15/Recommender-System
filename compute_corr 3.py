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
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.getcwd()
if sys.platform in ('linux', 'linux2'):
    os.chdir('/home/auto/rshaik2/code')
#    os.chdir('/home/rms15/kaggle')
else:
        os.chdir('C:\\Users\\rshaik2\\Documents\\Rice\stat640\kaggle')
    #    os.chdir('/Users/riazm_shaik/Rice/03 Fall 2015/STAT 640 - Data Mining/kaggle')

 #   os.chdir('C:\\Users\\rshaik2\\Documents\\Rice\stat640\kaggle')
#######################Read input files#################################################
ratings = pd.read_csv("ratings.csv")
idmap =  pd.read_csv("IDMap.csv",index_col=False)
gender =  pd.read_csv("gender.csv")
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
#### Useful insights - by user######################
#rmat_idx =  rmat>0
#temp1 = np.sum(rmat_idx,axis=1)
#users_who_dated_most = np.argsort(temp1)[::-1]
#nbr_of_dates = temp1[users_who_dated_most]
#plot(nbr_of_dates)
#plot(temp1)
## User 4461 has rated 4247 profiles -- max
#users_who_dated_most[0:20]
##array([4461, 7500, 9242, 9956, 5361, 9662, 2824, 1053, 1730, 8210, 9815,
##       9089, 9151, 5823,   31, 5016, 2570, 1489, 8876,  332], dtype=int64)
#nbr_of_dates[0:20]
##array([4871, 4595, 4517, 3968, 3909, 3886, 3524, 3322, 3272, 3216, 3122,
##       3116, 3060, 3052, 3020, 2957, 2937, 2903, 2863, 2780])
### by profile
#temp2 = np.sum(rmat_idx,axis=0)
#profiles_who_were_dated_most = np.argsort(temp2)[::-1]
#nbr_of_times_dated = temp2[profiles_who_were_dated_most]
#plot(nbr_of_times_dated)
#plot(temp2)
##top 20 most dated profiles
#profiles_who_were_dated_most[0:20]
##array([1167,  294, 5465, 5015, 8405, 9747, 5595, 3134, 2529, 7281, 5830,
##       3173, 2031, 2530, 5409, 5794,  615, 7303, 4322, 8773], dtype=int64)
#nbr_of_times_dated[0:20]
##array([4871, 4595, 4517, 3968, 3909, 3886, 3524, 3322, 3272, 3216, 3122,
##       3116, 3060, 3052, 3020, 2957, 2937, 2903, 2863, 2780])


#for i in range(10):
#    for j in range(10):
#       list(set(rmat_idx[i,]).intersection(rmat_idx[j,]))

####This code needs to be active for computing the correlation matrix.
min_comm_profile_threshold = 50;
corr_read = np.zeros((rmat.shape[0],rmat.shape[1],))
corr_read[:] = np.NAN

mat_user_comm_profiles = np.zeros((rmat.shape[0],rmat.shape[1],))
mat_user_comm_profiles[:] = np.NAN


for user1 in range(rmat.shape[0]):
#for user1 in range(5):
    print("Processing for user1:",user1)
    for user2 in range(user1,rmat.shape[0]):
#    for user2 in range(user1,5):        
        user1_profiles = np.where(~np.isnan(rmat[user1,:]))
        user2_profiles = np.where(~np.isnan(rmat[user2,:]))
        user1_profiles_list = list(user1_profiles[0])
        user2_profiles_list = list(user2_profiles[0])
        common_profiles = list(set(user1_profiles_list).intersection(user2_profiles_list))
#        corr_mat2[user1,user2] = round(np.corrcoef(rmat[user1,common_profiles], rmat[user2,common_profiles],rowvar=1)[0][1],2)
        mat_user_comm_profiles[user1,user2] = len(common_profiles)
#        multiplier = min(len(common_profiles) /(min_comm_profile_threshold),1)
        if mat_user_comm_profiles[user1,user2] != 0:
             x = pearsonr(rmat[user1,common_profiles], rmat[user2,common_profiles])[0]
             corr_read[user1,user2] = str.format("{0:.3f}", x)
        else:
            corr_read[user1,user2] = 0

for j in range(corr_read.shape[1]):
    for i in range(j+1,corr_read.shape[0]):
        corr_read[i,j] = corr_read[j,i]
        mat_user_comm_profiles[i,j] = mat_user_comm_profiles[j,i]
        


corr_read - np.transpose(corr_read)

np.savetxt("corr_user_adjpear.csv",corr_read,delimiter=",",fmt=("%1.3f"))
print("Saved the user-user correlation matrix to disk")

np.savetxt("mat_user_comm_profiles.csv",mat_user_comm_profiles,delimiter=",",fmt=("%1.3f"))
print("Saved the number of common profiles info to disk")



#corr_read = np.asmatrix(pd.read_csv("corr_mat_new2.csv",header=None))
#print("Loaded the user-user correlation matrix from disk")

#mat_user_comm_profiles = np.asmatrix(pd.read_csv("mat_user_comm_profiles.csv",header=None))
#print("Loaded the he number of common profiles info from disk")

#def myfunc(a, b):
#    if a>1:
#        return 1
#    else:
#        return a
#
#vfunc = np.vectorize(myfunc)      
#
#corr_read_adj = np.multiply(corr_read, mat_user_comm_profiles/min_comm_profile_threshold)
#corr_read_adj = vfunc(corr_read_adj,1)      
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
min_comm_user_threshold = 50
corr_prof = np.zeros((rmat.shape[0],rmat.shape[1],))
corr_prof[:] = np.NAN

mat_profile_comm_users = np.zeros((rmat.shape[0],rmat.shape[1],))
mat_profile_comm_users[:] = np.NAN

for profile1 in range(rmat.shape[1]):
#for user1 in range(5):
    print("Profile:",profile1)
    for profile2 in range(profile1,rmat.shape[1]):
#    for user2 in range(user1,5):        
        profile1_users_list = list(np.where(~np.isnan(rmat[:,profile1]))[0])
        profile2_users_list = list(np.where(~np.isnan(rmat[:,profile2]))[0])
        common_users = list(set(profile1_users_list).intersection(profile2_users_list))
#        corr_mat2[user1,user2] = round(np.corrcoef(rmat[user1,common_profiles], rmat[user2,common_profiles],rowvar=1)[0][1],2)
        mat_profile_comm_users[profile1,profile2] = len(common_users)
        if  mat_profile_comm_users[profile1,profile2] != 0:
             x = pearsonr(rmat[common_users,profile1], rmat[common_users,profile2])[0]
             corr_prof[profile1,profile2] = str.format("{0:.3f}", x)
        else:
             corr_prof[profile1,profile2] = 0
       

for j in range(corr_prof.shape[1]):
    for i in range(j+1,corr_prof.shape[0]):
        corr_prof[i,j] = corr_prof[j,i]
        mat_profile_comm_users[i,j] = mat_profile_comm_users[j,i]

np.savetxt("corr_profile_adjpear.csv",corr_prof,delimiter=",",fmt=("%1.3f"))
print("Saved the profile-profile correlation matrix to disk")

np.savetxt("mat_profile_comm_users.csv",mat_profile_comm_users,delimiter=",",fmt=("%1.3f"))
print("Saved the number of common users info to disk")
