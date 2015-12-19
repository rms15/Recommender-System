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
#    os.chdir('/home/auto/rshaik2/code/dataFiles_ver2')
    os.chdir('/home/rms15/kaggle/dataFiles_ver2')
else:
     #   os.chdir('C:\\Users\\rshaik2\\Documents\\Rice\stat640\kaggle\dataFiles_ver2')
        os.chdir('/Users/riazm_shaik/Rice/03 Fall 2015/STAT 640 - Data Mining/kaggle/dataFiles_ver2')

 #   os.chdir('C:\\Users\\rshaik2\\Documents\\Rice\stat640\kaggle')
#######################Read input files#################################################
ratings = pd.read_csv("ratings.csv")
ratings = pd.read_csv("ratings_X_user_profile_features.csv")
#idmap =  pd.read_csv("IDMap.csv",index_col=False)
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


#corr_read = np.asmatrix(pd.read_csv("corr_user_adjpear.csv",header=None))
#print("Loaded the user-user correlation matrix from disk")
#
#mat_user_comm_profiles = np.asmatrix(pd.read_csv("mat_user_comm_profiles.csv",header=None))
#print("Loaded the common profiles from disk")


corr_prof = np.asmatrix(pd.read_csv("corr_profile_adjpear.csv",header=None))
print("Loaded the profile-profile correlation matrix from disk")

mat_profile_comm_users = np.asmatrix(pd.read_csv("mat_profile_comm_users.csv",header=None))
print("Loaded the common users from disk")

min_comm_profile_threshold = 20 # obtained by averaging the mat_user_comm_profiles

######## Multiplier for adjusted correlation
def myfunc(a,b):
    return min(a,b)

vfunc = np.vectorize(myfunc)      

#corr_read_adj = np.multiply(corr_read, mat_user_comm_profiles/min_comm_profile_threshold)
#corr_read_adj_new = vfunc(np.array(corr_read_adj),1.0)     
#corr_read_adj_new = matrix(corr_read_adj_new)

corr_prof_adj = np.multiply(corr_prof, mat_profile_comm_users/min_comm_profile_threshold)
corr_prof_adj_new = vfunc(np.array(corr_prof_adj),1.0)     
corr_prof_adj_new = matrix(corr_prof_adj_new)

########
#threshold_sim = 0.7
#topn_sim = 50
#topn_sim_lwlimit = 0
##sample = ratings.head(100)
#avg_profile_rating_k_sim_users = np.empty(len(ratings))
#avg_profile_rating_topn_sim_users = np.empty(len(ratings))
#swt_avg_profile_rating_k_sim_users  = np.empty(len(ratings))
#swt_avg_profile_rating_topn_sim_users = np.empty(len(ratings))
#wt_avg_profile_rating_k_sim_users = np.empty(len(ratings))
#wt_avg_profile_rating_topn_sim_users = np.empty(len(ratings))
#previous_active_user = -1
#for row in ratings.iterrows():
#   active_user = row[1]['UserID']-1
#   if (active_user!= previous_active_user):
#    # get similar users to the active user
#       print('change in user:', active_user)
#       previous_active_user = active_user
#       active_user_avg_prof_rating =  np.nanmean(rmat[active_user,])
#       sim_array_temp = np.array(corr_read_adj_new[active_user,:])[0]
#       sim_array_temp[np.where(np.isnan(sim_array_temp))] = -2
#       sim_array_above_trshld = sim_array_temp[sim_array_temp >= threshold_sim]
#       users_above_sim_trshld = (np.where([sim_array_temp >= threshold_sim])[1])
##       # Pick top n similar users 
##       topn_sim_lwlimit = 0
##       topn_sim_uplimit = 0
##       topn_sim_users =   np.argsort(sim_array_temp)[::-1][:topn_sim]
##       profile_ratings_topn_sim_users = rmat[topn_sim_users,row[1]['ProfileID']-1]
##       topn_sim_users_valid = topn_sim_users[~np.isnan(profile_ratings_topn_sim_users)]
##       profile_ratings_topn_sim_users_valid = rmat[topn_sim_users_valid,row[1]['ProfileID']-1]   
##       topn_sim_values_valid = sim_array_temp[topn_sim_users_valid]   
##    
##       while (  len(profile_ratings_topn_sim_users_valid) < 5 and topn_sim_uplimit < 1000):
##           topn_sim_lwlimit = topn_sim_lwlimit + topn_sim
##           topn_sim_uplimit = topn_sim_lwlimit + topn_sim
##       #    print(topn_sim_uplimit)
##           topn_sim_users =   np.argsort(sim_array_temp)[::-1][topn_sim_lwlimit:topn_sim_uplimit]
#           
#   profile_ratings_sim_trshld_users = rmat[users_above_sim_trshld,row[1]['ProfileID']-1]
#   users_above_sim_trshld_valid =  users_above_sim_trshld[~np.isnan(profile_ratings_sim_trshld_users)]
#   sim_array_above_trshld_valid = np.array(corr_read_adj_new[active_user,users_above_sim_trshld_valid])[0]
#   profile_ratings_sim_trshld_users_valid = rmat[users_above_sim_trshld_valid,row[1]['ProfileID']-1]
#   
#
##   profile_ratings_topn_sim_users = rmat[topn_sim_users,row[1]['ProfileID']-1]   
##   topn_sim_users_valid = topn_sim_users[~np.isnan(profile_ratings_topn_sim_users)]
##   profile_ratings_topn_sim_users_valid = rmat[topn_sim_users_valid,row[1]['ProfileID']-1]   
##   topn_sim_values_valid = sim_array_temp[topn_sim_users_valid]
#
#
#   
##   profile_ratings_sim_trshld_users[np.isnan(profile_ratings_sim_trshld_users)] = 0
##   np.dot(profile_ratings_sim_trshld_users,sim_array_above_trshld)/sum(sim_array_above_trshld[~np.isnan(sim_array_above_trshld)])
########## A simple average of the ratings from similar users
#   avg_profile_rating_k_sim_users[row[0]] = np.nanmean(profile_ratings_sim_trshld_users_valid)
# #  avg_profile_rating_topn_sim_users[row[0]] = np.nanmean(profile_ratings_topn_sim_users_valid)
#
########## A weighted average of the ratings from similar users   
#   swt_avg_profile_rating_k_sim_users[row[0]] = sum(sim_array_above_trshld_valid*profile_ratings_sim_trshld_users_valid)/(sum(sim_array_above_trshld_valid))
# #  swt_avg_profile_rating_topn_sim_users[row[0]] = sum(topn_sim_values_valid*profile_ratings_topn_sim_users_valid)/(sum(abs(topn_sim_values_valid)))
#
########## A weighted average of the ratings from similar users       
#   users_above_sim_avg_prof_rating= np.nanmean(rmat[users_above_sim_trshld_valid,], axis= 1)
#   users_above_sim_numer = sum((profile_ratings_sim_trshld_users_valid - users_above_sim_avg_prof_rating)*sim_array_above_trshld_valid)
#   wt_avg_profile_rating_k_sim_users[row[0]] = active_user_avg_prof_rating +(users_above_sim_numer/(sum(abs(sim_array_above_trshld_valid))))
#
# #  topn_sim_avg_prof_rating= np.mean(rmat[topn_sim_users_valid,], axis= 1)
# #  topn_sim_sim_numer = sum((profile_ratings_topn_sim_users_valid - topn_sim_avg_prof_rating)*topn_sim_values_valid)
# #  wt_avg_profile_rating_topn_sim_users[row[0]] = active_user_avg_prof_rating + (topn_sim_sim_numer/(sum(abs(topn_sim_values_valid))))


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


threshold_sim = 0.7
topn_sim = 50
topn_sim_lwlimit = 0
#sample = ratings.head(100)
avg_user_rating_k_sim_profiles = np.empty(len(ratings))
avg_user_rating_topn_sim_profiles = np.empty(len(ratings))
swt_avg_user_rating_k_sim_profiles = np.empty(len(ratings))
swt_avg_user_rating_topn_sim_profiles = np.empty(len(ratings))
wt_avg_user_rating_k_sim_profiles = np.empty(len(ratings))
wt_avg_user_rating_topn_sim_profiles = np.empty(len(ratings))
previous_active_profile = -1
for row in ratings.iterrows():
   print("row:",row[0])
#   active_profile = row[1]['ProfileID']-1
# get similar users to the active user
   if (active_profile!= previous_active_profile):
       print('change in profile:', active_profile)
       previous_active_profile = active_profile
       sim_array_temp = np.array(corr_prof_adj_new[active_profile,:])[0]
       sim_array_temp[np.where(np.isnan(sim_array_temp))] = -2
# get indices of top k similarities   - userids
#   sim_array_sorted = np.sort(sim_array_temp)[::-1][0:topk]
#   top_k_similar_users = np.argsort(sim_array_temp)[::-1][0:topk]
#   profile_ratings_k_sim_users = rmat[top_k_similar_users,row[1]['profileid']-1]
       sim_array_above_trshld = sim_array_temp[sim_array_temp >= threshold_sim]
       profiles_above_sim_trshld = (np.where([sim_array_temp >= threshold_sim])[1])
   user_ratings_sim_trshld_profiles = rmat[row[1]['UserID']-1,profiles_above_sim_trshld]
   profiles_above_sim_trshld_valid =  profiles_above_sim_trshld[~np.isnan(user_ratings_sim_trshld_profiles)]
   sim_array_above_trshld_valid = np.array(corr_prof_adj_new[active_profile,profiles_above_sim_trshld_valid])[0]
   user_ratings_sim_trshld_profiles_valid = rmat[row[1]['UserID']-1,profiles_above_sim_trshld_valid]

## Pick top n similar profiles 
#   topn_sim_lwlimit = 0  
#   topn_sim_uplimit = 0
#   topn_sim_profiles =   np.argsort(sim_array_temp)[::-1][:topn_sim]
#   user_ratings_topn_sim_profiles = rmat[row[1]['UserID']-1, topn_sim_profiles]
#   topn_sim_profiles_valid = topn_sim_profiles[~np.isnan(user_ratings_topn_sim_profiles)]
#   user_ratings_topn_sim_profiles_valid = rmat[row[1]['UserID']-1,topn_sim_profiles_valid]   
#   topn_sim_values_valid = sim_array_temp[topn_sim_profiles_valid]
#   
#   while (  len(user_ratings_topn_sim_profiles_valid) < 5 and topn_sim_uplimit < 10000):
#       topn_sim_lwlimit = topn_sim_lwlimit + topn_sim
#       topn_sim_uplimit = topn_sim_lwlimit + topn_sim
#      # print(topn_sim_uplimit)
#       topn_sim_profiles =   np.argsort(sim_array_temp)[::-1][topn_sim_lwlimit:topn_sim_uplimit]
#       user_ratings_topn_sim_profiles = rmat[row[1]['UserID']-1, topn_sim_profiles]   
#
#       topn_sim_profiles_valid = topn_sim_profiles[~np.isnan(user_ratings_topn_sim_profiles)]
#       user_ratings_topn_sim_profiles_valid = rmat[row[1]['UserID']-1,topn_sim_profiles_valid]   
#       topn_sim_values_valid = sim_array_temp[topn_sim_profiles_valid]

#   profile_ratings_sim_trshld_users[np.isnan(profile_ratings_sim_trshld_users)] = 0
#   np.dot(profile_ratings_sim_trshld_users,sim_array_above_trshld)/sum(sim_array_above_trshld[~np.isnan(sim_array_above_trshld)])
   avg_user_rating_k_sim_profiles[row[0]] = np.nanmean(user_ratings_sim_trshld_profiles_valid)
 #  avg_user_rating_topn_sim_profiles[row[0]] = np.nanmean(user_ratings_topn_sim_profiles_valid)

   swt_avg_user_rating_k_sim_profiles[row[0]] = sum(sim_array_above_trshld_valid*user_ratings_sim_trshld_profiles_valid)/(sum(sim_array_above_trshld_valid))
 #  swt_avg_profile_rating_topn_sim_users[row[0]] = sum(topn_sim_values_valid*profile_ratings_topn_sim_users_valid)/(sum(abs(topn_sim_values_valid)))

######### A weighted average of the ratings from similar users   
   active_profile_avg_user_rating =  np.nanmean(rmat[:,active_profile])
 
   profiles_above_sim_avg_user_rating= np.nanmean(rmat[:,profiles_above_sim_trshld_valid], axis= 0)
   profiles_above_sim_numer = sum((user_ratings_sim_trshld_profiles_valid - profiles_above_sim_avg_user_rating)*sim_array_above_trshld_valid)
   wt_avg_user_rating_k_sim_profiles[row[0]] = active_profile_avg_user_rating +(profiles_above_sim_numer/(sum(abs(sim_array_above_trshld_valid))))

#   topn_sim_avg_user_rating= np.mean(rmat[:,topn_sim_profiles_valid], axis= 0)
#   topn_sim_sim_numer = sum((user_ratings_topn_sim_profiles_valid - topn_sim_avg_user_rating)*topn_sim_values_valid)
#   wt_avg_user_rating_topn_sim_profiles[row[0]] = active_profile_avg_user_rating + (topn_sim_sim_numer/(sum(abs(topn_sim_values_valid))))


#####
###
#avg_profile_rating = np.nanmean(rmat,axis=0)
#data = { 'ProfileID': range(1,10001),'avg_profile_ratings_all_users':avg_profile_rating}   
#avg_profile_ratings_all_users_df=pd.DataFrame(data)
#ratings = pd.merge(ratings,avg_profile_ratings_all_users_df ,
#                                  left_on=['ProfileID'],right_on=['ProfileID'],
#      how='inner')
      
#del ratings_ext_input_sim2['UserID']

      
data = { 'avg_user_rating_k_sim_profiles':avg_user_rating_k_sim_profiles,
        'swt_avg_user_rating_k_sim_profiles':swt_avg_user_rating_k_sim_profiles,
        'wt_avg_user_rating_k_sim_profiles':wt_avg_user_rating_k_sim_profiles}        
#        'avg_profile_rating_k_sim_users':avg_profile_rating_k_sim_users,
#        'swt_avg_profile_rating_k_sim_users':swt_avg_profile_rating_k_sim_users,
#        'wt_avg_profile_rating_k_sim_users':wt_avg_profile_rating_k_sim_users}
        
new_cols=pd.DataFrame(data=data)
ratings = pd.concat([ratings,new_cols],axis=1)


#### gender and average profile rating
#ratings = pd.merge(ratings, gender,left_on=['UserID'],right_on=['UserID'],
#      how='inner')
#gender_dict = dict([('M', 1), ('F', 2), ('U', 3)])
#gender_data = pd.DataFrame(data={'gender_num':[gender_dict[w] for w in ratings['Gender']]})
#ratings = pd.concat([ratings,gender_data],axis=1)

#X_features = ['top1_rating','top2_rating','top3_rating','top1_sim','top2_sim','top3_sim',
#              'gender_num','avg_profile_ratings_users','avg_profile_ratings_all_users']
#   
ratings.to_csv("ratings_X_user_profile_features.csv")