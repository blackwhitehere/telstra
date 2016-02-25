__author__ = 'stan'

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utility as u
from sklearn import preprocessing as pre

join=u.cat_join()
cat_cols=u.get_cat_cols()
samples=u.get_sample_names()
datasets=u.get_datasets()

stats=u.get_stats_features(join)
priors=u.get_prior_features(join)
datasets['stats']=stats
datasets['priors']=priors

#convert tables of categorical classes to binary row feature vectors for each unique id. Continuous factors are not affected
for key, dataset in datasets.items():
    if key not in ['stats','priors']:
        tmp=pd.get_dummies(dataset,dummy_na=True)
        tmp=tmp.groupby('id').sum()
        tmp['id']=tmp.index
        dataset=tmp
        datasets[key]=dataset

#joins features from different datasets into one dataframe, features only for ids that appear in train/test are extracted
#(i.e. some features may not be related to any id)
for key, dataset in datasets.items():
    if key not in samples:
        join=pd.merge(join,dataset,on='id',how='left')

join=pd.merge(join,stats,on='id')
join=join.set_index(['sample'])
join=pd.get_dummies(join) #fixes dummies for location

#TODO: TRY REMOVING ROWS WITH MISSING DATA
#fills in missing data
join=join.fillna(value=0)

#creates features and targets for final training
features=join.columns.values.tolist()
not_features=['fault_severity', 'id']
for x in features:
    if "_nan" in x:
        features.remove(x)
        not_features.append(x)

for x in not_features:
    if x in features:
        features.remove(x)

train_features = join.loc['train'][features]
train_target = join.loc['train']['fault_severity']

test_features = join.loc['test'][features]

#XGBOOST
import xgboost as xgb

sz=(train_features.values).shape

train_X = train_features.values[:int(sz[0] * 0.7), :]
train_Y = train_target.values[:int(sz[0]*0.7)]

test_X = train_features.values[int(sz[0] * 0.7):, :]
test_Y = train_target.values[int(sz[0] * 0.7):]

xg_train = xgb.DMatrix(train_X, label=train_Y,feature_names=train_features.columns) #
full_xg_train = xgb.DMatrix(train_features, label=train_target.values,feature_names=train_features.columns) #

xg_test = xgb.DMatrix(test_X, label=test_Y,feature_names=test_features.columns) #
full_xg_test = xgb.DMatrix(test_features,feature_names=test_features.columns) #

#TRAIN
# setup parameters for xgboost
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 1
param['gamma']= 4
param['max_depth'] = 15
param['min_child_weight']=4
param['max_delta_step']=4
param['subsample']=0.5

param['nthread'] = 3
#param['subsample']=1
param['num_class'] = 3
param['eval_metric']='mlogloss'



watchlist = [(xg_train,'train'), (xg_test, 'test')]
num_round = 4
bst = xgb.train(param, xg_train, num_round, evals=watchlist);

print(xgb.cv(param, xg_train, num_round, nfold=3, metrics=['mlogloss'], seed=0))

xgb.plot_importance(booster=bst)
plt.show()

bst=xgb.train(param, full_xg_train, num_round, evals=watchlist)
test_response=bst.predict(full_xg_test)
print(xgb.cv(param, xg_train, num_round, nfold=3, metrics=['mlogloss'], seed = 0))



#save predictions to required format
out=pd.DataFrame({'id':[],'predict_0':[],'predict_1':[],'predict_2':[]})
out['id']=join.loc['test']['id']
out['predict_0']=test_response.T[0]
out['predict_1']=test_response.T[1]
out['predict_2']=test_response.T[2]
out=out[['id','predict_0','predict_1','predict_2']]
out.to_csv('predictions.csv',index=False)