# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from mlxtend.classifier import StackingClassifier
from sklearn.metrics import recall_score

train1=pd.read_csv('input/train1_drop.csv')
train2=pd.read_csv('input/train2_drop.csv')
# test=pd.read_csv('input/test.csv')

train1=train1.fillna(0).drop_duplicates()
train2=train2.fillna(0).drop_duplicates()
# train=pd.concat([train1,train2],axis=0)
# test=test.fillna(0)
# num_train=10

train_y=train1.label
train_x=train1.drop(['label','date'],axis=1)
lgb_train=lgb.Dataset(train_x,train_y)
del train_x,train_y

val=train2
val_y=val.label
val_x=val.drop(['date','label'],axis=1)
lgb_val=lgb.Dataset(val_x,val_y)
del val_x,val_y,val

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'lambda_l1': 2.0,
    'lambda_l2': 5.0,
    'min_gain_to_split': 0.001,
    'min_sum_hessian_in_leaf': 1.0,
    'max_depth': 8,
    'bagging_freq': 1,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20000,
                valid_sets=[lgb_train, lgb_val],
                early_stopping_rounds=50)

# save model to fileá
gbm.save_model('lgb1.txt')

# save feature score
fs = pd.DataFrame(columns=['feature', 'score'])
fs['feature'] = list(gbm.feature_name())
fs['score'] = list(gbm.feature_importance())
fs.sort_values(by='score', ascending=False, inplace=True)
fs.to_csv('lgb1_feature_score.csv', index=None)





