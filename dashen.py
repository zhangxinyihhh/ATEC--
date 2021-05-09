# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

data_train = pd.read_csv("D:/workspace/mayi/data/atec_anti_fraud_train.csv")
data_test = pd.read_csv("D:/workspace/mayi/data/atec_anti_fraud_test_b.csv")
# unlabel = data_train[data_train.label == -1].drop(['id'], axis = 1)
# data_train = data_train[data_train.label != -1].drop(['id'], axis = 1).sort_values(by = 'date')
# unlabel = pd.read_csv("F:/contest/ATEC/unlabel.csv")
data_train = data_train.drop(['id'], axis = 1)
data_train['ifexist5'] = np.where(data_train.f5.isnull() == True, -1, 1 )
data_train['ifexist20-23'] = np.where(data_train.f20.isnull() == True, -1, 1 )
data_train['ifexist24-27'] = np.where(data_train.f24.isnull() == True, -1, 1)
data_train['ifexist28-31'] = np.where(data_train.f28.isnull() == True, -1, 1)
data_train['ifexist32-35'] = np.where(data_train.f32.isnull() == True, -1, 1)
data_train['ifexist36-47'] = np.where(data_train.f36.isnull() == True, -1, 1)
data_train['ifexist48-51'] = np.where(data_train.f48.isnull() == True, -1, 1)
data_train['ifexist52-53'] = np.where(data_train.f52.isnull() == True, -1, 1)
data_train['ifexist54-63'] = np.where(data_train.f54.isnull() == True, -1, 1)
data_train['ifexist64-71'] = np.where(data_train.f64.isnull() == True, -1, 1)
data_train['ifexist72-75'] = np.where(data_train.f72.isnull() == True, -1, 1)
data_train['ifexist76-101'] = np.where(data_train.f76.isnull() == True, -1, 1)
data_train['ifexist102-106'] = np.where(data_train.f102.isnull() == True, -1, 1)
data_train['ifexist107-110'] = np.where(data_train.f107.isnull() == True, -1, 1)
data_train['ifexist111-154'] = np.where(data_train.f111.isnull() == True, -1, 1)
data_train['ifexist155-160'] = np.where(data_train.f155.isnull() == True, -1, 1)
data_train['ifexist161-165'] = np.where(data_train.f161.isnull() == True, -1, 1)
data_train['ifexist166-210'] = np.where(data_train.f166.isnull() == True, -1, 1)
data_train['ifexist211-253'] = np.where(data_train.f211.isnull() == True, -1, 1)
data_train['ifexist254-277'] = np.where(data_train.f254.isnull() == True, -1, 1)
data_train['ifexist278-297'] = np.where(data_train.f278.isnull() == True, -1, 1)

data_test= data_test.sort_values(by = 'date')
dt = data_train
data_train = data_test
data_train['ifexist5'] = np.where(data_train.f5.isnull() == True, -1, 1 )
data_train['ifexist20-23'] = np.where(data_train.f20.isnull() == True, -1, 1 )
data_train['ifexist24-27'] = np.where(data_train.f24.isnull() == True, -1, 1)
data_train['ifexist28-31'] = np.where(data_train.f28.isnull() == True, -1, 1)
data_train['ifexist32-35'] = np.where(data_train.f32.isnull() == True, -1, 1)
data_train['ifexist36-47'] = np.where(data_train.f36.isnull() == True, -1, 1)
data_train['ifexist48-51'] = np.where(data_train.f48.isnull() == True, -1, 1)
data_train['ifexist52-53'] = np.where(data_train.f52.isnull() == True, -1, 1)
data_train['ifexist54-63'] = np.where(data_train.f54.isnull() == True, -1, 1)
data_train['ifexist64-71'] = np.where(data_train.f64.isnull() == True, -1, 1)
data_train['ifexist72-75'] = np.where(data_train.f72.isnull() == True, -1, 1)
data_train['ifexist76-101'] = np.where(data_train.f76.isnull() == True, -1, 1)
data_train['ifexist102-106'] = np.where(data_train.f102.isnull() == True, -1, 1)
data_train['ifexist107-110'] = np.where(data_train.f107.isnull() == True, -1, 1)
data_train['ifexist111-154'] = np.where(data_train.f111.isnull() == True, -1, 1)
data_train['ifexist155-160'] = np.where(data_train.f155.isnull() == True, -1, 1)
data_train['ifexist161-165'] = np.where(data_train.f161.isnull() == True, -1, 1)
data_train['ifexist166-210'] = np.where(data_train.f166.isnull() == True, -1, 1)
data_train['ifexist211-253'] = np.where(data_train.f211.isnull() == True, -1, 1)
data_train['ifexist254-277'] = np.where(data_train.f254.isnull() == True, -1, 1)
data_train['ifexist278-297'] = np.where(data_train.f278.isnull() == True, -1, 1)

data_test = data_train
data_train = dt
del dt

col = list(data_train.columns)[2:]
giveup = []
for i in col:
    #     if data_train[i].isnull().sum() > 250000 :
    if data_train[i].isnull().sum() > 200000:
        giveup.append(i)

col1 = [i for i in col if i not in giveup]
data_train.label.unique()

import lightgbm as lgb
lg = lgb.LGBMClassifier(max_depth = 8, n_estimators = 100, min_child_samples = 100)
lg.fit(data_train[col1].iloc[:350000].as_matrix(),data_train.label[:350000])

from sklearn.metrics import recall_score

def score(y,pred):
#     precesion = (pred[y == 1] > 0.5)/len(pred)
    a = pred > 0.5
    recall = recall_score(y,pred)
    return 'self',recall,True







import lightgbm as lgb

lg = lgb.LGBMClassifier(max_depth=8,n_estimators = 100,  min_child_samples=100,num_leaves=16)
lg.fit(data_train[col1].as_matrix(), data_train.label)

data_train = data_train.fillna(0)
data_test = data_test.fillna(0)
num_train = 10
train = []
for i in range(num_train):
    train.append(data_train.iloc[i*100000:i*100000+100000])

import lightgbm as lgb

lg = []
res = []
for i in range(num_train):
    lg.append(lgb.LGBMClassifier(max_depth=8, n_estimators=100, min_child_samples=100,num_leaves=31))
    lg[i].fit(train[i][col].as_matrix(), train[i]['label'])
    res.append(lg[i].predict_proba(data_test[col].as_matrix())[:, 1])

# col.remove('label')
#------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

rf = []
res = []

for i in range(num_train):
    rf.append(RandomForestClassifier(max_depth=8, n_estimators=30))
    rf[i].fit(train[i][col1].as_matrix(), train[i]['label'])
    res.append(rf[i].predict_proba(data_test[col1].as_matrix())[:, 1])
print(train[9].shape)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from mlxtend.classifier import StackingClassifier


lg = []
rf = []
lr = []
xgbmodel = []
res = []


for i in range(num_train):
    lg.append(lgb.LGBMClassifier(max_depth = 8, n_estimators = 100))
#     rf.append(RandomForestClassifier(max_depth = 8, n_estimators = 30))
    xgbmodel.append(xgb.XGBClassifier(max_depth = 8, n_estimators = 30))
#     lr.append(LogisticRegression(C = 1.0))
clf = []
for i in range(num_train):
#     clf.append( StackingClassifier(classifiers=[lg[i], xgbmodel[i]],
#                           average_probas=False,

#                           meta_classifier=lr[i]))
      clf.append(VotingClassifier(estimators = [('lg',lg[i]), ('xgb',xgbmodel[i])], voting = 'soft'))
      clf[i].fit(train[i][col1].as_matrix(),train[i]['label'])
#       append(clf[i].predict_proba(data_test[col1].as_matrix())[:,1])
      res.append(clf[i].predict_proba(data_test[col1].as_matrix())[:,1])


import lightgbm as lgb
lg1 = []
res = []
for i in range(10):
    lg1.append(lgb.LGBMClassifier(max_depth = 8, n_estimators = 100))
    lg1[i].fit(train[i][col1].as_matrix(),train[i]['label'])
    res.append(lg1[i].predict_proba(data_test[col1].as_matrix())[:,1])

from sklearn.metrics import roc_curve

def score1(y,pred):
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    score1=0.4*tpr[np.where(fpr>=0.001)[0][0]]+0.3*tpr[np.where(fpr>=0.005)[0][0]]+0.3*tpr[np.where(fpr>=0.01)[0][0]]
    return 'selfeval',score1, True

end = pd.DataFrame({'id': data_test['id'],'score':sum(res)/10})
end.to_csv("D:/workspace/mayi/output/preds.csv", index = False)
# print(end)

