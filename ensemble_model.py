# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve
import numpy as np

# train1=pd.read_csv('input/train1_drop.csv')
# train2=pd.read_csv('input/train2_drop.csv')
test=pd.read_csv('input/test_drop_fillna_labeled.csv')

# train1=train1.fillna(0)
# train2=train2.fillna(0)
train=pd.read_csv('input/train_drop_fillna_labeled.csv')
print(train.label.unique())
train.label=train.label.replace(-1,1)
print(train.label.unique())
# train=train.fillna(0)
# test=test.fillna(0)
num_train=40

train_data=[]
for i in range(num_train):
    train_data.append(train.iloc[i*25000:i*25000+25000])



col1 = list(train.columns)[2:]
lr=[]
lg=[]
xgbmodel=[]
res=[]
grad=[]
rf=[]
et=[]
for i in range(num_train):
    # rf.append(RandomForestClassifier(n_estimators=20,max_depth=10,n_jobs=-1))
    # et.append(ExtraTreesClassifier(n_estimators=20,max_depth=10,n_jobs=-1))
    lg.append(lgb.LGBMClassifier(max_depth = 8, n_estimators = 100,num_leaves=31))
    xgbmodel.append(xgb.XGBClassifier(max_depth = 8, n_estimators = 30,num_leaves=31))
    grad.append(GradientBoostingClassifier(n_estimators=15,max_depth=10))
    # lr.append(LogisticRegression(C=1.0))

clf = []
for i in range(num_train):
      # clf.append( StackingClassifier(classifiers=[lg[i], xgbmodel[i]],use_probas=True,average_probas=False,meta_classifier=lr[i]))
      clf.append(VotingClassifier(estimators = [('lg',lg[i]), ('xgb',xgbmodel[i]),('grad',grad[i])], voting = 'soft'))
      clf[i].fit(train_data[i][col1].as_matrix(),train_data[i]['label'])
      res.append(clf[i].predict_proba(test[col1].as_matrix())[:,1])

def score(y,pred):
    fpr,tpr,thresholds=roc_curve(y,pred,pos_label=1)
    score1 = 0.4 * tpr[np.where(fpr >= 0.001)[0][0]] + 0.3 * tpr[np.where(fpr >= 0.005)[0][0]] + 0.3 * tpr[np.where(fpr >= 0.01)[0][0]]
    return 'selfeval', score1, True

# print(score(train.label,res))


end=pd.DataFrame({'id':test['id'],'score':sum(res)/40})
print(end.describe())
end.to_csv('output/lgb_xgb_grad_preds_drop161_40_fillmode_labeled.csv',index=False)






