# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
train=pd.read_csv('data/atec_anti_fraud_train.csv',encoding='utf-8')
test=pd.read_csv('data/atec_anti_fraud_test_b.csv',encoding='utf-8')
train.label=train.label.replace(-1,1)
# 处理缺失多的特征
def process_na(train):
    train=train.fillna(-100)
    def queshi(x):
        if x==-100:
            return -1
        else:
            return 1

    train['ifexistf5']=train.f5.apply(queshi)
    train['ifexist20-23']=train.f20.apply(queshi)
    train['ifexist24-27'] = train.f24.apply(queshi)
    train['ifexist28-31'] = train.f28.apply(queshi)
    train['ifexist32-35'] = train.f32.apply(queshi)
    train['ifexist36-47'] = train.f36.apply(queshi)
    train['ifexist48-51'] = train.f48.apply(queshi)
    train['ifexist52-53'] = train.f52.apply(queshi)
    train['ifexist54-63'] = train.f54.apply(queshi)
    train['ifexist64-71'] = train.f64.apply(queshi)
    train['ifexist72-75'] = train.f72.apply(queshi)
    train['ifexist76-101'] = train.f76.apply(queshi)
    train['ifexist102-106'] = train.f102.apply(queshi)
    train['ifexist107-110'] = train.f107.apply(queshi)
    train['ifexist111-154'] = train.f111.apply(queshi)
    train['ifexist155-160'] = train.f155.apply(queshi)
    train['ifexist161-165'] = train.f161.apply(queshi)
    train['ifexist166-210'] = train.f166.apply(queshi)
    train['ifexist211-253'] = train.f211.apply(queshi)
    train['ifexist254-277'] = train.f254.apply(queshi)
    train['ifexist278-297'] = train.f278.apply(queshi)

    train=train.replace(-100,np.nan)
    return train
train=train.drop(['id'],axis=1)
train=process_na(train)
test=process_na(test)

#扔掉缺失值多的特征：
def giveup_features(train,test):
    col = list(train.columns)[2:]
    giveup = []
    for i in col:
        #     if data_train[i].isnull().sum() > 250000 :
        if train[i].isnull().sum() > 200000:
            giveup.append(i)
    test_drop=test.drop(giveup, axis=1)

    return test_drop
train_drop=giveup_features(train,train)
test_drop=giveup_features(train,test)

train_drop.to_csv('input/train_drop.csv',index=False)
test_drop.to_csv('input/test_drop.csv',index=False)






























