# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
train1=pd.read_csv('input/train1.csv')
train2=pd.read_csv('input/train2.csv')
test=pd.read_csv('input/test.csv')

# train=train1[train1.label!=-1]
train=pd.concat([train1,train2],axis=0)
col = list(train.columns)[2:]
giveup = []
for i in col:
    #     if data_train[i].isnull().sum() > 250000 :
    if train[i].isnull().sum() > 200000:
        giveup.append(i)
# print(len(giveup))
# # # # col1 = [i for i in col if i not in giveup]
test=test.drop(giveup,axis=1)
# # print(train.info())
test.to_csv('input/test_drop.csv',index=False)














