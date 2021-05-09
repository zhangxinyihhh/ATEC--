# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
train= pd.read_csv('data/atec_anti_fraud_train.csv',encoding='utf-8')
unlabel=pd.read_csv('input/unlabel_process.csv')

t=train[train.label==-1]
t=pd.merge(t,unlabel,on='id',how='left')
t.label_x=t.label_y
t=t.drop(['label_y'],axis=1)
t.rename(columns={'label_x':'label'},inplace=True)
label=train[train.label!=-1]
train=pd.concat([label,unlabel],axis=0)
train=pd.DataFrame(train)
print(train.label.unique())
train.to_csv('input/train_label.csv',index=False)



















