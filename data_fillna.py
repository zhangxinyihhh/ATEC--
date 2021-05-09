# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
# train1=pd.read_csv('data/atec_anti_fraud_train.csv',encoding='utf-8')
train1=pd.read_csv('data/atec_anti_fraud_test_b.csv',encoding='utf-8')
# train2=pd.read_csv('input/train2.csv')
# test=pd.read_csv('input/test.csv')
# train1=train1[train1.label!=-1]
train1=train1.fillna(-100)
#处理缺失
def queshi(x):
    if x==-100:
        return -1
    else:
        return 1


# train1 = train1.drop(['id'], axis = 1)
train1['ifexistf5']=train1.f5.apply(queshi)
train1['ifexist20-23']=train1.f20.apply(queshi)
train1['ifexist24-27'] = train1.f24.apply(queshi)
train1['ifexist28-31'] = train1.f28.apply(queshi)
train1['ifexist32-35'] = train1.f32.apply(queshi)
train1['ifexist36-47'] = train1.f36.apply(queshi)
train1['ifexist48-51'] = train1.f48.apply(queshi)
train1['ifexist52-53'] = train1.f52.apply(queshi)
train1['ifexist54-63'] = train1.f54.apply(queshi)
train1['ifexist64-71'] = train1.f64.apply(queshi)
train1['ifexist72-75'] = train1.f72.apply(queshi)
train1['ifexist76-101'] = train1.f76.apply(queshi)
train1['ifexist102-106'] = train1.f102.apply(queshi)
train1['ifexist107-110'] = train1.f107.apply(queshi)
train1['ifexist111-154'] = train1.f111.apply(queshi)
train1['ifexist155-160'] = train1.f155.apply(queshi)
train1['ifexist161-165'] = train1.f161.apply(queshi)
train1['ifexist166-210'] = train1.f166.apply(queshi)
train1['ifexist211-253'] = train1.f211.apply(queshi)
train1['ifexist254-277'] = train1.f254.apply(queshi)
train1['ifexist278-297'] = train1.f278.apply(queshi)

train1=train1.replace(-100,np.nan)

train1.to_csv('input/test.csv',index=False)










