# -*- coding:utf-8 -*-
import pandas as pd
# train= pd.read_csv('data/atec_anti_fraud_train.csv',encoding='utf-8')
test= pd.read_csv('data/atec_anti_fraud_test_b.csv',encoding='utf-8')

# train.label=train.label.replace(-1,1)
#
# train1=train[(train.date>=20170905)&(train.date<=20171005)]
# train2=train[(train.date>=20171006)&(train.date<=20171105)]
#
# train1.to_csv('input/train1.csv',index=False)
# train2.to_csv('input/train2.csv',index=False)
test.to_csv('input/test.csv',index=False)


