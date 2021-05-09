import pandas as pd



data_train=pd.read_csv('../input/atec_anti_fraud_train.csv')

data_train = data_train.drop(['id'], axis=1).sort_values(by='date')

colms = list(data_train.columns)[2:]
feat_drop = []
for colu in colms:
    if data_train[colu].isnull().sum() > 200000:
        feat_drop.append(colu)

data_train=data_train.fillna(-1)
data_train=data_train.label.replace(-1,1)

select_feat=['f5','f20','f28','f32','f36','f48','f52','f54','f64','f72','f76','f102','f107','f111','f155','f161','f166','f211','f254','f278']

def produce_feat(dataset):
    for feat in select_feat:
        dataset[feat+'solve_nan']=dataset[feat].apply(lambda x:x if x==-1 else 1)
    return dataset

train=produce_feat(data_train)
train=train.drop(feat_drop,axis=1)
train.drop('date',axis=1,inplace=True)
del data_train
train.to_csv('../output/latest/train.csv',index=None)
print(train.info())

train1=train.loc[:800000]
train1.to_csv('../output/latest/train1.csv',index=None)
del train1

train2=train.loc[800000:]
train2.to_csv('../output/latest/train2.csv',index=None)
del train2,train





data_test = pd.read_csv('../input/atec_anti_fraud_test_b.csv')
data_test=data_test.fillna(-1)
test=produce_feat(data_test)
del data_test
test.drop(feat_drop,axis=1,inplace=True)
test=test.drop('date',axis=1)
del data_test
test.to_csv('../output/latest/test.csv',index=None)
print(test.info)
del test