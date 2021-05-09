# -*- coding:utf-8 -*-

import pandas as pd
import lightgbm as lgb
import numpy as np
from tqdm import tqdm

train=pd.read_csv('input/train_drop.csv')
train=train.drop_duplicates()
# train2=pd.read_csv('input/train2_drop.csv')
# test=pd.read_csv('input/test.csv')
train.label=train.label.replace(-1,1)

train1=train[train.date<20171001]
train2=train[train.date>=20171001]
col1=list(train.columns)[2:]



def preprocess(data: pd.DataFrame):
    """ 对数据进行预处理
    """

    def fill_outliers(col: pd.Series):
        """ Remove outliers of each col
        """
        mean, std = col.mean(), col.std()
        upper, lower = mean + 3 * std, mean - 3 * std
        col[col > upper] = np.floor(upper)
        col[col < lower] = np.floor(lower)
        return col.values

    # 处理离散值 & 填充空值(使用众数填充)
    columns = data.columns
    for col_name in tqdm(columns):
        data[col_name] = fill_outliers(data[col_name].copy())
        mode = data[col_name].mode().values[0]
        data[col_name] = data[col_name].fillna(mode)

    return data
col=col1[:136]

nunique1 = train1[col].nunique()  # 每个特征分量unique值的数量
categorical_feature1 = list(nunique1[nunique1<= 10].index.values) # 所有类别变量的名称
data_train1 = preprocess(train1[col1].copy())

# nunique2 = data_test[col1].nunique()  # 每个特征分量unique值的数量
# categorical_feature2 = list(nunique2[nunique2 <= 10].index.values) # 所有类别变量的名称
data_train2 = preprocess(train2[col1].copy())

train_y=train1.label
train_x=data_train1
lgb_train=lgb.Dataset(train_x,train_y)
del train_x,train_y

val=train2
val_y=val.label
val_x=data_train2
lgb_val=lgb.Dataset(val_x,val_y)
del val_x,val_y,val

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
    'num_leaves': 31,
    'learning_rate': 0.1,
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
                categorical_feature=categorical_feature1,
                early_stopping_rounds=50)

# save model to fileá
gbm.save_model('lgb1.txt')

# save feature score
fs = pd.DataFrame(columns=['feature', 'score'])
fs['feature'] = list(gbm.feature_name())
fs['score'] = list(gbm.feature_importance())
fs.sort_values(by='score', ascending=False, inplace=True)
fs.to_csv('lgb1_feature_score.csv', index=None)





