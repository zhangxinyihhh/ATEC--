# -*- coding:utf-8 -*-
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
test = pd.read_csv('input/test.csv')
test=test.drop_duplicates()
col1=list(test.columns)[2:]
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

data_test = preprocess(test[col1].copy())
test_preds = test[['id']].copy()
test_x = data_test

# load model to predict
print('Load model to predict')
gbm = lgb.Booster(model_file='lgb1.txt')


#predict test set
test_preds['label'] = gbm.predict(test_x)
test_preds.label = MinMaxScaler().fit_transform(test_preds.label.reshape(-1, 1))
# test_preds.label = test_preds.user_label.apply(margin)
# test_preds=test_preds[test_preds.user_label.isin([1])].drop(['user_label'],axis=1).drop_duplicates()

test_preds.to_csv("output/lgb_preds.csv",index=None,header=['id','score'],sep=',')
print (test_preds.describe())























