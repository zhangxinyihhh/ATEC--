# -*- coding:utf-8 -*-
import pandas as pd
train=pd.read_csv('input/train.csv')
test=pd.read_csv('input/test.csv')
#填充缺失标签
train.label=train.label.replace(-1,1)

from tqdm import tqdm
import numpy as np


def preprocess(data: pd.DataFrame):
    """ 对数据进行预处理
    """

    def fill_outliers(col: pd.Series):
        """ Remove outliers of each col
        """
        mean = col.mean()
        std = col.std()
        upper = mean + 3 * std
        lower = mean - 3 * std
        col[col > upper] = np.floor(upper)
        col[col < lower] = np.floor(lower)
        return col.values

    # 处理离散值 & 填充空值(使用均值填充)
    columns = data.columns
    for col_name in tqdm(columns):
        data[col_name] = fill_outliers(data[col_name].copy())
        data[col_name] = data[col_name].fillna(data[col_name].mean())
    return data

# 筛选缺失率小于0.6的特征
col=list(train.columns)[2:]
giveup=[]
for item in col:
    tmp = np.sum(train[item].isnull()) / len(train)
    if tmp >= 0.2:
        giveup.append(item)
# train=train.drop(giveup,axis=1)
# test=test.drop(giveup,axis=1)
#
# col1=list(train.columns)[2:]
# train[col1]=preprocess(train[col1].copy())
# test[col1]=preprocess(test[col1].copy())
# train.to_csv('input/train_drop_fillmean.csv',index=False)
# test.to_csv('input/test_drop_fillmean.csv',index=False)
print(len(giveup))










