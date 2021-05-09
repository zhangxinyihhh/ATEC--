# -*- coding:utf-8 -*-
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from operator import itemgetter


train=pd.read_csv('input/train_drop.csv')
test=pd.read_csv('input/test_drop.csv')


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

    # 处理离散值 & 填充空值(使用众数填充)
    columns = data.columns
    for col_name in tqdm(columns):
        data[col_name] = fill_outliers(data[col_name].copy())
        mode = data[col_name].mode().values[0]
        data[col_name] = data[col_name].fillna(mode).astype('float64')
        # data[col_name] = data[col_name].fillna(-1)

    return data


train_col=list(train.columns)[2:]
test_col=list(test.columns)[2:]
train[train_col]=preprocess(train[train_col].copy())
test[test_col]=preprocess(test[test_col].copy())
train.to_csv('input/train_drop_fill-1.csv',index=False)
test.to_csv('input/test_drop_fill-1.csv',index=False)











