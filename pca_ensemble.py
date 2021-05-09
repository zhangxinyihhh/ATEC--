# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_curve
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from operator import itemgetter

test=pd.read_csv('input/test_drop.csv')
train=pd.read_csv('input/train_drop.csv')
train.label=train.label.replace(-1,1)
print(train.label.unique())

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

    return data


def normalization(X):
    """ 对样本进行归一化处理: 标准归一化 + 最小最大归一化
    """
    ss_scaler = StandardScaler()
    mm_scaler = MinMaxScaler()
    new_X = ss_scaler.fit_transform(X)
    new_X = mm_scaler.fit_transform(new_X)

    return new_X, [ss_scaler, mm_scaler]

def make_field_pipeline(field: str, *vec) -> Pipeline:
    """ Make Pipeline with refer to field : `field`, and some transform functions: `*vec`
    Input:
        - field: a data field
        - *vec: a sequence transformance functions
    """
    return make_pipeline(FunctionTransformer(itemgetter(field), validate=False), *vec)

#pca处理：
def pca_process(train):
    col=list(train.columns)[2:]

    # 需要做PCA处理的变量
    pca_vars_lists = []
    pca_vars_all = []
    pca_vars_name = 'pca_variables.txt'
    with open(pca_vars_name, 'r') as f:
        for line in f:
            list_ = ['f' + str(int(ele)) for ele in line.split(',')]
            pca_vars_lists.append(list_)
            pca_vars_all += list_

    # 排除了需要做PCA的变量, 剩下的变量
    other_vars = [ele for ele in col if ele not in pca_vars_all]
    vectorizer = make_union(
        *[make_field_pipeline(pca_vars_lists[i],PCA(n_components=int(np.ceil(len(pca_vars_lists[i]) / 2)))) for i in range(len(pca_vars_lists))],
        make_field_pipeline(other_vars)
    )
    feats=vectorizer.fit_transform(train[col])
    return feats

train_col=list(train.columns)[2:]
test_col=list(test.columns)[2:]
train[train_col]=preprocess(train[train_col].copy())
test[test_col]=preprocess(test[test_col].copy())
feats=pca_process(train)
# 分成训练和测试
feats_train = feats[((train['label'] == 1) | (train['label'] == 0)).values, :]
y_train = train.loc[(train['label'] == 1) | (train['label'] == 0), 'label'].values
test['label']=np.nan
feats_test = feats[test['label'].isnull().values, :]

pos_train, neg_train = feats_train[y_train == 1], feats_train[y_train == 0]
#
# rf = RandomForestClassifier(n_estimators=20,
#                                 max_depth=10,
#                                 n_jobs=-1)
#
num_pos = int(sum(y_train))  # 正样本数量
num_neg = int(len(y_train) - num_pos)  # 负样本数量
#
# # 训练多个模型
# classify_models, classify_num = [], 10
# neg_rate = 4
#
# for i in range(classify_num):
#     print('Training ', str(i), ' classifier')
#
# Make samples
num_pos_select = int(num_pos - 3000)
num_neg_select = int(np.floor(num_pos_select * 2.5))  # 选择的负样本数量

# 随机选出一部分正样本
pos_flag = np.random.choice([1, 0], num_pos, p=[num_pos_select / num_pos,
                                                1 - num_pos_select / num_pos])
# 随机选出一部分负样本
neg_flag = np.random.choice([1, 0], num_neg, p=[num_neg_select / num_neg,
                                                1 - num_neg_select / num_neg])

X = np.vstack((pos_train[pos_flag == 1], neg_train[neg_flag == 1]))
y = np.squeeze(np.vstack((np.ones((sum(pos_flag), 1)), np.zeros((sum(neg_flag), 1)))))

#     # Fit the model
#     rf = rf.fit(X=X, y=y)
#     classify_models.append(rf)
#
#     ## Test on the train dataset
#     y_pred_train = rf.predict(X)
#     y_prob_train = rf.predict_proba(X)[:, np.where(rf.classes_ == 1)[0][0]]
#     #
#     # print('test score on train dataset is ', str(score(y, y_prob_train)))
#     # print(confusion_matrix(y, y_pred_train))
#
#     ## Test on the whole dataset
#     y_pred = rf.predict(feats_train)
#     y_prob = rf.predict_proba(feats_train)[:, np.where(rf.classes_ == 1)[0][0]]
#
#     print('test score on whole dataset is ', str(score(y_train, y_prob)))
#     print('混淆矩阵:\n', confusion_matrix(y_train, y_pred))
#
# with timer('Mode test'):
#     test_prob_final = np.zeros((len(feats_test),))
#     for i in range(classify_num):
#         test_prob = classify_models[i].predict_proba(feats_test)[:, 1]
#         test_prob_final += (test_prob * 0.1)
#
# with timer('Write Result'):
#     result = pd.DataFrame()
#     result['id'] = test['id']
#     result['score'] = test_prob_final
#     result.to_csv('../submission/submission_180512_v2.csv', index=False)
#


num_train=20

train_data=[]
for i in range(num_train):
    train_data.append(train.iloc[i*50000:i*50000+50000])



col1 = list(train.columns)[2:]
lr=[]
lg=[]
xgbmodel=[]
res=[]
grad=[]
rf=[]
et=[]
for i in range(num_train):
    # rf.append(RandomForestClassifier(n_estimators=20,max_depth=10,n_jobs=-1))
    # et.append(ExtraTreesClassifier(n_estimators=20,max_depth=10,n_jobs=-1))
    lg.append(lgb.LGBMClassifier(max_depth = 8, n_estimators = 100,num_leaves=31))
    xgbmodel.append(xgb.XGBClassifier(max_depth = 8, n_estimators = 30,num_leaves=31))
    grad.append(GradientBoostingClassifier(n_estimators=15,max_depth=10))
    # lr.append(LogisticRegression(C=1.0))

clf = []
for i in range(num_train):
      # clf.append( StackingClassifier(classifiers=[lg[i], xgbmodel[i]],use_probas=True,average_probas=False,meta_classifier=lr[i]))
      clf.append(VotingClassifier(estimators = [('lg',lg[i]), ('xgb',xgbmodel[i]),('grad',grad[i])], voting = 'soft'))
      clf[i].fit(X,y)
      res.append(clf[i].predict_proba(feats_test)[:,-1])

def score(y,pred):
    fpr,tpr,thresholds=roc_curve(y,pred,pos_label=1)
    score1 = 0.4 * tpr[np.where(fpr >= 0.001)[0][0]] + 0.3 * tpr[np.where(fpr >= 0.005)[0][0]] + 0.3 * tpr[np.where(fpr >= 0.01)[0][0]]
    return 'selfeval', score1, True

# print(score(train.label,res))


end=pd.DataFrame({'id':test['id'],'score':sum(res)/20})
print(end.describe())
end.to_csv('output/lgb_xgb_grad_preds_-1_1_drop161_20_fillmode_pca.csv',index=False)


























