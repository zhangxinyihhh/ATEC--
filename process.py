# -*- coding:utf-8 -*-
#清理异常值：
import numpy as np
from matplotlib.pyplot import *
import pandas as pd
train= pd.read_csv('data/atec_anti_fraud_train.csv',encoding='utf-8')
# test= pd.read_csv('data/atec_anti_fraud_test_b.csv',encoding='utf-8')

x=train.f1
y=train.label
figure()
subplot(232)
bar(x,y)
show()

# #绘制直方图：
# import matplotlib.pyplot as plt
# x=train.f6
# ax=plt.gca()
# ax.hist(x,bins=40,color='r')
# ax.set_xlabel('f5')
# ax.set_ylabel('train_f1_freq')
# plt.show()

# #绘制散点图：
# import matplotlib.pyplot as plt
# x=train.f5
# y=train.label
# ax=plt.subplot(121)
# plt.scatter(x,y,color='indigo',alpha=0.3,edgecolors='white',label='no correl')
# plt.xlabel('train_f5')
# plt.grid(True)
# plt.legend()
#
# plt.show()










