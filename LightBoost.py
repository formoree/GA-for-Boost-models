import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
import random
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold#这个函数只能用于
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import r2_score
data = pd.read_excel('Data.xlsx')
data = np.array(data)
## 乱序
from sklearn.utils import shuffle
data = shuffle(data)

## 划分训练集 验证集
X = data[ : , :5]
Y = data[ : ,5: ].flatten()
L = []
for i in Y :
    if i >= 1:
        L.append(1)
    else :
        L.append(0)




paramgrid = {"learning_rate": [i*0.01 for i in range(1,21)],
             "n_estimators" : range(100,1100,100),
             "max_depth" : range(3,11),
             "min_child_weight":range(1,9)}

random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=LGBMRegressor(objective='regression'),
                                   params=paramgrid, #超参数搜索空间
                                   scoring="roc_auc", #accuracy的标准
                                   cv=StratifiedKFold(n_splits=5), #交叉验证4折
                                   verbose=1,
                                   population_size=50, #整个种群的染色体数目为50个超参数组合
                                   gene_mutation_prob=0.10, #我们的“孩子”超参数组合中每次会大概选择出10%的超参数进行随机取值
                                   gene_crossover_prob=0.5, #我们会选择每一条“父母”染色体（超参数组合）中的50%的基因（超参数）进行相互交叉
                                   tournament_size=3, #每次从上一代中选择出适应度最好的3个超参数组合直接进行 “复制”
                                   generations_number=5,
                                   n_jobs=1)
cv.fit(X, L)

import lightgbm as lgb

lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'max_depth': 3,
    'min_child_weight': ,
    'force_col_wise': 'true'
}

# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100, #'num_iterations ' :100
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

y_predict = gbm.predict(x_test)
MSE = mean_squared_error(y_test,y_predict)
RMSE = math.sqrt(MSE)
R2 = r2_score(y_test,y_predict)
print(f'MSE的值为{MSE},RMSE为{RMSE},R2为{R2}')

import matplotlib.pyplot as plt
fig2 = plt.figure(figsize=(20, 20))
ax = fig2.subplots()
lgb.plot_tree(gbm, tree_index=1, ax=ax)
plt.show()

#绘制特征重要性
import matplotlib.pyplot as plt
importance = gbm.feature_importance()/545
#获取特征标签
Impt_Series = pd.Series(importance, index = data.iloc[:,:5].columns)
print(Impt_Series)


#对特征影响大小进行排列
# Impt_Series.sort_values(ascending = True).plot('barh')
Impt_Series = Impt_Series.sort_values(ascending = True)

print(Impt_Series)
print(list(Impt_Series.index))
Y = list(Impt_Series.index)
# 绘制条形图
plt.figure(figsize=(10,5))
plt.barh(range(len(Y)), # 指定条形图y轴的刻度值
        Impt_Series.values, # 指定条形图x轴的数值
        tick_label = Y, # 指定条形图y轴的刻度标签
        color = 'steelblue', # 指定条形图的填充色
       )

print(Impt_Series.values)
# print()
for y,x in enumerate(Impt_Series.values):
    plt.text(x+0.0001,y,'%s' %round(x,3),va='center')
plt.show()