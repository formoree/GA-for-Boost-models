import pandas as pd
import numpy as np
import math
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
import random
from xgboost import XGBRegressor
from sklearn.model_selection import KFold#这个函数只能用于

paramgrid = {"learning_rate": [i*0.01 for i in range(1,21)],
             "n_estimators" : range(100,1100,100),
             "max_depth" : range(3,11),
             "min_child_weight":range(1,9)}

random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=XGBRegressor(),
                                   params=paramgrid, #超参数搜索空间
                                   scoring="r2", #accuracy的标准
                                   cv=KFold(n_splits=5), #交叉验证4折
                                   verbose=1,
                                   population_size=50, #整个种群的染色体数目为50个超参数组合
                                   gene_mutation_prob=0.10, #我们的“孩子”超参数组合中每次会大概选择出10%的超参数进行随机取值
                                   gene_crossover_prob=0.5, #我们会选择每一条“父母”染色体（超参数组合）中的50%的基因（超参数）进行相互交叉
                                   tournament_size=3, #每次从上一代中选择出适应度最好的3个超参数组合直接进行 “复制”
                                   generations_number=5,
                                   n_jobs=1)
cv.fit(X, Y)

