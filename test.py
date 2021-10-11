import math
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

#导入数据并进行数据集划分
data = pd.read_excel('Data.xlsx')
data1 = np.array(data)
X = data1[ : , :5]
Y = data1[ : ,5: ].flatten()
L = []
for i in Y :
    if i >= data.mean()[6]:
        L.append(1)
    else :
        L.append(0)
x_train,x_test,y_train,y_test = train_test_split(X,L,train_size=0.8)

import random
from sklearn.model_selection import StratifiedKFold#这个函数只能用于

paramgrid = {"learning_rate": [i*0.01 for i in range(1,21)],
             "n_estimators" : range(100,1100,100),
             "max_depth" : range(3,11)}

random.seed(1)
from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=GradientBoostingRegressor(),
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

# 用调好的参数训练模型
import numpy as np
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import r2_score

# 使用调好的参数进行训练
gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, min_samples_split=2, learning_rate=0.03)
gbr.fit(x_train, y_train)

y_predict = gbr.predict(x_test)
MSE = mean_squared_error(y_test,y_predict)
RMSE = math.sqrt(MSE)
R2 = r2_score(y_test,y_predict)
print(f'MSE的值为{MSE},RMSE为{RMSE},R2为{R2}')

#绘制特征重要性
import matplotlib.pyplot as plt
importance = gbr.feature_importances_
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