import sklearn.datasets
import numpy as np
import random

data = sklearn.datasets.load_digits()
X = data["data"]
y = data["target"]

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import StratifiedKFold #这个函数只能用于
from sklearn.model_selection import StratifiedShuffleSplit
# paramgrid = {"kernel": ["rbf"],
#              "C"     : np.logspace(-9, 9, num=25, base=10),
#              "gamma" : np.logspace(-9, 9, num=25, base=10)}
paramgrid = {"learning_rate": [i*0.01 for i in range(1,21)],
             "n_estimators" : range(100,1100,100),
             "max_depth" : range(3,11)}

random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=GradientBoostingRegressor(),
                                   params=paramgrid,
                                   scoring="accuracy",
                                   # cv=StratifiedKFold(n_splits=4),
                                   cv=StratifiedShuffleSplit(n_splits=4),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=1)
cv.fit(X, y)
print(y)
