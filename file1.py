import lightgbm as lgb
import numpy as np

objective = "regression"
metric = "mse"
boost_from_average = True
maxleaf = 7
numtrees = 20
lr = 1
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': objective,
    'metric': metric,
    'num_leaves': maxleaf,
    # 'min_data': 40,
    'boost_from_average': boost_from_average,
    # 'num_threads': 6,
    # 'feature_fraction': 0.8,
    # 'bagging_freq': 3,
    # 'bagging_fraction': 0.9,
    'learning_rate': lr,
}
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
print(y_train.shape)
lgb_train = lgb.Dataset(X_train, y_train, params=params)
lgb_val = lgb.Dataset(X_test, y_test, reference=lgb_train)
gbm = lgb.train(params=params,
                train_set=lgb_train,
                num_boost_round=numtrees,
                early_stopping_rounds=20,
                valid_sets=lgb_val)
preds = gbm.predict(X_test, pred_leaf=True).reshape(X_test.shape[0], -1)
print(preds.shape)  #
num_trees = preds.shape[1]  # 树的个数
print(preds[:5, ])
leaf_outputs = np.zeros((num_trees, maxleaf), dtype=np.int)
for tid in range(num_trees):
    num_leaf = np.max(preds[:, tid]) + 1
    # print(num_leaf)
    for lid in range(num_leaf):
        leaf_outputs[tid][lid] = gbm.get_leaf_output(tid, lid)
for i in range(10):
    val1 = 0
    for t in range(num_trees):
        l = preds[i][t]
        val1 += gbm.get_leaf_output(t, l)
    print(val1)
# print(leaf_outputs)
# print(leaf_outputs[:10])
print(gbm.predict(X_test)[:10])
print(y_train[:10])
