import argparse
import os
import lightgbm as lgb
from models.DeepFM import *
from models.Gbdt2nn import *
from preprocess.helper import *
from models.PNN import *
from preprocess.data_preprocess import *
from sklearn.linear_model import LogisticRegression
from models.newDeepModelPart import *
from models.PNN import *
from sklearn.metrics import mean_squared_error

train_num_epoch = 40
dim = 3
deep_lr = 5e-2
tree_lr1 = 1
tree_lr2 = 3e-2
file_name = 'Chicago'

train_x, train_cate_x, train_nume_x, test_x, test_cate_x, test_nume_x, train_y, test_y = pre_data(file_name)
train_cate_x, test_cate_x, field_size, feat_sizes = find_deep_params(train_cate_x, test_cate_x)
pnn_model = construct_pnn_model(train_x=train_cate_x, train_y=train_y, task='regression',
                                    num_epoch=train_num_epoch, lr=deep_lr,
                                    field_size=field_size,
                                    feat_sizes=feat_sizes, is_inner=True)
print(pnn_model)