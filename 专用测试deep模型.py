import argparse
import os
import lightgbm as lgb
from models.DeepFM import *
from models.Gbdt2nn import *
from preprocess.helper import *
from models.PNN import *
from preprocess.data_preprocess import *
from sklearn.linear_model import LogisticRegression
#from models.DeepGBM import *
from models.newDeepModelPart import *
from models.PNN import *
from sklearn.metrics import mean_squared_error
train_num_epoch = 40
dim = 3
deep_lr = 5e-2
tree_lr1 = 1
tree_lr2 = 3e-2
file_name = 'Chicago'


def main():
    train_x, train_cate_x, train_nume_x, test_x, test_cate_x, test_nume_x, train_y, test_y = pre_data(file_name)
    train_cate_x, test_cate_x, field_size, feat_sizes = find_deep_params(train_cate_x, test_cate_x)
    pnn_model = construct_pnn_model(train_x=train_cate_x, train_y=train_y, task='regression',
                                    num_epoch=train_num_epoch, lr=deep_lr,
                                    field_size=field_size,
                                    feat_sizes=feat_sizes, is_inner=True)
    deepfm_model = construct_deepfm_model(train_x=train_cate_x, train_y=train_y, task='binary',
                                          num_epoch=train_num_epoch, lr=deep_lr,
                                          field_size=field_size,
                                          feat_sizes=feat_sizes)
    deep_model2 = construct_deFmNu(train_x1=train_cate_x, train_x2=train_nume_x, train_y=train_y,
                                   task='binary',
                                   num_epoch=train_num_epoch, lr=deep_lr, field_size=field_size,
                                   num_field_size=train_nume_x.shape[1],
                                   feat_sizes=feat_sizes, embedding_size=4)

    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    out = lr.predict_proba(test_x)[:, 1]
    task = 'binary'
    # if task == 'regression':
    #     objective = 'regression'
    #     metric = 'mse'
    # else:
    #     # 二分类
    #     objective = 'binary'
    #     metric = {'auc'}
    # params = {
    #     'task': 'train',
    #     # 设置提升类型
    #     'boosting_type': 'gbdt',
    #     # 目标函数
    #     'objective': objective,
    #     # 评估函数
    #     'metric': metric,
    #     # 叶子节点数目
    #     'num_leaves': 10,
    #     'boost_from_average': True,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'num_threads': -1,
    #     'learning_rate': 0.01
    # }
    # lgb_train = lgb.Dataset(train_x, train_y, params=params)
    # lgb_val = lgb.Dataset(test_x, test_y, reference=lgb_train)
    # gbm = lgb.train(params=params, train_set=lgb_train, early_stopping_rounds=20, valid_sets=lgb_val)
    # pred_text = (gbm.predict(data=test_x))
   # score = roc_auc_score(test_y, pred_text)
    #out1 = eval_deep_model(pnn_model, test_cate_x, test_y, task='binary')
    out1=eval_deep_model(pnn_model,test_cate_x,test_y,task='binary')
    out2 = eval_deep_model(deepfm_model, test_cate_x, test_y, task='binary')
    out3 = eval_new_deep_model(deep_model2, test_cate_x, test_nume_x, test_y, task='binary')
    if task=='binary':
        print('lr auc:{}'.format(roc_auc_score(test_y, out)))
        print('PNN auc:{}'.format(roc_auc_score(test_y, out1)))
        print('deepFM auc:{}'.format(roc_auc_score(test_y, out2)))
        print('new_deep_model auc:{}'.format(roc_auc_score(test_y, out3)))
    #    print('gbdt:score:{}'.format(roc_auc_score(test_y,pred_text)))
    else:
        print('lr mse:{}'.format(mean_squared_error(test_y, out)))
     #   print('PNN mse:{}'.format(mean_squared_error(test_y, out1)))
        print('deepFM mse:{}'.format(mean_squared_error(test_y, out2)))
        print('new_deep_model mse:{}'.format(mean_squared_error(test_y, out3)))
    #    print('gbdt:score:{}'.format(mean_squared_error(test_y, pred_text)))


if __name__ == '__main__':
    main()
