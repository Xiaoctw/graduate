import argparse
import os
import lightgbm as lgb
from models.DeepFM import *
from models.Gbdt2nn import *
from preprocess.helper import *
from models.PNN import *
from preprocess.data_preprocess import *
from sklearn.linear_model import LogisticRegression,LinearRegression

#from models.DeepGBM import *
from models.newDeepModelPart import *
from models.PNN import *
from sklearn.metrics import mean_squared_error
train_num_epoch = 40
dim = 3
deep_lr = 5e-2
tree_lr1 = 1
tree_lr2 = 3e-2
file_name = 'promotion'
task1='binary'
if file_name=='house':
    task1='regression'

def main():
    train_x, train_cate_x, train_nume_x, test_x, test_cate_x, test_nume_x, train_y, test_y = pre_data(file_name)
    train_cate_x, test_cate_x, field_size, feat_sizes = find_deep_params(train_cate_x, test_cate_x)
    print('离散:{}'.format(train_cate_x.shape[1]))
    print('连续:{}'.format(train_nume_x.shape[1]))
    pnn_model = construct_pnn_model(train_x=train_cate_x, train_y=train_y, task=task1,
                                    num_epoch=train_num_epoch, lr=deep_lr,
                                    field_size=field_size,
                                    feat_sizes=feat_sizes, is_inner=True)
    deepfm_model = construct_deepfm_model(train_x=train_cate_x, train_y=train_y, task=task1,
                                          num_epoch=train_num_epoch, lr=deep_lr,
                                          field_size=field_size,
                                          feat_sizes=feat_sizes)
    deep_model2 = construct_deFmNu(train_x1=train_cate_x, train_x2=train_nume_x, train_y=train_y,
                                   task=task1,
                                   num_epoch=train_num_epoch, lr=deep_lr, field_size=field_size,
                                   num_field_size=train_nume_x.shape[1],
                                   feat_sizes=feat_sizes, embedding_size=4)
    if task1=='binary':
        lr = LogisticRegression()
        lr.fit(train_x, train_y)
        out = lr.predict_proba(test_x)[:, 1]
        out1=eval_deep_model(pnn_model,test_cate_x,test_y,task='binary')
        out2 = eval_deep_model(deepfm_model, test_cate_x, test_y, task='binary')
        out3 = eval_new_deep_model(deep_model2, test_cate_x, test_nume_x, test_y, task='binary')
        scores=make_scores(test_y,[out,out1,out2,out3])
        print('lr auc:{}'.format(scores[0]))
        print('PNN auc:{}'.format(scores[1]))
        print('deepFM auc:{}'.format(scores[2]))
        print('new_deep_model auc:{}'.format(scores[3]))
    else:
        lr=LinearRegression()
        lr.fit(train_cate_x,train_y)
        out=lr.predict(test_cate_x)
        out1 = eval_deep_model(pnn_model, test_cate_x, test_y, task=task1)
        out2 = eval_deep_model(deepfm_model, test_cate_x, test_y, task=task1)
        out3 = eval_new_deep_model(deep_model2, test_cate_x, test_nume_x, test_y, task=task1)
        print('lr mse:{}'.format(mean_squared_error(test_y, out)))
        print('PNN mse:{}'.format(mean_squared_error(test_y, out1)))
        print('deepFM mse:{}'.format(mean_squared_error(test_y, out2)))
        print('new_deep_model mse:{}'.format(mean_squared_error(test_y, out3)))
    #    print('gbdt:score:{}'.format(mean_squared_error(test_y, pred_text)))


if __name__ == '__main__':
    main()
