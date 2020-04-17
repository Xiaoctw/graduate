import numpy as np
import lightgbm as lgb
import xgboost as xgb
import torch
import math
import torch.nn as nn
import torch.nn.functional as fun
import torch.utils.data as Data
import matplotlib.pyplot as plt
from preprocess.data_preprocess import *


def construct_trees(train_x, train_y, val_x, val_y,
                    max_leaf: int,boosting_type='gbdt',
                    boost_from_average=True, lr1=1,lr2=3e-2, k=5, dim=3, task='regession', num_epoch=20):
    '''
    :param lr2:
    :param num_epoch: 训练过程中使用的迭代次数
    :param dim: 代表嵌入维度
    :return:
    '''
    if task == 'regression':
        objective = 'regression'
        metric = 'mse'
    else:
        # 二分类
        objective = 'binary'
        metric = {'auc'}
    params = {
        'task': 'train',
        #设置提升类型
        'boosting_type': boosting_type,
        #目标函数
        'objective': objective,
        #评估函数
        'metric': metric,
        #叶子节点数目
        'num_leaves': max_leaf,
        'boost_from_average': boost_from_average,
        'feature_fraction':0.9,
        'bagging_fraction':0.8,
        'num_threads': -1,
        'learning_rate': lr1
    }

    lgb_train = lgb.Dataset(train_x, train_y, params=params)
    lgb_val = lgb.Dataset(val_x, val_y, reference=lgb_train)
    gbm = lgb.train(params=params, train_set=lgb_train, early_stopping_rounds=20, valid_sets=lgb_val)
    preds = gbm.predict(train_x, pred_leaf=True).reshape(train_x.shape[0], -1).astype('int')
    num_tree = preds.shape[1]
    leaf_output = np.zeros((num_tree, max_leaf), dtype=np.float)
    num_leafs = []
    for tid in range(num_tree):
        num_leaf = np.max(preds[:, tid]) + 1
        num_leafs.append(num_leaf)
        for lid in range(num_leaf):
            leaf_output[tid][lid] = gbm.get_leaf_output(tid, lid)
    num_group = math.ceil(num_tree / k)  # 一共有多少个组,k代表每组里有多少人
    group_num_tree = []  # 代表每组里面有多少个树
    for i in range(num_group - 1):
        group_num_tree.append(k)
    group_num_tree.append(num_tree - (num_group - 1) * k)
    # 接下来针对每一组进行操作
    models = []
    leaf2pred = np.zeros((train_x.shape[0], num_group * dim))
    for i in range(num_group):
        model, train_leaf = make_train_model(train_x=train_x, num_group=num_group, i_group=i, k=k,lr=lr2,
                                             num_tree=num_tree, num_leafs=num_leafs, dim=dim, preds=preds)
        models.append(model)
        leaf2pred[:, i * dim:i * dim + dim] = train_leaf
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if task == 'regression':
        cri = nn.MSELoss(reduction='sum')
        leaf2pred, train_y = torch.Tensor(leaf2pred), torch.Tensor(train_y)
    else:
        cri = nn.NLLLoss(reduction='sum')
        leaf2pred, train_y = torch.Tensor(leaf2pred), torch.Tensor(train_y).long()
    dataSet = Data.TensorDataset(leaf2pred, train_y)
    train_loader = Data.DataLoader(
        dataset=dataSet,
        batch_size=train_x.shape[0] // 5,
        shuffle=True,
        num_workers=4
    )

    model: nn.Module = Gbdt2nn_model1(num_group * dim, task)
    model = model.to(device)
    #  num_epoch = 1
    opt = torch.optim.Adam(model.parameters(), lr=lr1)
    losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            opt.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = cri(outputs, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        losses.append(total_loss)
        if epoch%5==0:
            print('综合叶子节点网络过程,第{}次循环,当前loss为:{}'.format(epoch, total_loss))
    print('训练完成')
    # 返回训练好的模型
    plt.plot(losses)
    plt.show()
    return models, model


def make_train_model(train_x, num_group, i_group, k, num_tree, num_leafs, dim, preds, num_epoch=40, lr=3e-2):
    '''
    制作单个模型
    :param num_epoch:
    :param train_x: 训练集
    :param train_y: 标签
    :param num_group: 组数
    :param i_group: group的id,构建第i组树的模型
    :param k: 维度
    :param num_tree:
    :param num_leafs:
    :param dim:
    :param preds: 每个样本，在每一棵树上对应的叶子节点
    :param lr:
    :return: 一个已经训练好的深度网络模型，还有train_leaf,代表
    训练集一组数据的嵌入值
    '''
    if i_group != num_group - 1:
        tids = [j for j in range(i_group * k, i_group * k + k)]
    else:
        tids = [j for j in range(i_group * k, num_tree)]
    # dic, cnt = make_dir(tids, num_leafs)
    # 构建出嵌入矩阵
    # k = 1  # 获得了嵌入维度
    # for id in tids:
    #     k = k * num_leafs[id]
    k = 0
    leafs = []
    for id in tids:
        leafs.append(num_leafs[id])
        k += num_leafs[id]
    # num_leafs=leafs
    # 初始化一个矩阵，用作嵌入
    emb_weights = np.random.normal(0, 1, (k, dim))
    onehots = make_onehot(k, preds, tids, num_leafs)
    train_leaf = np.dot(onehots, emb_weights)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model: nn.Module = Gbdt2nn_model(train_x.shape[1], dim)
    model = model.to(device)
    train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_leaf)
    data_set = Data.TensorDataset(train_x, train_y)
    train_loader = Data.DataLoader(dataset=data_set, batch_size=train_x.shape[0] // 5, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 这里是输出是和嵌入的向量进行比对，因此只有mse
    criterion = nn.MSELoss(reduction='sum')
    losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
          #  print('第{}组,第{}次循环，第{}个批次，当前loss为:{}'.format(i_group, epoch + 1, step + 1, loss.item()))
        if epoch%5==0:
            print('第{}组树，第{}次循环，总的loss为{}'.format(i_group, epoch, total_loss))
        losses.append(total_loss)
    plt.plot(losses)
    plt.show()
    return model, train_leaf


def make_onehot(k, preds, tids, num_leafs):
    n = preds.shape[0]
    res = np.zeros((n, k)).astype('int')
    pre = np.zeros((n, 1)).astype('int')
    for id in tids:
        for i in range(n):
            j = (pre[i][0] + preds[i][id])
            res[i][j] = 1
        # res[range(n),pre + preds[id]] = 1
        pre = pre + num_leafs[id]
    return res


class Gbdt2nn_model(nn.Module):
    def __init__(self, i_size, o_size):
        super(Gbdt2nn_model, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(i_size, 32)
        self.batch_norm2 = nn.BatchNorm1d(10)
        self.linear2 = nn.Linear(32, 10)
        self.batch_norm3 = nn.BatchNorm1d(o_size)
        self.linear3 = nn.Linear(10, o_size)

    def forward(self, x):
        x = self.dropout1(x)
        x = fun.relu(self.batch_norm1(self.linear1(x)))
        x = self.dropout2(x)
        x = fun.relu(self.batch_norm2(self.linear2(x)))
        x = fun.relu(self.batch_norm3(self.linear3(x)))
        return x


class Gbdt2nn_model1(nn.Module):
    def __init__(self, i_size, task):
        super(Gbdt2nn_model1, self).__init__()
        self.task = task
        self.linear1 = nn.Linear(i_size, 20)
        self.batch_norm1 = nn.BatchNorm1d(20)
        # 丢弃的比例
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(20, 8)
        self.batch_norm2 = nn.BatchNorm1d(8)
        if task == 'regression':
            self.linear3 = nn.Linear(8, 1)
        else:
            self.batch_norm3 = nn.BatchNorm1d(2)
            self.linear3 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.dropout1(x)
        x = fun.relu(self.batch_norm1(self.linear1(x)))
        x = self.dropout2(x)
        x = fun.relu(self.batch_norm2(self.linear2(x)))
        if self.task == 'regression':
            x = self.linear3(x)
        else:
            x = fun.log_softmax(self.batch_norm3(self.linear3(x)), dim=0)
        return x


# def make_dir(tids, num_leafs):
#     '''
#     返回在每个叶子节点对应的嵌入下标,以及最大的嵌入下表
#     :param tids:
#     :param num_leafs:
#     :return:
#     '''
#     dic = {}
#     cnt = 0
#     for i in tids:
#         num_leaf = num_leafs[i]
#         for j in range(num_leaf):
#             dic[i, j] = cnt
#             cnt += 1
#     return dic, cnt

'''
这个模型在构建过程当中就需要使用到特定数据集
'''

class GBDT2NN(nn.Module):
    def __init__(self, train_x, train_y, val_x, val_y, boosting_type: str,
                 max_leaf=10, boost_from_average=True, task='regression',num_epoch=40,lr1=1,lr2=3e-2,dim=3):
        super(GBDT2NN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dim = dim
        self.fir_models, self.sec_model = construct_trees(train_x, train_y, val_x, val_y, boosting_type=boosting_type,
                                                          max_leaf=max_leaf, boost_from_average=boost_from_average,
                                                          dim=self.dim, task=task,num_epoch=num_epoch,lr1=lr1,lr2=lr2)
        self.num_fir_models = len(self.fir_models)  # 记录下第一种模型的个数

    '''
    这里只在预测时使用，
    因为在构建树时已经进行了模型的训练
    '''

    def forward(self, x):
        shape = x.shape
        pred_leaf = torch.zeros(shape[0], (self.dim * self.num_fir_models), requires_grad=True)
        pred_leaf = pred_leaf.to(self.device)
        pre = 0
        for i in range(1, self.num_fir_models):
            #   print('输入的shape{}'.format(shape))
            pred = self.fir_models[i](x)
            #   print(pred.shape)
            pred_leaf[:, pre:pre + self.dim] = pred
            pre = pre + self.dim
        return self.sec_model(pred_leaf)

    def get_paramaters(self):
        model_parameters = []
        for model in self.fir_models:
            model_parameters.append(model.parameters())
        model_parameters.append(self.sec_model.parameters())
        return model_parameters

def constructGBDT(max_leaf=10,boosting_type='gbdt',file_name='flight',
                    boost_from_average=True, lr1=1, task='regression'):
    if task == 'regression':
        objective = 'regression'
        metric = 'mse'
    else:
        # 二分类
        objective = 'binary'
        metric = {'auc'}
    params = {
        'task': 'train',
        #设置提升类型
        'boosting_type': boosting_type,
        #目标函数
        'objective': objective,
        #评估函数
        'metric': metric,
        #叶子节点数目
        'num_leaves': max_leaf,
        'boost_from_average': boost_from_average,
        'feature_fraction':0.9,
        'bagging_fraction':0.8,
        'num_threads': -1,
        'learning_rate': lr1
    }
    train_x, train_cate_x, train_nume_x, test_x, test_cate_x, test_nume_x, train_y, test_y = pre_data(file_name)
    n = train_x.shape[0]
    idxes = np.array(range(n))
    np.random.shuffle(idxes)
    idxes1, idxes2 = idxes[:(n // 10) * 9], idxes[(n // 10) * 9:]
    train_x, val_x = train_x[idxes1], train_x[idxes2]
    train_y, val_y = train_y[idxes1], train_y[idxes2]
    lgb_train = lgb.Dataset(train_x, train_y, params=params)
    lgb_val = lgb.Dataset(val_x, val_y, reference=lgb_train)
    gbm = lgb.train(params=params, train_set=lgb_train, early_stopping_rounds=20, valid_sets=lgb_val)
    print(gbm.predict(data=test_x))
    return gbm
