import lightgbm as lgb
import torch
import seaborn as sns
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import torch.utils.data as Data
import math
import matplotlib.pyplot as plt
from preprocess.data_preprocess import *


class dense_model1(nn.Module):
    def __init__(self, i_size, deep_layers=None, dropouts=None):
        super(dense_model1, self).__init__()
        if deep_layers is None:
            self.deep_layers = [i_size, 8, 2]
        else:
            self.deep_layers = deep_layers
        assert self.deep_layers[0] == i_size
        if dropouts is None:
            self.dropouts = [0.05, 0.05]
        else:
            self.dropouts = dropouts
        for i in range(1, len(self.deep_layers)):
            setattr(self, 'linear_' + str(i), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(self.dropouts[i - 1]))
            setattr(self, 'batch_norm_' + str(i), nn.BatchNorm1d(self.deep_layers[i]))
        self.last_linear = nn.Linear(self.deep_layers[-1], 1)

    def forward(self, x):
        for i in range(1, len(self.deep_layers)):
            x = getattr(self, 'dropout_' + str(i))(x)
            x = torch.relu(getattr(self, 'linear_' + str(i))(x))
        #    x = getattr(self, 'batch_norm_' + str(i))(x)
        return self.last_linear(x)


class gbdt_dense(nn.Module):
    def __init__(self, i_size, num_group, device, task='regression'):
        super(gbdt_dense, self).__init__()
        self.device = device
        self.i_size = i_size
        self.task = task
        self.num_group = num_group
        for i in range(1, num_group + 1):
            setattr(self, 'dense_' + str(i), dense_model1(i_size))

    def forward(self, x):
        # list1 = []
        # for i in range(1, self.num_group + 1):
        #     list1.append(getattr(self, 'dense_' + str(i))(x))
        list1 = [getattr(self, 'dense_' + str(i))(x) for i in range(1, self.num_group + 1)]
        x_cat = torch.cat(list1, 1)
        return x_cat

    def predict(self, x):
        self.eval()
        if type(x) != torch.Tensor:
            x = torch.Tensor(x).float().to(device=self.device)
        with torch.no_grad():
            outputs = self.forward(x)
            preds = torch.sum(outputs, dim=1)
        if self.task != 'regression':
            preds = torch.sigmoid(preds)
        self.train()
        return np.array(preds)

    def update_model(self, train_x, train_y, batch_x, batch_y, test_num_epoch=10, lr1=3e-2, alpha=0.5):
        '''
        利用训练集和新的测试集对模型进行更新
        :param train_x: 训练集
        :param train_y: 训练集标签
        :param batch_x: 新添的数据集
        :param batch_y:
        :param test_num_epoch:
        :param alpha: 训练集中重新训练的比例
        :param lr1:
        :return:
        '''
        cri = nn.MSELoss(reduction='sum')
        device = self.device
        idxes = np.array(range(train_x.shape[0]))
        np.random.shuffle(idxes)
        num_train = int(train_x.shape[0] * alpha)
        train_x = torch.Tensor(train_x[idxes][:num_train])
        train_y = torch.Tensor(train_y[idxes][:num_train])
        batch_x = torch.Tensor(batch_x)
        batch_y = torch.Tensor(batch_y)
        train_x = torch.cat([train_x, batch_x], dim=0)
        train_y = torch.cat([train_y, batch_y])
        data_set = Data.TensorDataset(train_x, train_y)
        data_loader = Data.DataLoader(dataset=data_set, batch_size=train_x.shape[0] // 5, shuffle=True)
        opt = torch.optim.Adam(lr=lr1, params=self.parameters())
        for epoch in range(test_num_epoch):
            # outputs = self.forward(batch_x)
            # loss = cri(outputs, batch_y)
            total_loss = 0
            for step, (batch_x, batch_y) in enumerate(data_loader):
                opt.zero_grad()
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = self.forward(batch_x)
                loss = cri(outputs, batch_y)
                loss.backward()
                total_loss += loss.item()
                opt.step()
            if epoch % 5 == 0:
                print('训练deep模型,第{}轮,当前loss为:{}'.format(epoch, total_loss))
        return


def construct_gbdt_dense(train_x, train_y, test_x, test_y, lr=3e-2, num_epoch=40, task='binary',
                         num_tree_a_group=4):
    if task == 'regression':
        objective = 'regression'
        metric = 'mse'
    else:
        # 二分类
        objective = 'binary'
        metric = {'auc'}
    params = {
        'task': 'train',
        # 设置提升类型
        'boosting_type': 'gbdt',
        # 目标函数
        'objective': objective,
        # 评估函数
        'metric': metric,
        # 叶子节点数目
        'num_leaves': 10,
        'boost_from_average': True,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'num_threads': -1,
        'learning_rate': 1
    }
    # 首先进行的是GBDT预测
    lgb_train = lgb.Dataset(train_x, train_y, params=params)
    lgb_val = lgb.Dataset(test_x, test_y, reference=lgb_train)
    Y = train_y
    gbm = lgb.train(params=params, train_set=lgb_train, early_stopping_rounds=20, valid_sets=lgb_val)
    pred_leaf = gbm.predict(train_x, pred_leaf=True).reshape(train_x.shape[0], -1).astype('int')
    pred_train = (gbm.predict(data=train_x))
    gbdt_roc = roc_auc_score(Y, pred_train)
    num_item, num_tree = pred_leaf.shape
    num_group = math.ceil(num_tree / num_tree_a_group)
    print('一共有{}组树'.format(num_group))
    temp_y = np.zeros((num_item, num_group))
    # for i in range(num_item):
    #     val = 0
    #     for t in range(1,num_tree+1):
    #         l = pred_leaf[i][t-1]
    #         val += gbm.get_leaf_output(t-1, l)
    #         if ((t) % num_tree_a_group == 0) or t == num_tree:
    #             temp_y[i][math.ceil((t) / num_tree_a_group) - 1] = val
    #             #print(val)
    #             val = 0
    for i in range(num_item):
        val = 0
        for t in range(1,num_tree+1):
            l = pred_leaf[i][t-1]
            val += gbm.get_leaf_output(t-1, l)
            if (t > 0 and t % num_tree_a_group == 0) or t == num_tree:
                temp_y[i][math.ceil(t / num_tree_a_group) - 1] = val
                val = 0
    train_x, train_y = torch.Tensor(train_x).float(), torch.Tensor(temp_y).float()
    cri = nn.MSELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model: nn.Module = gbdt_dense(train_x.size()[1], num_group, device, task='binary')
    model = model.to(device)
    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    data_set = Data.TensorDataset(train_x, train_y)
    data_loader = Data.DataLoader(dataset=data_set, batch_size=256, shuffle=True, )
    total_losses = []
    roc_auces = []
    for epoch in range(num_epoch):
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(data_loader):
            opt.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = torch.abs((outputs - batch_y)).sum()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                pred_train = model.predict(train_x)
                # print('pred:{}'.format(pred_test[:20]))
                # print('label:{}'.format(test_y[:20]))
                roc_score = roc_auc_score(np.array(Y), np.array(pred_train))
                # print("前几个test_y:{}".format(test_y[:20]))
                # print("pred_text前几个:{}".format(pred_test[:20]))
                print('epoch:{},roc_auc:{}'.format(epoch, roc_score))
                roc_auces.append(roc_score)
            model.train()
        if epoch%3==0:
              print("epoch:{},loss:{}".format(epoch, total_loss))
        total_losses.append(total_loss)
    plt.plot(roc_auces, label='deep_model_roc', color='g', lw=2, ls=':')
    plt.scatter(list(range(1, len(roc_auces) + 1)), roc_auces, color='y')
    plt.plot([gbdt_roc] * len(roc_auces), label='gbdt_roc', color='b', lw=2, ls='--')
    plt.scatter(list(range(1, len(roc_auces) + 1)), [gbdt_roc] * len(roc_auces), color='m')
    plt.title('roc_auc')
    plt.legend()
    plt.show()
    return gbm, model



def predict_gbdt_batch(gbm, batch_x, num_tree_a_group):
    num_item = batch_x.shape[0]
    pred_leaf = gbm.predict(batch_x, pred_leaf=True).reshape(batch_x.shape[0], -1).astype('int')
    num_tree = pred_leaf.shape[1]
    num_group = math.ceil(num_tree / num_tree_a_group)
    batch_y = np.zeros((num_item, num_group))
    # for i in range(num_item):
    #     val = 0
    #     for t in range(1,num_tree+1):
    #         l = pred_leaf[i][t-1]
    #         val += gbm.get_leaf_output(t-1, l)
    #         if ((t) % num_tree_a_group == 0) or t == num_tree:
    #             temp_y[i][math.ceil((t) / num_tree_a_group) - 1] = val
    #             #print(val)
    #             val = 0
    for i in range(num_item):
        val = 0
        for t in range(1,num_tree+1):
            l = pred_leaf[i][t-1]
            val += gbm.get_leaf_output(t-1, l)
            if ( t % num_tree_a_group == 0) or t == num_tree:
                batch_y[i][math.ceil(t / num_tree_a_group) - 1] = val
                val = 0
    return batch_y


if __name__ == '__main__':
    train_x, train_cate_x, train_nume_x, test_x, test_cate_x, test_nume_x, train_y, test_y = pre_data('Chicago')
    # model = gbdt_dense(3, 3)
    num_tree_a_group = 3
    gbm, model = construct_gbdt_dense(train_x=train_nume_x, train_y=train_y, test_x=test_nume_x, test_y=test_y,
                                      lr=3e-2,
                                      task='binary', num_epoch=40, num_tree_a_group=num_tree_a_group)
    num_test = test_y.shape[0]
    num_update = 10
    batch_size = num_test // num_update
    roc_es = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(num_update):
        beg = i * batch_size
        end = min((i + 1) * batch_size, num_test)
        batch_x = test_nume_x[beg:end]
        tensor_x = torch.Tensor(test_nume_x).to(device)
        out1 = model.predict(tensor_x)
        roc_val = roc_auc_score(test_y, out1)
        roc_es.append(roc_val)
        print('经过{}次更新，当前roc值为:{}'.format(i, roc_val))
        if end == num_test:
            break
        batch_y = predict_gbdt_batch(gbm, batch_x, num_tree_a_group)
        train_y=predict_gbdt_batch(gbm,train_nume_x,num_tree_a_group)
        # 这个tem_y和神经网络输出一个格式
        model.update_model(train_nume_x, train_y, batch_x, batch_y)
    # 显示类别
    # roc_es=[10,2,3,4,4.5,7,7,5]
    # gbdt=[3,4,5,4,4.5,7,7,5]
    _len = len(roc_es)
    pred_test = (gbm.predict(data=test_nume_x))
    gbdt_roc = roc_auc_score(test_y, pred_test)
    gbdt = [gbdt_roc] * _len
    x_label = [i % 10 + 1 for i in range(_len)]
    #df = pd.DataFrame({'id': x_label, 'gbdt': gbdt, 'roc': roc_es})
    plt.plot(x_label, roc_es, color='g', label='deep_model', lw=2, ls='-')
    plt.scatter(x_label, roc_es, color='m', marker='.')
    plt.plot(x_label, gbdt, color='b', label='gbdt', lw=2, ls='--')
    plt.legend()
    plt.title('deFmNu approach gbdt retrain')
    plt.show()
    # sns.relplot('id','roc',data=df,kind='line')
    # plt.title('id and roc')
#   plt.show()
