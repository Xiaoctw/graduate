import numpy as np
import torch
import torch.nn as nn
import math
import torch.utils.data as Data
import torch.nn.functional as fun
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score


class deFmNu(nn.Module):
    def __init__(self, field_size, feature_sizes, num_field_size,
                 embedding_size=4,
                 h_depth=2, deep_layers=None,
                 dropout_shallow=None, dropout_deep=None):
        super(deFmNu, self).__init__()
        # 默认中间有两个连续层，12个节点和8个节点

        if dropout_shallow is None:
            dropout_shallow = [0.2, 0.2]
        #列表长度加一的关系
        if deep_layers is None:
            deep_layers = [12, 8, 4]
        if dropout_deep is None:
            dropout_deep = [0.2, 0.2, 0.2,0.2]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.deep_layers = deep_layers
        self.h_depth = h_depth
        self.emb_size = embedding_size
        self.feat_sizes = feature_sizes
        self.field_size = field_size
        self.num_field_size = num_field_size
        self.dropout_deep = dropout_deep  # 这个是在deep网络部分使用的dropout系数
        self.dropout_shallow = dropout_shallow  # 这个是在一维特征和组合特征上使用的dropout
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)
        stdv = math.sqrt(0.2 / len(self.feat_sizes))
        print('初始化deep中FM部分')
        self.dropout1 = nn.Dropout(dropout_shallow[0])
        # 这一部分可以看做是LR
        self.fm_first = nn.Embedding(sum(feature_sizes), 1)
        self.fm_first.weight.data.normal_(0, std=stdv)
        # 交叉连接层
        self.fm_second = nn.Embedding(sum(feature_sizes), self.emb_size)
        self.dropout2 = nn.Dropout(dropout_shallow[1])
        self.fm_second.weight.data.normal_(0, std=stdv)
        print('初始化deep中Deep模型')
        # 一个全连接层
        self.lin_1 = nn.Linear(self.field_size * self.emb_size + self.num_field_size, self.deep_layers[0])
        self.deep_drop_0 = nn.Dropout(self.dropout_deep[0])
        self.batch_norm_1 = nn.BatchNorm1d(self.deep_layers[0])
        self.deep_drop_1 = nn.Dropout(self.dropout_deep[1])
        for i, h in enumerate(self.deep_layers[1:],1):
            setattr(self, 'lin_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
            setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(self.deep_layers[i]))
            setattr(self, 'deep_drop_' + str(i + 1), nn.Dropout(self.dropout_deep[i + 1]))
        # self.dropout3 = nn.Dropout(dropout_shallow[2])
        print('初始化Deep模型完成')

    def forward(self, train_x1, train_x2):
        '''

        :param x1: 离散型数据，通过嵌入进入网络
        :param x2: 连续性数据，直接进入网络中，在deep部分和离散数据进行合并，参与运算
        :return:
        '''
        num_item = train_x1.shape[0]
        # shape = x1.shape
        # print(shape)
        x1 = train_x1.view(num_item * self.field_size)
        # print(x1.shape)
        fm_first = self.fm_first(x1)
        fm_first = fm_first.view(train_x1.size(0), -1)
        fm_first = self.dropout1(fm_first)
        fm_sec_emb = self.fm_second(x1).view(train_x1.size(0), self.field_size, -1)  # (20,7,4)
        #  print('fm_sec_emb:{}'.format(fm_sec_emb.shape))
        fm_sum_sec_emb = torch.sum(fm_sec_emb, 1)  # (20,4)
        #  print('fm_sum_Sec_emb{}'.format(fm_sum_sec_emb.shape))
        # (20,4)
        fm_sum_sec_emb_squ = fm_sum_sec_emb * fm_sum_sec_emb  # (x+y)^2
        # (20,7,4)
        fm_sec_emb_squ = fm_sec_emb * fm_sec_emb
        # (20,4)
        fm_sec_emb_squ_sum = torch.sum(fm_sec_emb_squ, 1)  # x^2+y^2
        fm_second = (fm_sum_sec_emb_squ - fm_sec_emb_squ_sum) * 0.5
        # (20,4)
        fm_second = self.dropout2(fm_second)
        deep_emb = fm_sec_emb.reshape(num_item, -1)
        # 将两部分结合
        deep_data = torch.cat((deep_emb, train_x2), dim=1)
        deep_data = self.deep_drop_0(deep_data)
        x_deep = fun.relu(self.batch_norm_1(self.lin_1(deep_data)))
        x_deep = self.deep_drop_1(x_deep)
        for i in range(1, len(self.deep_layers)):
            x_deep = getattr(self, 'lin_' + str(i + 1))(x_deep)
            x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
            x_deep = fun.relu(x_deep)
            x_deep = getattr(self, 'deep_drop_' + str(i + 1))(x_deep)
        # 返回总的结果
        total_sum = torch.sigmoid(torch.sum(fm_first, 1) + torch.sum(fm_second, 1) + torch.sum(x_deep, 1) + self.bias)
        return total_sum


def construct_deFmNu(train_x1, train_x2, train_y, field_size, num_field_size, feat_sizes, embedding_size=4, lr=3e-2,
                     task='regression', num_epoch=40):
    if task == 'regression':  # 回归
        cri = nn.MSELoss(reduction='sum')
    else:  # 二分类
        cri = nn.BCELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_x, test_x, field_size, feat_sizes = find_deepfm_params(x1=train_x, x2=test_x)

    model: nn.Module = deFmNu(field_size=field_size, feature_sizes=feat_sizes, num_field_size=num_field_size
                              , embedding_size=embedding_size)
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    # 注意这里全部都要转化为long形式，因为要嵌入
    train_x1, train_x2, train_y = torch.Tensor(train_x1).long(), torch.Tensor(train_x2), torch.Tensor(train_y)
    # print(train_x1.shape)
    # print(train_x2.shape)
    # data_set = Data.TensorDataset(train_x1, train_y)
    # data_loader = Data.DataLoader(dataset=data_set, batch_size=train_x1.shape[0] // 5, shuffle=True, num_workers=4)
    data_set = cat_num_dataSet(train_x1, train_x2, train_y)
    #  print(data_set)
    data_loader = Data.DataLoader(data_set, batch_size=train_x1.shape[0] // 5, shuffle=True, )
    total_losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for step, sample in enumerate(data_loader):
            opt.zero_grad()
            #  print(sample)
            x1, x2, target = sample['x1'], sample['x2'], sample['target']
            # print(x1.shape)
            # print(x2.shape)
            # print(target.shape)
            # batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(x1, x2)
            loss = cri(outputs, target)
            # if step == 0:
            #     print('预测:{}'.format(outputs[:15]))
            #     print('标准:{}'.format(batch_y[:15]))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch % 4 == 0:
            print('deFmNu模型训练过程，epoch:{}，当前loss为：{:.4f}'.format(epoch, total_loss))
        total_losses.append(total_loss)
    plt.plot(total_losses, ls='--', color='b')
    plt.scatter(list(range(len(total_losses))), total_losses, color='w',marker='o')
    plt.title('deFmNu losses')
    plt.show()
    return model

def update_deFmNu_model(model:nn.Module, train_cate_x,train_nume_x,train_y,batch_cate_x, batch_nume_x, batch_y, lr=3e-2, num_epoch=10, task='regression',alpha=0.4):
    if task == 'regression':  # 回归
        cri = nn.MSELoss(reduction='sum')
    else:  # 二分类
        cri = nn.BCELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    idxes=np.array(range((train_y.shape[0])))
    np.random.shuffle(idxes)
    model.train()
    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    num_train=int(train_y.shape[0]*alpha)
    train_cate_x,train_nume_x,train_y=torch.Tensor(train_cate_x[idxes][:num_train]).long(),torch.Tensor(train_nume_x[idxes][:num_train]),torch.Tensor(train_y[idxes][:num_train])
    batch_cate_x, batch_nume_x, batch_y = torch.Tensor(batch_cate_x).long(), torch.Tensor(batch_nume_x), torch.Tensor(batch_y)
    X1=torch.cat([train_cate_x,batch_cate_x],dim=0)
    X2=torch.cat([train_nume_x,batch_nume_x],dim=0)
    Y=torch.cat([train_y,batch_y])
    data_set = cat_num_dataSet(X1, X2, Y)
    #  print(data_set)
    data_loader = Data.DataLoader(data_set, batch_size=X1.shape[0] // 5, shuffle=True, )
    for epoch in range(num_epoch):
        total_loss=0
        for step,sample in enumerate(data_loader):
            opt.zero_grad()
            x1, x2, target = sample['x1'], sample['x2'], sample['target']
            outputs = model(x1, x2)
            loss = cri(outputs, target)
            loss.backward()
            opt.step()
            total_loss+=loss.item()
        #if epoch*4==0:
        # print('epoch:{},loss:{}'.format(epoch,total_loss))
    return model


def make_scores(test_y,outs):
    scores=[]
    for out in outs:
        scores.append(roc_auc_score(test_y,out))
    scores.sort(reverse=False)
    return scores

def eval_new_deep_model(model: nn.Module, test_x1, test_x2, test_y, task='regression'):
    # if task == 'regression':  # 回归
    #     pass
    # else:  # 二分类
    #     pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_x1, test_x2, test_y = torch.Tensor(test_x1).long(), torch.Tensor(test_x2), torch.Tensor(test_y)
    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(test_x1, test_x2)
        # score = roc_auc_score(test_y, outputs)
        # print('测试集上roc:{}'.format(score))
    return np.array(outputs)


class cat_num_dataSet(Dataset):
    '''
    构建新的数据集合类
    用于传入到模型当中
    '''

    def __init__(self, x1, x2, y1):
        self.x1 = x1
        self.x2 = x2
        self.target = y1

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        sample = {
            'x1': self.x1[idx],
            'x2': self.x2[idx],
            'target': self.target[idx]
        }
        return sample
