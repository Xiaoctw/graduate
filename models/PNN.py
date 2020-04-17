import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as fun
import torch.utils.data as Data


class PNN(nn.Module):
    def __init__(self, field_size, feature_sizes, embedding_size=4, d1_size=10,
                 h_depth=2, is_inner=True):
        super(PNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.field_size = field_size
        self.p_size = self.field_size * (self.field_size - 1) // 2
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.h_depth = h_depth
        # self.deep_layers = deep_layers
        # self.dropout_p_z = dropout_p_z
        # self.dropout_deep = dropout_deep
        self.d1_size = d1_size
        self.is_inner=is_inner
        # 嵌入层
        self.feat_embedd = nn.Embedding(sum(feature_sizes), self.embedding_size)
        self.feat_embedd.weight.data.normal_(0, 1/self.d1_size)
        self.feat_bias = nn.Parameter(torch.randn(1), requires_grad=True)
        self.weight2p=nn.Parameter(torch.randn(self.field_size*self.field_size,self.d1_size)/(self.field_size),requires_grad=True)
        # 这个分别和每一个特征的嵌入向量做点积，得到z向量
        self.weight2z=nn.Parameter(torch.randn(self.field_size*self.embedding_size,self.d1_size)/(self.field_size),requires_grad=True)
        # 开始完成全连接层
        self.lin_1 = nn.Linear(2 * self.d1_size,12)
        self.deep_drop_1 = nn.Dropout(0.2)
        self.batch_norm_1 = nn.BatchNorm1d(12)
        self.lin_2=nn.Linear(12,4)
        self.deep_drop_2=nn.Dropout(0.2)
        self.batch_norm_2 = nn.BatchNorm1d(4)
        self.lin_3=nn.Linear(4,1)
        print('初始化PNN模型完成')

    def forward(self, x):
        num_item = x.shape[0]
        # print('num_item:{}'.format(num_item))
        # print('field_size:{}'.format(self.field_size))
        # print(x.shape[1])
        # 通过嵌入层
        x = self.feat_embedd(x)
        x1=torch.mm(x.view(num_item,-1),self.weight2z)
        x2=torch.bmm(x,x.permute(0,2,1))
        x2=x2.view(num_item,-1)
        x2=torch.mm(x2,self.weight2p)
        x = torch.cat((x1, x2), 1)
      #  print(x.shape)
        x = fun.relu(self.batch_norm_1(self.lin_1(x)))
        x = self.deep_drop_1(x)
        x=fun.relu(self.batch_norm_2(self.lin_2(x)))
        x=self.deep_drop_2(x)
        x=self.lin_3(x)
        x = torch.sigmoid(x)
        #这一行不可以少,否则计算loss会出问题
        x=x.view(num_item)
        return x


def construct_pnn_model(train_x, train_y, field_size, feat_sizes, lr=3e-2, task='regression', num_epoch=40,is_inner=True):
    cri = None
    if task == 'regression':
        cri = nn.MSELoss(reduction='sum')
    else:
        cri = nn.BCELoss(reduction='sum')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_x, test_x, field_size, feat_sizes = find_deepfm_params(x1=train_x, x2=test_x)
    model: nn.Module = PNN(field_size=field_size, feature_sizes=feat_sizes, is_inner=is_inner)
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(lr=lr, params=model.parameters())
    # 注意这里全部都要转化为long形式，因为要嵌入
    train_x, train_y = torch.Tensor(train_x).long(), torch.Tensor(train_y)
    data_set = Data.TensorDataset(train_x, train_y)
    data_loader = Data.DataLoader(dataset=data_set, batch_size=train_x.shape[0] // 4, shuffle=True, )
    total_losses = []
    for epoch in range(num_epoch):
        total_loss = 0
        for step, (batch_x, batch_y) in enumerate(data_loader):
            opt.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = cri(outputs, batch_y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if epoch%4==0:
            print('PNN训练过程,第{}次循环，当前loss为：{}'.format(epoch, total_loss))
        total_losses.append(total_loss)
    plt.plot(total_losses,ls='--',color='r')
    plt.scatter(list(range(len(total_losses))),total_losses,color='b')
    plt.title('O_PNN losses')
    plt.show()
    return model
