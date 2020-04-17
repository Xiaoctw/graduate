import numpy as np
import torch
from models.DeepFM import *
from models.Gbdt_Dense import *
from models.newDeepModelPart import *
from sklearn.metrics import mean_squared_error
from preprocess.data_preprocess import *

file_name = 'Chicago'
train_num_epoch = 40
dim = 3
deep_lr = 3e-2
# 这个alpha代表着deep模型所占比例
alpha = 0.5
num_tree_a_group = 4
num_update = 10

train_x, train_cate_x, train_nume_x, test_x, test_cate_x, test_nume_x, train_y, test_y = pre_data(file_name)
# 构建deep模型
train_cate_x, test_cate_x, field_size, feat_sizes = find_deep_params(train_cate_x, test_cate_x)
deFmNu_model = construct_deFmNu(train_x1=train_cate_x, train_x2=train_nume_x, train_y=train_y,
                                task='binary',
                                num_epoch=train_num_epoch, lr=deep_lr, field_size=field_size,
                                num_field_size=train_nume_x.shape[1],
                                feat_sizes=feat_sizes, embedding_size=dim)
deep_out = eval_new_deep_model(deFmNu_model, test_cate_x, test_nume_x, test_y, task='binary')
# 树形分类器转deep模型
gbm, gbdt_dense_model = construct_gbdt_dense(train_x=train_nume_x, train_y=train_y, test_x=test_nume_x, test_y=test_y,
                                             lr=deep_lr,
                                             task='binary', num_epoch=train_num_epoch,
                                             num_tree_a_group=num_tree_a_group)
gbdt2deep_out = gbdt_dense_model.predict(torch.Tensor(test_nume_x))
out = deep_out * alpha + gbdt2deep_out * (1 - alpha)
print('第一轮训练结束，当前roc值为：{}'.format(roc_auc_score(np.array(test_y), np.array(out))))
# 开始更新模型
num_test = test_y.shape[0]
batch_size = num_test // num_update
gbdt_dense_roc_es = []
deep_roc_es = []
total_roc_es = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range(num_update):
    beg = i * batch_size
    end = min((i + 1) * batch_size, num_test)
    batch_x = test_nume_x[beg:end]
    tensor_x = torch.Tensor(test_nume_x).to(device)
    #这个才是gbdt对应的预测结果
    out1 = gbm.predict(test_nume_x)
    roc_val1 = roc_auc_score(test_y, out1)
    gbdt_dense_roc_es.append(roc_val1)
    print('经过{}次更新，当前gbdt_dense的roc值为:{}'.format(i, roc_val1))
    batch_tem_y = predict_gbdt_batch(gbm, batch_x, num_tree_a_group)
    train_tem_y = predict_gbdt_batch(gbm, train_nume_x, num_tree_a_group)
    # 这个tem_y和神经网络输出一个格式
    gbdt_dense_model.update_model(train_nume_x, train_tem_y, batch_x, batch_tem_y)
    batch_cate_x = test_cate_x[beg:end]
    batch_nume_x = test_nume_x[beg:end]
    batch_y = test_y[beg:end]
    update_deFmNu_model(deFmNu_model, train_cate_x=train_cate_x, train_nume_x=train_nume_x, train_y=train_y,
                        batch_cate_x=batch_cate_x, batch_nume_x=batch_nume_x, batch_y=batch_y)
    out2 = eval_new_deep_model(deFmNu_model, test_cate_x, test_nume_x, test_y, task='binary')
    roc_val2 = roc_auc_score(test_y, out2)
    deep_roc_es.append(roc_val2)
    print('经过{}次更新，当前deFmNu的roc值为:{}'.format(i, roc_val2))
    out = alpha * out1 + (1 - alpha) * out2
    print('经过{}次更新，当前deFmNu的roc值为:{}'.format(i, roc_auc_score(test_y, out)))
    total_roc_es.append(roc_auc_score(test_y, out))
    if end == num_test:
        break

plt.plot(gbdt_dense_roc_es, color='b', lw=2, label='gbdt')
plt.scatter(list(range(len(gbdt_dense_roc_es))), gbdt_dense_roc_es, color='g', marker='^', s=40)
plt.plot(deep_roc_es, color='r', lw=2, label='deFmNu')
plt.scatter(list(range(len(deep_roc_es))), deep_roc_es, color='g', marker='<', s=40)
plt.plot(total_roc_es, label='ComDeep', color='y', lw=2)
plt.scatter(list(range(len(total_roc_es))), total_roc_es, color='g', marker='>', s=40)
plt.title(file_name)
plt.legend()
plt.show()
