
from models.Gbdt_Dense import *
import math

def main():
    task='binary'
    learning_rate=1
    file_name='Chicago'
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
        'learning_rate': learning_rate
    }
    train_x, train_cate_x, train_nume_x, test_x, test_cate_x, test_nume_x, train_y, test_y = pre_data(file_name)
    n = train_x.shape[0]
    idxes = np.array(range(n))
    np.random.shuffle(idxes)
    print('使用GBDT模型')
    lgb_train = lgb.Dataset(train_x, train_y, params=params)
    lgb_val = lgb.Dataset(test_x, test_y, reference=lgb_train)
    gbm = lgb.train(params=params, train_set=lgb_train, early_stopping_rounds=20, valid_sets=lgb_val)
    pred_test=(gbm.predict(data=test_x))
    preds = gbm.predict(test_x, pred_leaf=True).reshape(test_x.shape[0], -1).astype('int')
    score1=roc_auc_score(test_y,pred_test)
    print('gbdt:score1:{}'.format(score1))
    num_tree_a_group=3
    gbm, model = construct_gbdt_dense(train_x=train_nume_x, train_y=train_y, test_x=test_nume_x, test_y=test_y,
                                      lr=3e-2,
                                      task='binary', num_epoch=40, num_tree_a_group=num_tree_a_group)
    num_test = test_y.shape[0]
    num_update = 20
    batch_size = num_test // num_update
    roc_es = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out1=model.predict(torch.Tensor(test_nume_x).to(device))
    model_roc=roc_auc_score(test_y,out1)
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
        train_y = predict_gbdt_batch(gbm, train_nume_x, num_tree_a_group)
        # 这个tem_y和神经网络输出一个格式
        model.update_model(train_nume_x, train_y, batch_x, batch_y)
    _len=len(roc_es)
    x_label = [i % num_update + 1 for i in range(_len)]
    plt.plot(x_label, roc_es, color='g', label='deep_model', lw=2, ls='-')
    plt.scatter(x_label, roc_es, color='m', marker='.')
    plt.plot(x_label, [model_roc]*_len, color='b', label='gbdt', lw=2, ls='--')
    plt.legend()
    plt.title('deep_model approach gbdt retrain')
    plt.show()



if __name__ == '__main__':
    main()
