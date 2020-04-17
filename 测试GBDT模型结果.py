from models.Gbdt_Dense import *
import math

def main():
    task='binary'
    learning_rate=1
    file_name='adv_predict'
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

if __name__ == '__main__':
    main()