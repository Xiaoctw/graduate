import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
'''
在这里预处理要做的就是去除多余的feat这一种工作
'''


def Data_preprocessing(x: pd.DataFrame, feat_Remv):
    # x = x.drop_duplicates()
    x = x.drop(feat_Remv, axis=1)
    return x


def Feature_engineering(X, y, train_num, skb_samples=10000):
    '''

    :param X: 特征向量
    :param y: 标签
    :param train_num:
    :param skb_samples:
    :return:
    '''

    def get_mDay(s):
        s = s.split(' ')[0]
        s = s.split('-')[1]
        return int(s)

    def get_wDay(s):
        s = s.split(' ')[0]
        month = int(s.split('-')[1])
        day = int(s.split('-')[2])
        l = ['31', '28', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']
        for i in range(month - 1):
            day += int(l[i])
        return day % 7

    def get_hday(s):
        s = s.split(' ')[1]
        return int(s.split(':')[0])

    cate_list = ['D1', 'E4', 'E8',
                 'E11', 'E15',
                 'E18', 'E13', 'E9',
                 'E21', 'E22', 'E25', 'D2', 'E3', 'E12', 'E16'
                 ]
    nume_list = ['A1', 'A3', 'B1', 'B3', 'A2', 'B2',
                 'C2', 'E1', 'E2', 'E6',
                 'E5', 'E7', 'E10',
                 'E14', 'E19', 'E20', 'E17', 'C1', 'C3', 'E24', 'E26',
                 'E23',
                 'E27', 'E28', 'E29'
                 ]
    x = (X.copy())
    # x=Data_preprocessing(x,nume_list)
    # nume:A2
    # x = Data_preprocessing(x, 'A2')
    # nume:B2
    # x = Data_preprocessing(x, 'B2')
    x['day_week'] = x['date'].apply(lambda x: get_wDay(x))
    x['day_month'] = x['date'].apply(lambda x: get_mDay(x) // 10)
    x['hour'] = x['date'].apply(lambda x: get_hday(x) // 6)
    all_list = cate_list + nume_list + ['day_week', 'day_month', 'hour']
    del x['date']
    print('构造日期完成')
    # CATEGORICAL_FEATURES.extend(['day_week','day_month','hour'])
    for feature in cate_list + ['day_week', 'day_month', 'hour']:
        label_oncoder = preprocessing.LabelEncoder()
        x[feature] = label_oncoder.fit_transform(x[feature])
    print('类别标签数值化完成')
    standard_scaler = preprocessing.StandardScaler()
    x[nume_list] = standard_scaler.fit(x[nume_list]).transform(x[nume_list])
    features = np.array(x.columns)
    print('归一化连续数值完成')
    x = np.array(x)
    train_x, test_x = x[:train_num, :], x[train_num:, :]
    # print(train_x.shape)
    skb = SelectKBest(k=(40 if 40 < x.shape[1] else x.shape[1]))
    skb.fit(train_x[:skb_samples], y[:skb_samples])
    val = skb.get_support()
    # print(features[val])
    # print(val)
    support_index = [index for index in range(skb.get_support().shape[0]) if skb.get_support()[index]]
    train_x, test_x = train_x[:, support_index], test_x[:, support_index]
    print('特征提取完成')
    return train_x, test_x
