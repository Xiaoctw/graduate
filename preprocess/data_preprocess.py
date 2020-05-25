from preprocess.helper import *
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
import os

def pre_data(data_name):
    '''
    在这里数据集的名字固定，就是那几个需要的数据集，根据不同数据集进行读取文件和预处理
    :param data_name: 传入数据集的名字
    :return:
    '''
    if data_name == 'adv_predict':
        return pre_adv_predict()
    # elif data_name == 'flight':
    #     return pre_flight()
    elif data_name == 'titanic':
        return pre_titanic()
    elif data_name=='Chicago':
        return pre_Chicago()
    # elif data_name=='moscow':
    #     return pre_moscow()


def pre_adv_predict():

    root1=os.path.dirname(os.path.realpath(__file__))
    TRAIN_DATA_PATH=root1+'/data/adv_train.csv'
    TRAIN_LABEL_PATH=root1+'/data/adv_train_label.csv'
    NUMERICAL_FEATURES = ['A1', 'A3', 'B1', 'B3',
                          'C1', 'C2', 'C3', 'E1', 'E2',
                          'E5', 'E6', 'E7', 'E10',
                          'E14', 'E19', 'E20',
                          'E23', 'E24', 'E26',
                          'E27', 'E28', 'E29'
                          ]
    train_y = pd.read_csv(TRAIN_LABEL_PATH)
    train_x = pd.read_csv(TRAIN_DATA_PATH)
    print('共有特征{}'.format(train_x.shape[1]))
    print('连续特征{}'.format(len(NUMERICAL_FEATURES)))
    nume_x=train_x[NUMERICAL_FEATURES].copy()
    train_x['Label'] = train_y['label']
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
    train_x['day_week'] = train_x['date'].apply(lambda x: get_wDay(x))
    train_x['day_month'] = train_x['date'].apply(lambda x: get_mDay(x) // 10)
    train_x['hour'] = train_x['date'].apply(lambda x: get_hday(x) // 6)
    train_x = train_x.drop('date', axis=1)
    train_x = train_x.drop('day_month', axis=1)
    train_x=train_x.drop('ID',axis=1)
    def cut_A1(val):
        if val <= -4:
            return 0
        elif val >= 4:
            return 1
        return 2
    train_x['A1'] = train_x['A1'].apply(cut_A1)
    train_x = train_x.drop('A2', axis=1)
    def cut_A3(val):
        if val <= 0.74:
            return 0
        return 1
    train_x['A3'] = train_x['A3'].apply(cut_A3)
    def cut_B1(val):
        if val <= -0.75:
            return 0
        elif val <= 0:
            return 1
        return 2
    train_x['B1'] = train_x['B1'].apply(cut_A3)
    train_x=train_x.drop('B2',axis=1)
    def cut_B3(val):
        if val <= -0.5:
            return 0
        elif val <= 0.5:
            return 1
        return 2
    train_x['B3'] = train_x['B3'].apply(cut_B3)
    def cut_C1(val):
        if val <= 0:
            return 0
        elif val <= 1e10:
            return 1
        return 2
    train_x['C1'] = train_x['C1'].apply(cut_C1)
    def cut_C2(val):
        if val <= -6.705654e+18:
            return 0
        elif val <= -6.705654e+18:
            return 1
        return 2
    train_x['C2'] = train_x['C2'].apply(cut_C2)
    def cut_C3(val):
        if val <= -4.567091e+18:
            return 0
        elif val <= 4.627437e+18:
            return 1
        return 2
    train_x['C3'] = train_x['C3'].apply(cut_C3)
    train_x=train_x.drop('E2',axis=1)
    def cut_E1(val):
        if val <= 5:
            return 0
        return 1
    train_x['E1'] = train_x['E1'].apply(cut_E1)
    train_x['E3'].describe()
    def cut_E3(val):
        if val <= 0:
            return 0
        return 1
    train_x['E3'] = train_x['E3'].apply(cut_E3)
    def cut_E5(val):
        if val <= -1.17:
            return 0
        elif val <= 0.85:
            return 1
        return 2
    train_x['E5'] = train_x['E5'].apply(cut_E5)
    def cut_E6(val):
        if val <= 1:
            return 0
        elif val <= 8:
            return 1
        return 2
    train_x['E6'] = train_x['E6'].apply(cut_E6)
    train_x['E7'].describe()
    def cut_E7(val):
        if val <= 0.8:
            return 0
        return 1
    train_x['E7'] = train_x['E7'].apply(cut_E7)
    def cut_E9(val):
        if val <= -2:
            return 0
        elif val > 0:
            return 1
        return 2
    train_x['E9'] = train_x['E9'].apply(cut_E9)
    def cut_E10(val):
        if val <= 0:
            return 0
        return 1
    train_x['E10'] = train_x['E10'].apply(cut_E10)
    def cut_E12(val):
        if val == 8:
            return 0
        return 1
    train_x['E12'] = train_x['E12'].apply(cut_E12)
    def cut_E13(val):
        if -4.08 < val < -4.06:
            return 0
        elif val < -4.27:
            return 1
        return 2
    train_x['E13'] = train_x['E13'].apply(cut_E13)
    def cut_E14(val):
        if val <= 15:
            return 0
        elif val <= 163:
            return 1
        return 2
    train_x['E14'] = train_x['E14'].apply(cut_E14)
    def cut_E15(val):
        if val == 9 or val == 3:
            return 0
        if val == 0 or val == 7 or val == 1:
            return 1
        return 2
    train_x['E15'] = train_x['E15'].apply(cut_E15)
    def cut_E16(val):
        if -3.559 <= val <= -3.558242 or -3.3437 <= val == -3.3436:
            return 0
        elif val > 0:
            return 1
        return 2
    train_x['E16'] = train_x['E16'].apply(cut_E16)
    def cut_E17(val):
        if val <= -1:
            return 0
        return 1
    train_x['E17'] = train_x['E17'].apply(cut_E17)
    def cut_E18(val):
        if val == 7 or val == 1:
            return 0
        elif val == 5 or val == 6 or val == 8:
            return 1
        return 2
    train_x['E18'] = train_x['E18'].apply(cut_E18)
    def cut_E19(val):
        if val <= -1.13:
            return 0
        return 1
    train_x['E19'] = train_x['E19'].apply(cut_E19)
    def cut_E20(val):
        if val < 1:
            return 0
        elif val <= 7:
            return 1
        return 2
    train_x['E20'] = train_x['E20'].apply(cut_E20)
    def cut_E21(val):
        if -3.7238 <= val <= -3.7237:
            return 0
        elif val <= -3.98:
            return 1
        return 2
    train_x['E21'] = train_x['E21'].apply(cut_E21)
    def cut_E23(val):
        if val <= 17:
            return 0
        elif val <= 20:
            return 1
        else:
            return 2
    train_x['E23'] = train_x['E23'].apply(cut_E23)
    def cut_E24(val):
        if val <= 6:
            return 0
        elif val <= 14:
            return 1
        else:
            return 2
    train_x['E24'] = train_x['E24'].apply(cut_E24)
    def cut_E25(val):
        if val == 7:
            return 0
        elif val == 4 or val == 5 or val == 6:
            return 1
        else:
            return 2
    train_x['E25'] = train_x['E25'].apply(cut_E25)
    def cut_E26(val):
        if val <= 4:
            return 0
        elif val <= 10:
            return 1
        else:
            return 2
    train_x['E26'] = train_x['E26'].apply(cut_E26)
    def cut_E27(val):
        if val <= 8:
            return 0
        else:
            return 2
    train_x['E27'] = train_x['E27'].apply(cut_E27)
    def cut_E28(val):
        if val <= 4:
            return 0
        elif val <= 25:
            return 1
        else:
            return 2
    train_x['E28'] = train_x['E28'].apply(cut_E28)
    def cut_E29(val):
        if val <= 20:
            return 0
        elif val <= 25:
            return 1
        else:
            return 2
    train_x['E29'] = train_x['E29'].apply(cut_E29)
    X=train_x.drop('Label',axis=1)
    for col in X.columns:
        #print('{}:{}'.format(col,X[col].unique()))
        enc=LabelEncoder()
        X[col]=enc.fit_transform(X[col])
      #  print(col)
    for feat in NUMERICAL_FEATURES:
        stand_scaler = preprocessing.StandardScaler()
        nume_x[feat] = stand_scaler.fit(nume_x[[feat]]).transform(nume_x[[feat]])
    num_train=X.shape[0]//10*(9)
    X,nume_x,Y=np.array(X),np.array(nume_x),np.array(train_y['label'])
    train_x,train_cate_x,train_nume_x,train_y=X[:num_train],X[:num_train],nume_x[:num_train],Y[:num_train]
    test_x,test_cate_x,test_nume_x,test_y=X[num_train:],X[num_train:],nume_x[num_train:],Y[num_train:]
    return train_x, train_cate_x, train_nume_x, test_x, test_cate_x, test_nume_x, train_y, test_y

# def pre_flight():
#     TRAIN_DATA_PATH = '/home/xiao/PycharmProjects/毕业设计/data/train_sample_flight.csv'
#     TEST_DATA_PATH = '/home/xiao/PycharmProjects/毕业设计/data/test_sample_flight.csv'
#     # TRAIN_LABEL_PATH = 'data/adv_predict_train_label.csv'
#     train_data = pd.read_csv(TRAIN_DATA_PATH)
#     test_data = pd.read_csv(TEST_DATA_PATH)
#     # test_label = train_data['']
#     m1, n1 = train_data.shape
#     m2, n2 = test_data.shape
#     X: pd.DataFrame = pd.concat([train_data, test_data])
#     labels = X['Cancelled']
#     feat_remv = ['Cancelled', 'Year', 'FlightNum', 'TailNum']
#     Data_preprocessing(X, feat_remv)
#     cate_feats = ['Month', 'DayofMonth',
#                   'DayOfWeek',
#                   'UniqueCarrier',
#                   'Origin', 'Dest', 'Diverted']
#     nume_feats = ['DepTime', 'ArrTime',
#                   'CRSArrTime', 'ActualElapsedTime',
#                   'CRSElapsedTime', 'ArrDelay',
#                   'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut']
#     X['DayofMonth'] = X['DayofMonth'].apply(lambda x: x // 10)
#     X.fillna(0, inplace=True)
#     for feat in cate_feats:
#         label_encoder = preprocessing.LabelEncoder()
#         X[feat] = label_encoder.fit_transform(X[feat])
#     print('离散特征标签化完成')
#     # print(train_data.columns)
#     for feat in nume_feats:
#         stand_scaler = preprocessing.StandardScaler()
#         X[feat] = stand_scaler.fit(X[[feat]]).transform(X[[feat]])
#     print('连续性特征处理完成')
#     cate_x, nume_x = np.array(X[cate_feats]), np.array(X[nume_feats])
#     x = np.array(X[cate_feats + nume_feats])
#     train_x, test_x = x[:m1, :], x[m1:, :]
#     labels = np.array(labels)
#     train_y, test_y = labels[:m1], labels[m1:]
#
#     train_cate_x, test_cate_x = cate_x[:m1, :], cate_x[m1:, :]
#     train_nume_x, test_nume_x = nume_x[:m1, :], nume_x[m1:, :]
#
#     # train_x, test_x = x[:m1, :], x[m1:, :]
#     print('总共有正项:{}'.format(sum(train_y)))
#     return train_x, train_cate_x, train_nume_x, test_x, test_cate_x, test_nume_x, np.array(train_y), np.array(test_y)


# def pre_criteo(label_name='treatment'):
#     '''
#
#     :param label_name: 有两个可选项，一个是treatment标签，
#     另外一个是conversion标签
#     这个数据集中只存在连续性特征
#     :return:
#     '''
#     train_path=''
#     test_path='/home/xiao/criteo/test.txt'
#     f=open(test_path,'r')
#     for i in range(10):
#         line=f.readline()
#         print(line)

def pre_titanic():
    root1 = os.path.dirname(os.path.realpath(__file__))
    data_path=root1+'/data/titanic_train'
    X = pd.read_csv(data_path)
    X = X.drop(['Ticket', 'Cabin'], axis=1)
    Y = np.array(X['Survived'])
    X['Title'] = X['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    X['Title'] = X['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                     'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                     'Jonkheer', 'Dona'], 'Rare')
    X['Title'] = X['Title'].replace('Mlle', 'Miss')
    X['Title'] = X['Title'].replace('Ms', 'Miss')
    X['Title'] = X['Title'].replace('Mme', 'Mrs')
    enc1 = LabelEncoder()
    X['Title'] = enc1.fit_transform(X['Title'])
    enc1 = LabelEncoder()
    X['Sex'] = enc1.fit_transform(X['Sex'])
    guess_ages = np.zeros((2, 3))
    # 根据性别和等级预测年龄的步骤
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = X[(X['Sex'] == i) & (X['Pclass'] == j + 1)]['Age'].dropna()
            #             print(guess_ages)
            #         age_mean = guess_df.mean()
            #         age_std = guess_df.std()
            #         age_guess = rnd.uniform(age_mean - age_std,age_mean+age_std)
            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    # 进行填充
    for i in range(0, 2):
        for j in range(0, 3):
            X.loc[(X.Age.isnull()) & (X.Sex == i) & (X.Pclass == j + 1), 'Age'] = guess_ages[i, j]
    X['Age'] = X['Age'].astype(int)
    def cut_age(val):
        if val<=16:
            return 0
        elif val<=32:
            return 1
        elif val<=48:
            return 2
        return 3
    X['Age']=X['Age'].apply(lambda x:cut_age(x)).astype('int')
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X['IsAlone'] = 0
    X.loc[X['FamilySize'] == 1, 'IsAlone'] = 1
    X = X.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    X = X.drop(['Name', 'PassengerId'], axis=1)
    #组合特征
    X['Age*Class'] = X.Age * X.Pclass
    # train_df.loc[:,['Age*Class','Age','Pclass']].head(10)
    freq_port = X.Embarked.dropna().mode()[0]
    X['Embarked'] = X['Embarked'].fillna(freq_port)
    enc1 = LabelEncoder()
    X['Embarked'] = enc1.fit_transform(X['Embarked'])
    def cut_fare(x):
        if x<=7.91:
            return 0
        elif x<=14.454:
            return 1
        elif x<=31:
            return 2
        return 3
    X['Fare']=X['Fare'].apply(cut_fare)
    X['Fare'] = X['Fare'].astype(int)
    X = np.array(X)
    num_item = X.shape[0]
    idxes = np.array(range(num_item))
    np.random.shuffle(idxes)
    num_train = (num_item // 10) * 8
    train_x = X[:num_train]
    test_x = X[num_train:]
    train_y = Y[:num_train]
    test_y = Y[num_train:]
    return train_x, train_x, train_x, test_x, test_x, test_x, train_y, test_y


def pre_moscow():
    X=pd.read_csv('/home/xiao/俄罗斯房价/adv_train.csv')
    Y=np.array(X['price_doc'])
    X = X.drop(['id','price_doc'], axis=1)
    X=X.dropna(axis=1)
    num_feats = []
    cate_feats = []
    for col in X.columns:
        if X[col].dtype == 'float':
            num_feats.append(col)
            std = StandardScaler()
            X[[col]] = std.fit(X[[col]]).transform(X[[col]])
            num_feats.append(col)
        elif X[col].dtype == 'int':
            std = StandardScaler()
            num_feats.append(col)
            X[[col]] = std.fit_transform(X[[col]])
        else:
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col])
            cate_feats.append(col)
    num_item = X.shape[0]
    idxes = np.array(range(num_item))
    np.random.shuffle(idxes)
    num_train = (num_item // 10) * 8
    num_X=np.array(X[num_feats])
    cate_X=np.array(X[cate_feats])
    train_x=np.array(X)[:num_train]
    test_x=np.array(X)[num_train:]
    train_num_x,test_num_x=num_X[:num_train],num_X[num_train:]
    train_cate_x,test_cate_x=cate_X[:num_train],cate_X[num_train:]
    return train_x, train_cate_x, train_num_x, test_x, test_cate_x, test_num_x, Y[:num_train], Y[num_train:]

def pre_Chicago():
    root1 = os.path.dirname(os.path.realpath(__file__))
    data_path = root1 + '/data/Chicago_Crimes.csv'
    X = pd.read_csv(data_path, nrows=8000)
    Y = X['Arrest']
    X = X.drop(['Arrest', 'Unnamed: 0', 'ID', 'Case Number', 'Location Description', 'Year', 'Updated On', 'Location'],
               axis=1)
    # X=X.drop('ID',axis=1)
    # X=X.drop('Case Number',axis=1)
    X['Date'] = pd.to_datetime(X['Date'])
    X['year'] = X['Date'].dt.year
    X['DayOfWeek'] = X['Date'].dt.dayofweek
    X['WeekOfYear'] = X['Date'].dt.weekofyear
    X['month'] = X['Date'].dt.month
    X['Hour'] = X['Date'].dt.hour
    X['Date'] = X['Date'].dt.date
    X = X.drop('Date', axis=1)
    X['Latitude'] = X['Latitude'].fillna(X['Latitude'].mean())
    X['Longitude'] = X['Longitude'].fillna(X['Longitude'].mean())
    X['X Coordinate'] = X['X Coordinate'].fillna(method='pad')
    X['Y Coordinate'] = X['Y Coordinate'].fillna(method='pad')
    X['Community Area'] = X['Community Area'].fillna(method='pad')
    # print(sum(X.isnull().sum()))
    if sum(X.isnull().sum()!=0):
        print('存在缺失值，当前程序退出')
        return
    else:
        print('当前数据不存在缺失值')
    num_feats = ['month','year','Beat', 'Latitude', 'Longitude', 'District', 'Ward', 'Community Area']
    print('连续属性:{}'.format(len(num_feats)))
    print('离散属性个数:{}'.format(X.shape[1]-len(num_feats)))
    num_X = X[num_feats].copy()
    def cut_Block(val):
        if val.split(' ')[-1] == 'AVE':
            return 0
        return 1
    X['Block'] = X['Block'].apply(cut_Block)
    def cut_IUCR(val):
        if val == '0110':
            return 0
        return 1
    X['IUCR'] = X['IUCR'].apply(cut_IUCR)
    def cut_PrimaryType(val):
        if val == 'GAMBLING' or val == 'SEX OFFENSE' or val == 'PROSTITUTION':
            return 0
        elif val == 'NARCOTICS' or val == 'WEAPONS VIOLATION':
            return 1
        elif val == 'HOMICIDE':
            return 2
        elif val == 'BATTERY' or val == 'ASSAULT' or val == 'PUBLIC PEACE VIOLATION':
            return 3
        elif val == 'OFFENSE INVOLVING CHILDREN' or val == 'CRIMINAL TRESPASS':
            return 4
        elif val == 'THEFT' or val == 'CRIMINAL DAMAGE' or val == 'OTHER OFFENSE' or val == 'MOTOR VEHICLE THEFT':
            return 5
        return 6
    X['Primary Type'] = X['Primary Type'].apply(cut_PrimaryType)
    def cut_Description(val):
        if val == 'FIRST DEGREE MURDER':
            return 0
        return 1
    X['Description'] = X['Description'].apply(cut_Description)
    # def cut_Location_Description(val):
    #     if val=='STREET':
    #         return 0
    #     return 1
    # X['Location Description']=X['Location Description'].apply(cut_Location_Description)
    # X=X.drop('Location Description',axis=1)
    def cut_Beat(val):
        if val <= 500:
            return 0
        elif val <= 2000:
            return 1
        return 2
    X['Beat'] = X['Beat'].apply(cut_Beat)
    def cut_District(val):
        if val <= 6:
            return 0
        elif val <= 16:
            return 1
        return 2
    X['District'] = X['District'].apply(cut_District)
    def cut_Ward(val):
        if val <= 9:
            return 0
        elif val <= 21:
            return 2
        elif val <= 31:
            return 3
        return 4
    X['Ward'] = X['Ward'].apply(cut_Ward)
    def cut_Community_Area(val):
        if val <= 24:
            return 0
        elif val <= 39:
            return 1
        elif val <= 59:
            return 2
        return 3
    X['Community Area'] = X['Community Area'].apply(cut_Community_Area)
    def cut_FBICode(val):
        if val == '19' or val == '16':
            return 0
        elif val == '18' or val == '15':
            return 1
        elif val == '24' or val == '04A' or val == '17':
            return 2
        elif val == '01A' or val == '04B':
            return 3
        elif val == '26' or val == '08B' or val == '20':
            return 4
        if val in {'06', '07', '14'}:
            return 5
    X['FBI Code'] = X['FBI Code'].apply(cut_FBICode)
    for col in X.columns:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col])
    for feat in num_feats:
        enc = StandardScaler()
        num_X[feat] = enc.fit(num_X[[feat]]).transform(num_X[[feat]])
    num_train = (X.shape[0])
  #  print(X.shape)
    X, num_X, Y = np.array(X), np.array(num_X), np.array(Y)
    idxes=np.array(range(num_train))
    np.random.shuffle(idxes)
    num_train = (X.shape[0]//10)*8
    X, num_X, Y = X[idxes], num_X[idxes], Y[idxes]
    train_X, train_num_X, train_Y = X[:num_train], num_X[:num_train], Y[:num_train]
    test_X, test_num_X, test_Y = X[num_train:], num_X[num_train:], Y[num_train:]
 #   train_x, train_cate_x, train_num_x, test_x, test_cate_x, test_num_x, Y[:num_train], Y[num_train:]
 #    print(train_X.shape)
 #    print(train_num_X.shape)
 #    print(train_X.shape)
 #    print(test_X.shape)
    return train_X,train_X,train_num_X,test_X,test_X,test_num_X,train_Y,test_Y


if __name__ == '__main__':
    train_x, train_cate_x, train_num_x, test_x, test_cate_x, test_num_x, train_y,test_y=pre_Chicago()
    print(train_cate_x.shape[1])
    print(train_num_x.shape[1])
   # print(train_cate_x[:10])
    # train_x, _, _, test_x, _, _, train_y, test_y = pre_titanic()
    # print(train_x[:10])
    # print(train_x.shape)
    # print(train_y.shape)
    # print()
