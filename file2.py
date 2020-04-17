
from models.DeepFM import *
from preprocess.data_preprocess import *
from models.Gbdt2nn import *
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
breast_cancer=load_breast_cancer()
data=breast_cancer.data
target=breast_cancer.target
print(data.shape)
print(target.shape)
#print(breast_cancer['feature_names'])
# print(data[:10])
# print(target[:10])
train_x,test_x,train_y,test_y=train_test_split(data,target,test_size=0.2)
print(train_y[:10])
print(train_x.shape)