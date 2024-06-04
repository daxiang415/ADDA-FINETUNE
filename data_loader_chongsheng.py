import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    random.seed(seed)         # 设置Python内建的随机种子
    np.random.seed(seed)     # 设置NumPy的随机种子
    torch.manual_seed(seed)  # 设置PyTorch的随机种子

    if torch.cuda.is_available():   # 如果你使用CUDA
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_all_seeds(42)

def scale(x, y, trans):
    #x = trans.transform(x).astype(np.float32) ####transform 将训练数据转换成正态分布

    x = x.to_numpy().astype(np.float32)
    #y = np.log10(y.to_numpy().astype(np.float32))
    y=y.to_numpy().astype(np.float32)

    return x, y


def load_data():
    # 读取数据
    laixi_data = pd.read_excel('chongsheng_data.xlsx')

    tokyo_data = pd.read_excel('chongsheng_data.xlsx')

    props = [#'day_of_year',
             #'month',
             'hour',
             'sun_rise',
             'wind_speed',
             'radiation_time',
             'rainfall',
             'temperature',
             'humidity',
             'water_vapor_pressure',
             'dew_temperature',
             ]

    target = ['solar_radiation_modification']

    # 时间划分
    laixi_data['time'] = pd.to_datetime(laixi_data['time'], dayfirst=True)  # 改time为时间戳格式
    s_date = '2020-01-01 00:00'
    e_date = '2021-6-30 23:00'
    s_date_test = '2021-06-24 00:00'
    e_date_test = '2021-12-31 23:00'

    # 划分数据集
    train_data_tokyo = tokyo_data[(s_date <= tokyo_data['time']) & (tokyo_data['time'] <= e_date)]

    train_x_tokyo, train_y_tokyo = train_data_tokyo[props], train_data_tokyo[target]

    trans_tokyo = StandardScaler().fit(train_x_tokyo)



    train_data = laixi_data[(s_date <= laixi_data['time']) & (laixi_data['time'] <= e_date)]
    test_data = laixi_data[(s_date_test <= laixi_data['time']) & (laixi_data['time'] <= e_date_test)]

    train_x, train_y = train_data[props], train_data[target]
    test_x, test_y = test_data[props], test_data[target]

    # 归一化
    #trans = StandardScaler().fit(train_x)  ####StandardScaler().fit()用于计算 train_x的均值和方差
    train_x, train_y = scale(train_x, train_y, trans_tokyo)
    test_x, test_y = scale(test_x, test_y, trans_tokyo)

    y_mean = np.mean(train_y)
    y_std = np.std(train_y)

    # train_y = (train_y - y_mean) / y_std
    #
    # test_y = (test_y - y_mean) / y_std





    #print('训练集: ', train_x.shape, train_y.shape)
    #print('测试集: ', test_x.shape, test_y.shape)
    return (train_x, train_y), (test_x, test_y), (y_mean, y_std)



class TrainData_chongsheng(Dataset):
    def __init__(self):
        super(TrainData_chongsheng, self).__init__()
        (self.train_x, self.train_y), _, _ = load_data()
        self.encode_x = np.concatenate([self.train_x, self.train_y], axis=-1) ####编码意味着在最后的维度上将 train_x和train_y 进行拼接
        self.step = 168  ####时间周期
        self.prum = 1  ####单步预测

    def __len__(self):   ####下划线的定义： _ 隐藏函数, __实例化后自动调用此函数
        return self.train_x.shape[0] - self.step

    def __getitem__(self, item):   ###item: 在len的范围内随机取一个数字
        en = self.train_x[item:item + self.step, :]
        de = self.train_x[item + self.step:item + self.step + self.prum, :2]
        tg = self.train_y[item + self.step:item + self.step + self.prum].reshape(-1)
        return en, de, tg


class TestData_chongsheng(Dataset):
    def __init__(self):
        super(TestData_chongsheng, self).__init__()
        _, (self.test_x, self.test_y), (self.mean, self.std) = load_data()
        self.encode_x = np.concatenate([self.test_x, self.test_y], axis=-1)
        self.step = 168
        self.prum = 1

    def __len__(self):  ####下划线的定义： _ 隐藏函数, __实例化后自动调用此函数
        return self.test_x.shape[0] - self.step

    def __getitem__(self, item):  ###item: 在len的范围内随机取一个数字
        en = self.test_x[item:item + self.step, :]
        de = self.test_x[item + self.step:item + self.step + self.prum, :2]
        tg = self.test_y[item + self.step:item + self.step + self.prum].reshape(-1)
        return en, de, tg, self.mean, self.std


if __name__ == "__main__":
    trainData = TestData_chongsheng()
    genData = DataLoader(trainData, batch_size=16, shuffle=True)
    for i, (en, de, tg) in enumerate(genData):
        print(i, en.shape, de.shape, tg.shape)