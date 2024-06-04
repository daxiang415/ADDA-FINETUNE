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
    tokyo_data = pd.read_excel('tokyo_data.xlsx')

    chongsheng_data = pd.read_excel('chongsheng_data.xlsx')

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
    tokyo_data['time'] = pd.to_datetime(tokyo_data['time'], dayfirst=True)  # 改time为时间戳格式

    chongsheng_data['time'] = pd.to_datetime(chongsheng_data['time'], dayfirst=True)  # 改time为时间戳格式

    s_date = '2020-01-01 00:00'
    e_date = '2021-6-30 23:00'
    s_date_test = '2021-06-24 00:00'
    e_date_test = '2021-12-31 23:00'
    # 划分数据集
    train_data_tokyo = tokyo_data[(s_date <= tokyo_data['time']) & (tokyo_data['time'] <= e_date)]
    test_data_tokyo = tokyo_data[(s_date_test <= tokyo_data['time']) & (tokyo_data['time'] <= e_date_test)]

    train_data_chongsheng = chongsheng_data[(s_date <= chongsheng_data['time']) & (chongsheng_data['time'] <= e_date)]
    test_data_chongsheng = chongsheng_data[(s_date_test <= chongsheng_data['time']) & (chongsheng_data['time'] <= e_date_test)]

    train_x_tokyo, train_y_tokyo = train_data_tokyo[props], train_data_tokyo[target]

    train_x_chongsheng, train_y_chongsheng = train_data_chongsheng[props], train_data_chongsheng[target]


    # 归一化
    trans_tokyo = StandardScaler().fit(train_x_tokyo)  ####StandardScaler().fit()用于计算 train_x的均值和方差
    train_x_tokyo, train_y_tokyo = scale(train_x_tokyo, train_y_tokyo, trans_tokyo)

    #trans_chongsheng = StandardScaler().fit(train_x_chongsheng)  ####StandardScaler().fit()用于计算 train_x的均值和方差
    train_x_chongsheng, train_y_chongsheng = scale(train_x_chongsheng, train_y_chongsheng, trans_tokyo)

    y_mean = train_y_tokyo.mean()
    y_std = train_y_tokyo.std()

    # train_y_tokyo = (train_y_tokyo - y_mean) / y_std
    #
    # train_y_chongsheng = (train_y_chongsheng- y_mean) / y_std





    return (train_x_tokyo, train_y_tokyo), (train_x_chongsheng, train_y_chongsheng)



class TrainData(Dataset):
    def __init__(self):
        super(TrainData, self).__init__()
        (self.train_x_tokyo, self.train_y_tokyo), (self.train_x_chongsheng, self.train_y_chongsheng) = load_data()
        self.encode_x_tokyo = np.concatenate([self.train_x_tokyo, self.train_y_tokyo], axis=-1) ####编码意味着在最后的维度上将 train_x和train_y 进行拼接
        self.encode_x_chongsheng = np.concatenate([self.train_x_chongsheng, self.train_y_chongsheng], axis=-1)
        self.step = 168  ####时间周期
        self.prum = 1  ####单步预测

    def __len__(self):   ####下划线的定义： _ 隐藏函数, __实例化后自动调用此函数
        return self.train_x_tokyo.shape[0] - self.step

    def __getitem__(self, item):   ###item: 在len的范围内随机取一个数字
        en_tokyo = self.train_x_tokyo[item:item + self.step, :]
        de_tokyo = self.train_x_tokyo[item + self.step:item + self.step + self.prum, :2]
        tg_tokyo = self.train_y_tokyo[item + self.step:item + self.step + self.prum].reshape(-1)

        en_chongsheng = self.train_x_chongsheng[item:item + self.step, :]
        de_chongsheng = self.train_x_chongsheng[item + self.step:item + self.step + self.prum, :2]
        tg_chongsheng = self.train_y_chongsheng[item + self.step:item + self.step + self.prum].reshape(-1)
        return en_tokyo, de_tokyo, tg_tokyo, en_chongsheng, de_chongsheng, tg_chongsheng





if __name__ == "__main__":
    trainData = TrainData()
    genData = DataLoader(trainData, batch_size=16, shuffle=True)
    for i, (en, de, tg, _, _, _) in enumerate(genData):
        print(i, en.shape, de.shape, tg.shape)