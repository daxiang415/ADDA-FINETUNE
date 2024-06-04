"""Main script for ADDA."""

import random
import torch
import numpy as np
from data_loader import TrainData
from transformer import Att_NMT, classifier, Discriminator
from torch.utils.data import DataLoader
from core import train_tgt
from data_loader_chongsheng import TestData_chongsheng
import pandas as pd
from sklearn.metrics import r2_score
import params

def set_seed(seed=66):
    # Python的随机种子
    random.seed(seed)

    # Numpy的随机种子
    np.random.seed(seed)

    # PyTorch的随机种子
    torch.manual_seed(seed)

    # 如果你正在使用CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    set_seed(66)

    # load dataset
    trainData = TrainData()
    genTrainData = DataLoader(trainData, batch_size=32, shuffle=True, worker_init_fn=66)


    # load models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f"Using device {device}")

    # Load transformer with Adam optimizer and MSE loss function

    src_encoder = torch.load('Best_tokyo_encoder.pkl')
    src_classifier = torch.load('Best_tokyo_classifier.pkl')

    tgt_encoder = Att_NMT(feature_size=8, addition_feature_size=2, output_dim=1, encoder_length=168, lstm_size=128,
                            training=True).to(device)

    # 加载权重
    loaded_model = torch.load('Best_tokyo_encoder.pkl')

    weights = loaded_model.state_dict()

    # 将权重初始化到模型中
    tgt_encoder.load_state_dict(weights)

    critic = Discriminator(input_dims = 128, hidden_dims = 128, output_dims = 2).to(device)







    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)


    tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic, genTrainData, device)

    # 从这里开始测试迁移的结果

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载数据
    testData = TestData_chongsheng()
    genTestData = DataLoader(testData, batch_size=1, shuffle=False)
    # 加载模型
    classifier = torch.load('Best_tokyo_classifier.pkl')

    tgt_encoder.eval()
    classifier.eval()

    # 开始测试
    pred = []
    tagt = []

    with torch.no_grad():
        for i, (en, de, tg, mean, std) in enumerate(genTestData):
            out = classifier(tgt_encoder(en.to(device), de.to(device)))
            pred.append(out[0].item())
            tagt.append(tg.item())

    # 结果保存
    result_target = np.array(tagt).squeeze()
    #result_target = np.where(result_target < 0.012, 0, result_target)

    result_pred = np.array(pred).squeeze()
    #result_pred = np.where(result_pred < 0.012, 0, result_pred)

    results = {'target': result_target, 'pred': result_pred}
    df = pd.DataFrame(results)
    df.to_excel('result.xlsx', index=False)
    # 计算MAPE
    print('MAPE_indicator: ', r2_score(result_target, result_pred))







