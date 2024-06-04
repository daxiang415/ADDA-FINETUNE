import torch
from transformer import Att_NMT, classifier
import torch.nn as nn
import torch.optim as optim
from data_loader import TrainData, TestData
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score
import random
import pandas as pd


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

def test(encoder, classifier, genTestData, MAPE, device):
    encoder.eval()
    classifier.eval()
    # 开始测试
    pred = []
    tagt = []
    with torch.no_grad():
        for i, (en, de, tg) in enumerate(genTestData):
            out = classifier(encoder(en.to(device), de.to(device)))
            pred.append(out[0].item())
            tagt.append(tg.item())


    result_target = np.array(tagt).squeeze()


    result_pred = np.array(pred).squeeze()


    _MAPE = r2_score(result_target, result_pred)
    print('MAPE: ', _MAPE)
    if _MAPE > MAPE:
        MAPE = _MAPE
        # 保存模型
        torch.save(encoder, "./Best_chongsheng_encoder_adda_FINETUNE.pkl")
        torch.save(classifier, "./Best_chongsheng_classifier_adda_FINETUNE.pkl")
        print("The model updated...")

        results = {'target': result_target, 'pred': result_pred}
        df = pd.DataFrame(results)

        df.to_excel('东京到冲绳ADDA微调.xlsx', index=False)


    return MAPE






Epoch = 100
# Model parameters


# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(f"Using device {device}")

# Load transformer with Adam optimizer and MSE loss function

encoder_model = Att_NMT(feature_size = 8, addition_feature_size=2, output_dim= 1, encoder_length=168,lstm_size=128, training=True).to(device)

classifier = classifier(lstm_size=128).to(device)

# 加载权重
loaded_model = torch.load('Best_tokyo_to_chongsheng_encoder.pkl')

weights = loaded_model.state_dict()

# 将权重初始化到模型中
encoder_model.load_state_dict(weights)


# 加载权重分类器
loaded_model_classifer = torch.load('Best_tokyo_classifier.pkl')

weights_class = loaded_model_classifer.state_dict()

# 将权重初始化到模型中
classifier.load_state_dict(weights_class)


# 冻结encoder

# for name, para in encoder_model.named_parameters():
#     para.requires_grad_(False)

criterion = nn.MSELoss(reduce=True)
# criterion = nn.MSELoss()
encoder_lr = 0.000001
#encoder_lr = 0.001# 为 encoder 设置的学习率
classifier_lr = 0.001  # 为 classifier 设置的学习率

# 创建一个包含不同参数组和各自学习率的优化器
optimizer = optim.Adam([
    {'params': encoder_model.parameters(), 'lr': encoder_lr},
    {'params': classifier.parameters(), 'lr': classifier_lr}
])

# 加载数据
trainData = TrainData()
genTrainData = DataLoader(trainData, batch_size=32, shuffle=True, worker_init_fn=66)
testData = TestData()
genTestData = DataLoader(testData, batch_size=1, worker_init_fn=66)

KL_weighted =0.0000001


set_seed(66)

# 训练
MAPE = -np.inf
for ep in range(Epoch):
    loss_mean = 0.
    encoder_model.train()
    classifier.train()
    for i, (feature, add_feature, tg) in enumerate(genTrainData):
        optimizer.zero_grad()
        feature, add_feature, tg = feature.to(device), add_feature.to(device), tg.to(device)
        out = classifier(encoder_model(feature, add_feature))
        m = out.reshape(-1, 1)

        loss = criterion(m, tg)

        loss.backward()
        # update
        optimizer.step()
        # 打印训练信息
        loss_mean += loss.item()





    # 测试
    MAPE = test(encoder_model, classifier, genTestData, MAPE, device)
