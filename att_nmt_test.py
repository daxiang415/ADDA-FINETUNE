from torch.utils.data import DataLoader
from data_loader import TestData
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载数据
testData = TestData()
genTestData = DataLoader(testData, batch_size=1, shuffle=False)
# 加载模型
encoder = torch.load('Best_chongsheng_encoder_ADDA_FINETUNE.pkl')
classifier = torch.load('Best_chongsheng_classifier_ADDA_FINETUNE.pkl')
encoder.to(device)
classifier.to(device)
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



# 结果保存
result_target = np.array(tagt).squeeze()
#result_target = np.where(result_target < 0.012, 0, result_target)

result_pred = np.array(pred).squeeze()
#result_pred = np.where(result_pred < 0.012, 0, result_pred)

results = {'target': result_target, 'pred': result_pred }
df = pd.DataFrame(results)
df.to_excel('直接微调.xlsx', index=False)
# 计算MAPE
print('MAPE_indicator: ', r2_score(result_target, result_pred))
# # 画图
# plt.figure(figsize=(10, 5))
# plt.plot(np.array(tagt).squeeze())
# plt.plot(np.array(pred).squeeze())
# plt.legend(['gnd', 'pred'])
# plt.show()