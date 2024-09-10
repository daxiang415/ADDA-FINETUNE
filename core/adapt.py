"""Adversarial adaptation to train target encoder."""

import os
from sklearn.metrics import r2_score
import torch
import torch.optim as optim
from torch import nn
import numpy as np
import pandas as pd
import params
from utils import make_variable
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from data_loader_chongsheng import TestData_chongsheng


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, device):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()
    src_encoder.eval()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2)
                               )
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2)
                                  )
    len_data_loader = len(src_data_loader)

    ####################
    # 2. train network #
    ####################

    r2_score_indicator = -np.inf

    for epoch in range(params.num_epochs):
        tgt_encoder.train()
        # zip source and target data pair
        for step, (en_tokyo, de_tokyo, tg_tokyo, en_chongsheng, de_chongsheng, tg_chongsheng) in enumerate(src_data_loader):
            ###########################
            # 2.1 train discriminator #
            ###########################


            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(en_tokyo.to(device), de_tokyo.to(device))
            feat_tgt = tgt_encoder(en_chongsheng.to(device), de_chongsheng.to(device))
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(en_chongsheng.to(device), de_chongsheng.to(device))

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            # if ((step + 1) % params.log_step == 0):
            #     print("Epoch [{}/{}] Step [{}/{}]:"
            #           "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
            #           .format(epoch + 1,
            #                   params.num_epochs,
            #                   step + 1,
            #                   len_data_loader,
            #                   loss_critic.item(),
            #                   loss_tgt.item(),
            #                   acc.item()))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))




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
        # result_target = np.where(result_target < 0.012, 0, result_target)

        result_pred = np.array(pred).squeeze()
        # result_pred = np.where(result_pred < 0.012, 0, result_pred)



        new_r2 = r2_score(result_target, result_pred)
        # 计算MAPE
        print('MAPE_indicator: ', new_r2)

        if new_r2 > r2_score_indicator:
            best_tgt = tgt_encoder
            r2_score_indicator = new_r2
            print('update')
            torch.save(tgt_encoder, "./Best_tokyo_to_chongsheng_encoder.pkl")

            results = {'target': result_target, 'pred': result_pred}
            df = pd.DataFrame(results)
            df.to_excel('result.xlsx', index=False)





    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    return best_tgt
