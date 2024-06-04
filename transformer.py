import torch
import torch.nn as nn
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F


from torch.autograd import Variable



class Att_NMT(nn.Module):
    def __init__(self, feature_size, addition_feature_size, output_dim, encoder_length, lstm_size, training=True):
        super(Att_NMT, self).__init__()
        # encoder应该包含label的序列
        # 作为修改，此处尝试双向LSTM
        self.encoder = nn.GRU(input_size=feature_size + 1, hidden_size=lstm_size, num_layers=1, bidirectional=False,
                               batch_first=True, dropout=0.3)  # if batch_first==False: input_shape=[length,batch_size,embedding_size]
        # decoder的inputsize需要变化，加上2倍的LSTM_size，因为attention的维度就是这个
        self.decoder = nn.Linear(lstm_size * encoder_length, lstm_size)

        # 最后预测的FC的维度也需要修改

        self.fc_1 = nn.Linear(addition_feature_size + lstm_size, lstm_size)
        self.fc_2 = nn.Linear(lstm_size, 1)
        self.training = training
        self.encoder_length = encoder_length


    def forward(self, source_data, target_data):

        enc_output, enc_hidden = self.encoder(source_data)

        class_input = enc_output.reshape(enc_output.shape[0], -1)


        out = self.decoder(class_input)
        out = F.relu(out)

        target_data = target_data.reshape(enc_output.shape[0], -1)

        out = torch.cat([out, target_data], dim=-1)

        out = self.fc_1(out)

        return out



class classifier(nn.Module):
    def __init__(self, lstm_size, training=True):
        super(classifier, self).__init__()

        self.fc_1 = nn.Linear(lstm_size, 1, bias=True)
        self.training = training



    def forward(self, out):


        out = self.fc_1(out)

        return out


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out