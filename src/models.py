import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNNNorm(nn.Module):
    """Normalization built for cnns input"""
    def __init__(self, norm_type, n_channels):
        super(CNNNorm, self).__init__()
        if norm_type == 'Group':
            self.norm = nn.GroupNorm(int(n_channels/4), n_channels)
            # If 16 channels, then separate 16 channels into n_groups = n_channels/4 i.e. 16/4 = 4 groups (1 group having 4 channels each)
        elif norm_type == 'Layer':
            self.norm = nn.GroupNorm(1, n_channels)
            # If 16 channels, then separate 16 channels into 1 group & use GroupNorm. This akin to "layer norm"
        elif norm_type == 'Batch':
            self.norm  = nn.BatchNorm2d(n_channels)
        else:
            raise Exception('Illegal normalization type')

    def forward(self, x):
        x = self.norm(x)
        return x

class CNNBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dropout_value, norm_type):
        super(CNNBlocks, self).__init__()

        self.cnn     = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout_value)
        self.norm_type = norm_type
        self.norm = CNNNorm(norm_type, out_channels)

    def forward(self, x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class S6_CNNModel(nn.Module):

    def __init__(self, n_cnn_blocks, n_class, dropout_value, norm_type):
        super(S6_CNNModel, self).__init__()

        self.n_class = n_class
        self.cnn_block0 = nn.Sequential(CNNBlocks(1, 16, (3, 3), 0, dropout_value, norm_type))
        self.cnn_block1 = nn.Sequential(CNNBlocks(16, 16, (3, 3), 0, dropout_value, norm_type))
        self.pool1 = nn.MaxPool2d(2, 2)

        self.cnn_block_2_3_4 = nn.Sequential(*[
            CNNBlocks(16, 16, (3, 3), 0, dropout_value, norm_type)
            for block in range(n_cnn_blocks)
        ])

        self.Gap1 = nn.Sequential(nn.AvgPool2d(kernel_size=6))
        self.fc1 = nn.Conv2d(in_channels=16, out_channels=self.n_class, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x):
        x = self.cnn_block0(x)
        x = self.cnn_block1(x)
        x = self.pool1(x)
        x = self.cnn_block_2_3_4(x)
        x = self.Gap1(x)
        x = self.fc1(x)
        x = x.view(-1, self.n_class)
        return F.log_softmax(x, dim=-1)