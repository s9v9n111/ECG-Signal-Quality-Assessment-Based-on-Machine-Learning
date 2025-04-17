import torch
import torch.nn as nn


class MLP(nn.Module):
    """多层感知机模型（3层结构）"""

    def __init__(self, input_size, hidden_size=64, num_classes=3):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    """一维卷积神经网络模型（2层卷积+全连接）"""

    def __init__(self, input_length, num_classes=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 计算全连接层输入尺寸
        self.fc_input_size = 32 * (input_length // (2 ** 2))
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))  # (batch, 16, L/2)
        x = self.pool2(self.relu2(self.conv2(x)))  # (batch, 32, L/4)
        x = x.view(x.size(0), -1)  # 展平为全连接输入
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x