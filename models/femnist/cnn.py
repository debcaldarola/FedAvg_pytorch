import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

IMAGE_SIZE = 28


class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        super(ClientModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, self.num_classes)
        self.size = self.model_size()

    def forward(self, x):
        x = self.layer1(x.float())
        x = self.layer2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits

    def process_x(self, raw_x_batch):
        x_batch = np.array(raw_x_batch)
        x_batch = np.reshape(x_batch, (x_batch.shape[0], 1, IMAGE_SIZE, IMAGE_SIZE))  # 1 channel (black and white img)
        return x_batch

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size