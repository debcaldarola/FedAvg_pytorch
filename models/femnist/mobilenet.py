import numpy as np
import torch
import torch.nn as nn

IMAGE_SIZE = 28

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        self.size = self.model_size()

    def forward(self, x):
        x = self.model(x)
        return x

    def process_x(self, raw_x_batch):
        x_batch = np.array(raw_x_batch)
        x_batch = np.reshape(x_batch, (x_batch.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1))
        x_batch = np.repeat(x_batch,3,3) # to RGB
        return x_batch

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size

    def process_batch(self, x_list, y_list):
        x_batch = self.process_x(x_list)
        y_batch = self.process_y(y_list)
        return x_batch, y_batch