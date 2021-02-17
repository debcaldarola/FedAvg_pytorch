import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
#from model import Model

IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('..', 'data', 'celeba', 'data', 'raw', 'img_align_celeba')


class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        super(ClientModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32*7*7, self.num_classes)
        #self.fc2 = nn.Linear(1024, self.num_classes) #4 filters => 4 feature maps
        # nn.Linear equivalent to tf.layers.dense()
        self.size = self.model_size()

    def forward(self, x):
        x = self.layer1(x.float())
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.reshape(x,(x.shape[0], -1))
        logits = self.fc1(x)
        return logits

    def process_x(self, raw_x_batch):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        return raw_y_batch

    def _load_image(self, img_name):
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        return np.array(img)

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size
