import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('..', 'data', 'celeba', 'data', 'raw', 'img_align_celeba')

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.device = device
        self.num_classes = num_classes
        model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        model.classifier[1] = nn.Linear(in_features=11520, out_features=num_classes, bias=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(*list(model.children())[-1])
        state_dict = dict(zip(self.state_dict().keys(), model.state_dict().values()))
        self.load_state_dict(state_dict)
        self.size = self.model_size()

    def forward(self, x):
        x = self.features(x)
        x = torch.reshape(x,(x.shape[0], -1))
        logits = self.classifier(x)
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