import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMAGES_DIR = os.path.join('..', 'data', 'glv2', 'data', 'raw', 'train')
IMAGE_SIZE = 224

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device, model):
        super(ClientModel, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(*list(model.children())[:1])
        state_dict = dict(zip(self.state_dict().keys(), model.state_dict().values()))
        self.load_state_dict(state_dict)
        self.size = self._model_size()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def process_x(self, x_list):
        x_batch = [self._load_image(i) for i in x_list]
        x_batch = np.array(x_batch)
        return x_batch

    def _load_image(self, img_name):
        path = os.path.join(IMAGES_DIR, img_name[0], img_name[1], img_name[2])
        img_name = img_name + ".jpg"
        img = Image.open(os.path.join(path, img_name))
        preprocess = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        return input_tensor.cpu().detach().numpy()

    def process_y(self, raw_y_batch):
        return raw_y_batch

    def _model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size