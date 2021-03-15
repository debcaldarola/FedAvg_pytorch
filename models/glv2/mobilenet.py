import os
import numpy as np
import torch
import torch.nn as nn
import PIL
from PIL import Image
from torchvision import transforms

# IMAGES_DIR = os.path.join('..', 'data', 'glv2', 'data', 'raw', 'train')
# IMAGES_DIR = os.path.join('/', 'work', 'gberton', 'shared', 'datasets', 'glv2', 'train')
IMAGES_DIR = os.path.join('/', 'home', 'valerio', 'datasets', 'classification_datasets', 'glv2', 'train')
IMAGE_SIZE = 224

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)

        # self.features = nn.Sequential(*list(model.children())[:-1])
        # self.classifier = nn.Sequential(*list(model.children())[-1])
        #
        # state_dict = dict(zip(self.state_dict().keys(), model.state_dict().values()))
        # self.load_state_dict(state_dict)
        #
        # # substitute BN layers with GN
        # self.features[0][0][1] = nn.GroupNorm(1, self.features[0][0][1].num_features)
        # self.features[0][1].conv[0][1] = nn.GroupNorm(1, self.features[0][1].conv[0][1].num_features)
        # self.features[0][1].conv[2] = nn.GroupNorm(1, self.features[0][1].conv[2].num_features)
        #
        # for i in range(2, 18):
        #     self.features[0][i].conv[0][1] = nn.GroupNorm(1, self.features[0][i].conv[0][1].num_features)
        #     self.features[0][i].conv[1][1] = nn.GroupNorm(1, self.features[0][i].conv[1][1].num_features)
        #     self.features[0][i].conv[3] = nn.GroupNorm(1, self.features[0][i].conv[3].num_features)
        # self.features[0][18][1] = nn.GroupNorm(1, self.features[0][18][1].num_features)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.relu = nn.ReLU()
        # self.classifier2 = nn.Linear(in_features=64, out_features=self.num_classes, bias=True)
        self.size = self._model_size()

    def forward(self, x):
        # x = self.features(x)
        x = self.model(x)
        # x = self.avgpool(x)
        # x = x.reshape(x.shape[0], -1)
        # x = self.classifier(x)
        # x = self.relu(x)
        # x = self.classifier2(x)
        return x

    def process_x(self, x_list):
        x_batch = []
        for i in x_list:
            x = self._load_image(i)
            if x is None:
                continue
            x_batch.append(x)
        # x_batch = [self._load_image(i) for i in x_list]
        if len(x_batch) == 0:
            print("No images")
            return None
        x_batch = np.array(x_batch)
        x_batch = np.reshape(x_batch, (x_batch.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
        return x_batch

    def _load_image(self, img_name):
        path = os.path.join(IMAGES_DIR, img_name[0], img_name[1], img_name[2])
        if not os.path.exists(path):
            # print("not existing path:", path)
            # return np.random.rand(3,224,224)
            # return np.zeros((3,224,224))
            return None
        img_name = img_name + ".jpg"
        img_path = os.path.join(path, img_name)
        if not os.path.exists(img_path):
            # print("not existing img:", img_name)
            # return np.random.rand(3,224,224)
            # return np.zeros((3,224,224))
            return None
        try:
            img = Image.open(img_path)
        except PIL.UnidentifiedImageError:
            # print("Corrupted image:",img_path)
            # return np.random.rand(3, 224, 224)
            return None
            # return np.zeros((3,224,224))
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

    def process_batch(self, x_list, y_list):
        x_batch = []
        y_batch = []
        for x, y in zip(x_list, y_list):
            i = self._load_image(x)
            if i is not None:
                x_batch.append(i)
                y_batch.append(y)
        if len(y_batch) == 0:
            print("empty batch")
        x_batch = np.array(x_batch)
        x_batch = np.reshape(x_batch, (x_batch.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
        return x_batch, y_batch

    def _model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size
