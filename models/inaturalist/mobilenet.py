import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

IMAGE_SIZE = 299
TRAIN_IMAGES_DIR = os.path.join('..', 'data', 'inaturalist', 'data', 'raw', 'train')
TEST_IMAGES_DIR = os.path.join('..', 'data', 'inaturalist', 'data', 'raw', 'test')

transform_train = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        self.indexes_to_be_removed = None
        self.size = self.model_size()

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True))

    def forward(self, x):
        x = self.model.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x

    def process_x(self, raw_x_batch):
        x_batch = []
        for i, img_name in enumerate(raw_x_batch):
            img = self._load_image(img_name)
            if type(img) is int and img == -1:
                if self.indexes_to_be_removed is None:
                    self.indexes_to_be_removed = []
                self.indexes_to_be_removed.append(i)
            else:
                x_batch.append(img)
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        if self.indexes_to_be_removed is not None:
            for i in reversed(self.indexes_to_be_removed):  # altrimenti cambio gli indici
                del raw_y_batch[i]
            self.indexes_to_be_removed = None
        return np.array(raw_y_batch)

    def _load_image(self, img_name):
        if os.path.exists(os.path.join(TRAIN_IMAGES_DIR, img_name + '.jpg')):
            img_dir = TRAIN_IMAGES_DIR
        else:
            img_dir = TEST_IMAGES_DIR

        img = Image.open(os.path.join(img_dir, img_name + '.jpg')).convert('RGB')
        w, h = img.size
        if w < IMAGE_SIZE or h < IMAGE_SIZE:
            return -1
        if self.training:
            img = transform_train(img)
        else:
            img = transform_test(img)
        img = img.cpu().detach().numpy()
        return img

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size