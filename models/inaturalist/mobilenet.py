import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.models as models
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
        # self.model = models.mobilenet_v2(pretrained=True)
        self.swapBN_toGN() # Swap bach norm layers with group norm ones
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True))
        # print(self.model)

    def forward(self, x):
        x = self.model.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x

    def swapBN_toGN(self):
        """Swaps Batch Normalization layers in self.model with Group Normalization layers
        (best suited for federated scenarios).
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Get current bn layer
                bn = self._get_layer(self.model, name)
                # Create new gn layer
                gn = nn.GroupNorm(1, bn.num_features)
                # Assign gn
                # print("Swapping {} with {}".format(bn, gn))
                self._set_layer(name, gn)

    def process_x(self, raw_x_batch):
        """Extracts images according to their path.
        Images having size < IMAGE_SIZE are removed.

        Return:
            numpy array containing batch of opened images
        """
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
        """Processes images labels.
        Labels corresponding to removed images are deleted.

        Return:
            numpy array of labels
        """
        if self.indexes_to_be_removed is not None:
            for i in reversed(self.indexes_to_be_removed):  # altrimenti cambio gli indici
                del raw_y_batch[i]
            self.indexes_to_be_removed = None
        return np.array(raw_y_batch)

    def _load_image(self, img_name):
        """Loads images from path and applies transforms.

        Args:
            img_name: image name to be appended to the train/test path

        Return:
            img: image opened and converted to RGB format with applied transforms in numpy format
        """
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

    def _get_layer(self, model, name):
        layer = model
        for attr in name.split("."):
            layer = getattr(layer, attr)
        return layer

    def _set_layer(self, name, layer):
        try:
            attrs, name = name.rsplit(".", 1)
            model = self._get_layer(self.model, attrs)
        except ValueError:
            pass
        setattr(model, name, layer)
