import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import torchvision
import torchvision.transforms as transforms



IMAGE_SIZE = 32
IMAGES_DIR = os.path.join('..', 'data', 'cifar100', 'data', 'raw', 'cifar-100-python')
duplicates = [32,1091,4]


class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        self.data = None    # dictionary containing {img_id: np.array(img)}

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64*5*5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.num_classes)

        self.size = self.model_size()


    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def process_x(self, raw_x_batch):
        if self.data is None:
            self.data = {}
            self._load_images()
        x_batch = [self.data[i] for i in raw_x_batch]
        x_batch = np.array(x_batch)
        x_batch = np.reshape(x_batch, (x_batch.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
        return x_batch

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def _load_images(self):
        files = os.listdir(IMAGES_DIR)
        files = [f for f in files if not f == 'meta' and not f.startswith('file.txt')]
        assert set(files) == {'train', 'test'}
        id = 0
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        for f in files:
            full_path = os.path.join(IMAGES_DIR, f)
            trainset = torchvision.datasets.CIFAR100(root=os.path.join('..', 'data', 'cifar100', 'data', 'raw'), train=True,
                                                     download=False, transform=transform)
            data = torch.utils.data.DataLoader(trainset,
                                        batch_size=trainset.__len__(),
                                        shuffle=False)
            images, labels = next(iter(data))
            print(labels)

            imgs_dict = self.unpickle(full_path)
            data = imgs_dict[b'data']
            print(imgs_dict[b'fine_labels'])
            print(imgs_dict[b'fine_labels'].count(70))
            final_id = id + len(imgs_dict[b'labels'])
            ids = list(range(id, final_id))
            self.save_images(data, ids)
            id += len(imgs_dict[b'labels'])
        return

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def save_images(self, data, ids):
        for i, id in enumerate(ids):
            self.data[id] = np.array(data[i])

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size