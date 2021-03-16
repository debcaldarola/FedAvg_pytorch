import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

IMAGE_SIZE = 32
IMAGES_DIR = os.path.join('..', 'data', 'cifar100', 'data', 'raw', 'img')

transform_train = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# Normalize the test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.lr = lr

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*5*5, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, self.num_classes)
        )

        self.size = self.model_size()


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

    def process_x(self, raw_x_batch):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def _load_image(self, img_name):
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        if self.training:
            img = transform_train(img)
        else:
            img = transform_test(img)
        img = img.cpu().detach().numpy()
        return img

    # def _load_images(self):
    #     files = os.listdir(IMAGES_DIR)
    #     files = [f for f in files if not f == 'meta' and not f.startswith('file.txt')]
    #     assert set(files) == {'train', 'test'}
    #     id = 0
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(),
    #          transforms.Normalize((0.5), (0.5))])
    #     for f in files:
    #         full_path = os.path.join(IMAGES_DIR, f)
    #         # trainset = torchvision.datasets.CIFAR100(root=os.path.join('..', 'data', 'cifar100', 'data', 'raw'), train=True,
    #         #                                          download=False, transform=transform)
    #         # data = torch.utils.data.DataLoader(trainset,
    #         #                             batch_size=trainset.__len__(),
    #         #                             shuffle=False)
    #         # images, labels = next(iter(data))
    #         # print(labels)
    #
    #         imgs_dict = self.unpickle(full_path)
    #         print(imgs_dict.keys())
    #         data = imgs_dict[b'data']
    #         names = imgs_dict[b'filenames']
    #         labels = imgs_dict[b'fine_labels']
    #         # res = filter(lambda x: x.endswith(b'_000043.png'), names)
    #         # for r in res:
    #         #     print(r)
    #         for i in range(len(names)):
    #             if labels[i] == 1:
    #                 print(labels[i], names[i])
    #         # print(imgs_dict[b'fine_labels'][:10])
    #         # print(imgs_dict[b'filenames'][:10])
    #         # print(imgs_dict[b'filenames'])
    #         # print(imgs_dict[b'fine_labels'].count(70))
    #         # final_id = id + len(imgs_dict[b'labels'])
    #         # ids = list(range(id, final_id))
    #         # self.save_images(data, ids)
    #         # id += len(imgs_dict[b'labels'])
    #     return
    #
    # def unpickle(self, file):
    #     with open(file, 'rb') as fo:
    #         dict = pickle.load(fo, encoding='bytes')
    #     return dict

    def save_images(self, data, ids):
        for i, id in enumerate(ids):
            self.data[id] = np.array(data[i])

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size