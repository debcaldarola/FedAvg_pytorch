import os
import argparse
from collections import defaultdict
import json
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

IMAGE_SIZE = 299
TRAIN_IMAGES_DIR = os.path.join('..', '..', 'data', 'inaturalist', 'data', 'raw', 'train')
TEST_IMAGES_DIR = os.path.join('..', '..', 'data', 'inaturalist', 'data', 'raw', 'test')

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


class MyMobileNet(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(MyMobileNet, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.lr = lr

        model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True))

        self.indexes_to_be_removed = None

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class iNaturalistDataset(Dataset):
    """ iNaturalist Dataset """

    def __init__(self, json_file, root_dir, train_transform=None, test_transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = defaultdict(lambda: None)
        userdata = defaultdict(lambda: None)
        with open(json_file, 'r') as inf:
            cdata = json.load(inf)
        userdata.update(cdata['user_data'])
        for user, user_dict in userdata.items():
            for x, y in zip(user_dict['x'], user_dict['y']):
                self.data[x] = y
        # print(len(self.data.values())) # 120300
        self.data_frame = pd.DataFrame(list(self.data.items()))
        print(self.data_frame.head(3))

        self.root_dir = root_dir
        self.train_transform = train_transform
        self.test_transform = test_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        w, h = image.size
        if w < IMAGE_SIZE or h < IMAGE_SIZE:
            return None
        label = self.data_frame.iloc[idx, 1]

        if self.train_transform:
            image = self.train_transform(image)
        elif self.test_transform:
            image = self.test_transform(image)
        return image, label


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                        help='name of dataset;',
                        type=str,
                        choices=['inaturalist'],
                        required=True)
    parser.add_argument('-model',
                        help='name of model;',
                        type=str,
                        required=True)
    parser.add_argument('--num-epochs',
                        type=int,
                        required=True)
    parser.add_argument('--batch-size',
                        type=int,
                        required=True)
    parser.add_argument('-lr',
                        type=float,
                        required=True)
    parser.add_argument('-device',
                        default='cuda',
                        type=str)
    return parser.parse_args()


def mycollate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    args = parse_args()
    model_path = '../../models/%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)
    print('############################## %s ##############################' % model_path)
    print('Offline test on one client')

    # Setup GPU
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    num_classes = 1203
    model_params = (args.lr, num_classes)
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Load dataset
    print("Loading dataset...")
    train_data = iNaturalistDataset(json_file='../../data/inaturalist/data/train/federated_train_user_120k.json',
                                    root_dir=TRAIN_IMAGES_DIR, train_transform=transform_train)
    print("Train set length:", train_data.__len__())
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=6, collate_fn=mycollate_fn)
    test_data = iNaturalistDataset(json_file='../../data/inaturalist/data/test/test.json',
                                   root_dir=TEST_IMAGES_DIR, train_transform=transform_test)
    print("Test set length:", test_data.__len__())
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=6, collate_fn=mycollate_fn)

    # Create model
    model = MyMobileNet(*model_params, device)
    model = nn.DataParallel(model).to(device)

    res_path = os.path.join('results')
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=4 * 10 ** (-5))
    num_epochs = args.num_epochs
    print('num_epochs:', num_epochs)

    print("--- Training model ---")
    start_time = datetime.now()
    current_time = start_time.strftime("%m%d%y_%H:%M:%S")

    file = os.path.join(res_path, args.model + '_' + str(num_epochs) + 'epochs_' + current_time + '.txt')
    fp = open(file, "w")

    train_losses, train_acc = train_net(model, trainloader, num_epochs, optimizer, criterion, device, fp)
    figpath = os.path.join('plots')
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    figname = 'eval_' + current_time
    plot_metrics(train_acc, train_losses, num_epochs, figname, figpath, 'Evaluation of centralized model training')

    print("--- Testing model ---")
    test_loss, accuracy = test_net(model, testloader, device)
    print("Loss: {:.3f}, Accuracy: {:.3f}".format(test_loss, accuracy))
    fp.write("Loss: {:.3f}, Accuracy: {:.3f}\n".format(test_loss, accuracy))

    print("--- Saving model ---")
    ckpt_path = os.path.join('checkpoints')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = os.path.join(ckpt_path, '{}.ckpt'.format(args.model + '_' + str(num_epochs) + 'epochs_' + current_time))
    torch.save(model.state_dict(), save_path)
    print("Model saved in", save_path)


def train_net(model, trainloader, num_epochs, optimizer, criterion, device, fp):
    eval_losses = np.empty(num_epochs)
    eval_accuracy = np.empty(num_epochs)
    j = 0
    # Train model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # gradient inside the optimizer (memory usage increases here)
            running_loss += loss.item()
            optimizer.step()
            torch.cuda.empty_cache()

        eval_losses[j] = running_loss / i
        print("Epoch {}/{}, Loss: {:.3f}".format(epoch + 1, num_epochs, eval_losses[j]))
        fp.write("Epoch {}/{}, Loss: {:.3f}\n".format(epoch + 1, num_epochs, eval_losses[j]))

        print("\tEvaluating model...")
        eval_loss, eval_acc = test_net(model, trainloader, device)
        eval_accuracy[j] = eval_acc
        print("Epoch {}/{}, Accuracy: {:.3f}".format(epoch + 1, num_epochs, eval_accuracy[j]))
        fp.write("Epoch {}/{}, Accuracy: {:.3f}\n".format(epoch + 1, num_epochs, eval_accuracy[j]))

        j += 1
    return eval_losses, eval_accuracy


def test_net(model, testloader, device):
    model.eval()
    correct = 0
    test_loss = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = model(images)
            test_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs.data, 1)  # same as torch.argmax()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss /= total
    return test_loss, accuracy


def plot_metrics(accuracy, loss, n_epochs, figname, figpath, title, prefix='val_'):
    name = os.path.join(figpath, figname)

    plt.plot(list(range(1, n_epochs + 1)), loss, '-b', label=prefix + 'loss')
    plt.plot(list(range(1, n_epochs + 1)), accuracy, '-r', label=prefix + 'accuracy')

    plt.xlabel("Epochs")
    plt.ylabel("Average accuracy")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(name + '.png')
    plt.close()


if __name__ == '__main__':
    main()
