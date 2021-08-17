import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime

IMAGE_SIZE = 32

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

class LeNet5(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(LeNet5, self).__init__()
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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=['cifar100'],
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
                        default='cuda:0',
                        type=str)
    return parser.parse_args()


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

    num_classes = 100
    model_params = (args.lr, num_classes)
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Load dataset
    print("Loading dataset...")
    train_data = torchvision.datasets.CIFAR100('../../../CIFAR100/', train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=6)
    test_data = torchvision.datasets.CIFAR100('../../../CIFAR100/', train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=6)

    # Create model
    model = LeNet5(*model_params, device).to(device)

    res_path = os.path.join('results')
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=4*10**(-4))
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

    plt.plot(list(range(1, n_epochs+1)), loss, '-b', label=prefix + 'loss')
    plt.plot(list(range(1, n_epochs+1)), accuracy, '-r', label=prefix + 'accuracy')

    plt.xlabel("Epochs")
    plt.ylabel("Average accuracy")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(name + '.png')
    plt.close()


if __name__ == '__main__':
    main()