import argparse
import os
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

IMAGES_DIR = os.path.join('..', 'data', 'glv2', 'data', 'raw', 'train')
IMAGES_DIR_TEST = os.path.join('..', 'data', 'glv2', 'data', 'raw', 'test')
TRAIN_FILE_DIR = os.path.join('..', 'data', 'glv2', 'landmarks-user-160k')
TEST_FILE_DIR = os.path.join('..', 'data', 'glv2', 'landmarks-user-160k')
IMAGE_SIZE = 224

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                        help='name of dataset;',
                        type=str,
                        default='landmarks_160')
    parser.add_argument('-device',
                        type=str,
                        default='cuda')
    parser.add_argument('--train-file',
                        required=True,
                        type=str)
    parser.add_argument('--test-file',
                        required=True,
                        type=str)
    return parser.parse_args()

def load_image(img_name):
    path = os.path.join(IMAGES_DIR, img_name[0], img_name[1], img_name[2])
    # img_name = img_name + ".jpg"
    if not os.path.exists(path):
        # print("not existing path:", path)
        # return np.random.rand(3,224,224)
        return np.zeros((3, 224, 224))
    img_name = img_name + ".jpg"
    img_path = os.path.join(path, img_name)
    if not os.path.exists(img_path):
        # print("not existing img:", img_name)
        # return np.random.rand(3,224,224)
        return np.zeros((3, 224, 224))
    try:
        img = Image.open(img_path)
    except PIL.UnidentifiedImageError:
        # print("Corrupted image:",img_path)
        # return np.random.rand(3, 224, 224)
        # return None
        return np.zeros((3, 224, 224))

    # img = Image.open(os.path.join(path, img_name))
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    return input_tensor.cpu().detach().numpy()

def process_x(x_list):
    x_batch = [load_image(i) for i in x_list]
    x_batch = np.array(x_batch)
    return x_batch

def plot_score(scores, n_epochs, title, ylabel, fig_name):
    epochs = range(n_epochs)
    plt.plot(epochs, scores, 'g', label=ylabel)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    path = os.path.join('.', 'figures')
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join('.', path, fig_name)
    plt.savefig(path)

def train(model, train_dataset, labels, **kwargs):
    criterion = kwargs['criterion']
    optimizer = kwargs['optimizer']
    n_epochs = kwargs['n_epochs']
    batch_size = kwargs['batch_size']
    device = kwargs['device']

    print("Training with {:d} epochs and batch size {:d}".format(n_epochs, batch_size))
    losses = []
    eval_accuracies = []
    for epoch in range(n_epochs):
        print(f'\tEpoch [{epoch + 1}/{n_epochs}]', end='')
        train_loss = 0
        i = 0
        for k in range(0, len(train_dataset), batch_size):
            batched_x = process_x(train_dataset[k:k + batch_size])
            batched_y = labels[k:k + batch_size]
            input_data_tensor = torch.from_numpy(batched_x).type(torch.FloatTensor).to(device)
            target_data_tensor = torch.LongTensor(batched_y).to(device)
            optimizer.zero_grad()
            outputs = model(input_data_tensor)
            loss = criterion(outputs, target_data_tensor)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            i += 1
        print(f'\t train_loss: {train_loss/i:.2f}')
        losses.append(train_loss/i)
        model.eval()
        eval_accuracy, eval_loss = test(model, train_dataset, labels, batch_size=batch_size, device=device)
        print("\tEval accuracy: {:.2f}. Eval loss: {:.2f}".format(eval_accuracy, eval_loss))
        eval_accuracies.append(eval_accuracy)
        model.train()
    fig_name = 'train_loss_' + str(n_epochs) + 'epochs'
    plot_score(losses, n_epochs, 'Train loss', 'Train Loss', fig_name)
    fig_name = 'eval_accuracy_' + str(n_epochs) + 'epochs'
    plot_score(eval_accuracies, n_epochs, 'Eval accuracy', 'Evaluation Accuracy', fig_name)

def test(model, data, labels, **kwargs):
    batch_size = kwargs['batch_size']
    device = kwargs['device']
    test_loss = 0
    total = 0
    correct = 0
    for k in range(0, len(data), batch_size):
        batched_x = process_x(data[k:k + batch_size])
        batched_y = labels[k:k + batch_size]
        input_data_tensor = torch.from_numpy(batched_x).type(torch.FloatTensor).to(device)
        target_data_tensor = torch.LongTensor(batched_y).to(device)
        with torch.no_grad():
            outputs = model(input_data_tensor)
            test_loss += F.cross_entropy(outputs, target_data_tensor, reduction='sum').item()
            _, predicted = torch.max(outputs.data, 1)  # same as torch.argmax()
            total += target_data_tensor.size(0)
            correct += (predicted == target_data_tensor).sum().item()

    test_loss = test_loss / len(data)
    print(correct, total)
    return (100 * correct / total), test_loss

def main():
    args = parse_args()
    df = pd.read_csv(os.path.join(TRAIN_FILE_DIR, args.train_file))
    df_test = pd.read_csv(os.path.join(TEST_FILE_DIR, args.test_file))

    n_classes = len(set(df['class'].tolist()))
    batch_size = 32
    n_epochs = 20
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')

    print("Classification on {:d} classes".format(n_classes))

    model_path = '%s.%s' % ('glv2', 'mobilenet')
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    model = ClientModel(0.01, n_classes, device).to(device)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
    # model = nn.DataParallel(model)
    # model.module.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True).to(device)
    #
    # m2 = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
    # m2 = nn.DataParallel(m2)
    # m2.module.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True).to(device)
    # m2.load_state_dict(model.state_dict())

    # Train
    print("--- Training network ---")
    model.train()
    input_data = df['image_id'].tolist()
    labels = df['class'].tolist()
    data = list(zip(input_data, labels))
    random.shuffle(data)
    input_data, labels = zip(*data)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)
    train(model=model, train_dataset=input_data, labels=labels, criterion=criterion,
          optimizer=optimizer, n_epochs=n_epochs, batch_size=batch_size, device=device)

    # diff = sum((x - y).abs().sum() for x, y in zip(model.state_dict().values(), m2.state_dict().values()))
    # print("Modelli differenti dopo il training:", torch.is_nonzero(diff))

    print("--- Saving model ---")
    path = os.path.join('.', 'checkpoints')
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model, os.path.join(path, 'offline_model1'))

    # Test
    print("--- Testing ---")
    test_data = df_test['image_id'].tolist()
    test_labels = df_test['class'].tolist()
    model.eval()
    accuracy, loss = test(model, test_data, test_labels, batch_size=batch_size, device=device)
    print("Test accuracy: {:.2f}. Test loss: {:.2f}".format(accuracy, loss))

if __name__ == '__main__':
    main()