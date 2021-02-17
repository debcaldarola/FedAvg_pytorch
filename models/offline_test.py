import importlib
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.args import parse_args
from baseline_constants import MODEL_PARAMS
from utils.model_utils import batch_data


def main():
    args = parse_args()
    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)
    print('############################## %s ##############################' % model_path)
    print('Offline test on one client')

    # Setup GPU
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Load dataset
    print("Loading dataset...")
    train, test = setup_dataset(args.dataset)

    # Create model
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')
    model = ClientModel(*model_params, device).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs
    print('num_epochs:', num_epochs)
    print("--- Training model ---")
    model = train_net(model, train, num_epochs, optimizer, criterion, args.batch_size, args.seed, device)

    print("--- Testing model ---")
    test_loss, accuracy = test_net(model, test, device, args.batch_size, args.seed)
    print("Loss: {:.3f}, Accuracy: {:.3f}".format(test_loss, accuracy))

    print("--- Saving model ---")
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = os.path.join(ckpt_path, '{}.ckpt'.format('offline_'+args.model))
    torch.save(model.state_dict(), save_path)
    print("Model saved in", save_path)

def setup_dataset(dataset, use_val_set=False):
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)
    train_set = create_dataset(train_data_dir)
    test_set = create_dataset(test_data_dir)
    return train_set, test_set


def create_dataset(dir_path):
    """ Returns dataset with the format {'x': [list of images pixels], 'y': [labels]} """

    dataset = {'x': [], 'y': []}
    files = os.listdir(dir_path)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(dir_path, f)
        with open(file_path, 'r') as file_content:
            data = json.load(file_content)
            users = data['user_data']
            for u in users:
                images = data['user_data'][u]['x']
                labels = data['user_data'][u]['y']
                for i in range(len(labels)):
                    dataset['x'].append(images[i])
                    dataset['y'].append(labels[i])
    return dataset


def train_net(model, train, num_epochs, optimizer, criterion, batch_size, seed, device):
    losses = np.empty(num_epochs)
    j = 0

    # Train model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        i = 0
        for batched_x, batched_y in batch_data(train, batch_size, seed=seed):
            input_data = model.process_x(batched_x)
            target_data = model.process_y(batched_y)
            input_data_tensor = torch.from_numpy(input_data).permute(0, 3, 1, 2).to(device)
            target_data_tensor = torch.LongTensor(target_data).to(device)
            optimizer.zero_grad()
            outputs = model(input_data_tensor)
            loss = criterion(outputs, target_data_tensor)  # loss between the prediction and ground truth (labels)
            loss.backward()  # gradient inside the optimizer (memory usage increases here)
            running_loss += loss.item()
            optimizer.step()  # update of weights
            i += 1
            torch.cuda.empty_cache()
            # print('Current loss:', running_loss/i)
        losses[j] = running_loss / i
        print("Epoch {}/{}, Loss: {:.3f}".format(epoch + 1, num_epochs, losses[j]))
        j += 1
    plot_score(losses, num_epochs, 'Offline model CelebA - Train loss', 'train_loss', 'celeba_offline_train_loss')
    return model


def test_net(model, test, device, batch_size, seed):
    model.eval()
    correct = 0
    test_loss = 0
    total = 0
    for batched_x, batched_y in batch_data(test, batch_size, seed=seed):
        input = model.process_x(batched_x)
        labels = model.process_y(batched_y)
        input_tensor = torch.from_numpy(input).permute(0, 3, 1, 2).to(device)
        labels_tensor = torch.LongTensor(labels).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            test_loss += F.cross_entropy(outputs, labels_tensor, reduction='sum').item()
            _, predicted = torch.max(outputs.data, 1)  # same as torch.argmax()
            total += labels_tensor.size(0)
            correct += (predicted == labels_tensor).sum().item()
    accuracy = 100 * correct / total
    test_loss /= total
    return test_loss, accuracy

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

if __name__ == '__main__':
    main()
