import importlib
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from models.utils.args import parse_args
from models.baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from models.utils.model_utils import batch_data


def main():
    args = parse_args()
    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)
    print('############################## %s ##############################' % model_path)
    print('Offline test of one client')
    # Setup GPU
    device = torch.device('cuda:0')

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
    model = ClientModel(*model_params, device)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available:
        model = model.to(device)
        criterion = criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs
    print('num_epochs:', num_epochs)

    model = train_net(model, train, num_epochs, optimizer, criterion, args.batch_size, args.seed, device)
    test_loss, accuracy = test_net(model, test, device)
    print("Loss: {:.3f}, Accuracy: {:.3f}".format(test_loss, accuracy))


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
                images = data[u]['x']
                labels = data[u]['y']
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
            input_data_tensor = torch.from_numpy(input_data).permute(0, 3, 1, 2)
            target_data_tensor = torch.LongTensor(target_data)
            if torch.cuda.is_available:
                input_data_tensor = input_data_tensor.to(device)
                target_data_tensor = target_data_tensor.to(device)
            optimizer.zero_grad()
            outputs = model(input_data_tensor)
            loss = criterion(outputs, target_data_tensor)  # loss between the prediction and ground truth (labels)
            loss.backward()  # gradient inside the optimizer (memory usage increases here)
            running_loss += loss.item()
            optimizer.step()  # update of weights
            i += 1
            torch.cuda.empty_cache()
        losses[j] = running_loss / i
        print("Epoch {}/{}, Loss: {:.3f}".format(epoch + 1, num_epochs, losses[j]))
        j += 1
    return model


def test_net(model, test, device):
    model.eval()
    correct = 0
    test_loss = 0
    input = model.process_x(test['x'])
    labels = model.process_y(test['y'])
    input_tensor = torch.from_numpy(input).permute(0, 3, 1, 2)
    labels_tensor = torch.LongTensor(labels)
    if torch.cuda.is_available:
        input_tensor = input_tensor.to(device)
        labels_tensor = labels_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        test_loss += F.cross_entropy(outputs, labels_tensor, reduction='sum').item()
        _, predicted = torch.max(outputs.data, 1)  # same as torch.argmax()
        total = labels_tensor.size(0)
        correct += (predicted == labels_tensor).sum().item()
    accuracy = 100 * correct / total
    test_loss /= total
    return test_loss, accuracy


if __name__ == '__main__':
    main()
