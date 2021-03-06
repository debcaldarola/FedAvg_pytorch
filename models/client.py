import copy
import random
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

from utils.model_utils import batch_data
from baseline_constants import ACCURACY_KEY


class Client:

    def __init__(self, seed, client_id, lr, weight_decay, group=None, train_data={'x': [], 'y': []}, eval_data={'x': [], 'y': []},
                 model=None, device=None):
        self._model = model
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.seed = seed
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

    def train(self, num_epochs=1, batch_size=10, minibatch=None):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            num_samples: number of samples used in training
            update: state dictionary of the trained model
        """
        if minibatch is None:
            data = self.train_data
            num_data = batch_size
        else:
            frac = min(1.0, minibatch)
            num_data = max(1, int(frac * len(self.train_data["x"])))
            xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
            data = {'x': xs, 'y': ys}
            # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
            num_epochs = 1

        # Train model
        criterion = nn.CrossEntropyLoss().to(self.device)  # it already does softmax computation
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        losses = np.empty(num_epochs)
        j = 0
        # print("initial model: ", list(self.model.state_dict().values())[0][0])

        for epoch in range(num_epochs):
            self.model.train()
            losses[j] = self.run_epoch(data, num_data, optimizer, criterion)
            j += 1

        self.losses = losses
        num_train_samples = len(data['y'])

        update = self.model.state_dict()
        return num_train_samples, update

    def run_epoch(self, data, batch_size, optimizer, criterion):
        """Runs single training epoch of self.model on client's data.

        Return:
            epoch loss
        """
        running_loss = 0.0
        i = 0
        for batched_x, batched_y in batch_data(data, batch_size, seed=self.seed):
            if isinstance(self.model, nn.DataParallel):
                input_data = self.model.module.process_x(batched_x)
                target_data = self.model.module.process_y(batched_y)
            else:
                input_data = self.model.process_x(batched_x)
                target_data = self.model.process_y(batched_y)

            if input_data.size == 0:
                continue

            input_data_tensor = torch.from_numpy(input_data).type(torch.FloatTensor).to(self.device)
            target_data_tensor = torch.LongTensor(target_data).to(self.device)
            optimizer.zero_grad()
            outputs = self.model(input_data_tensor)
            # print(outputs, target_data_tensor)
            loss = criterion(outputs, target_data_tensor)  # loss between the prediction and ground truth (labels)
            loss.backward()  # gradient inside the optimizer (memory usage increases here)
            # _, predicted = torch.max(outputs.data, 1)
            # print(predicted, target_data_tensor)
            running_loss += loss.item()
            optimizer.step()  # update of weights
            i += 1
            # torch.cuda.empty_cache()
        if i == 0:
            print("Not running epoch", self.id)
            return 0
        return running_loss / i

    def test(self, batch_size, set_to_use='test'):
        """Tests self.model on self.test_data.

        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data

        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        for batched_x, batched_y in batch_data(data, batch_size, self.seed):
            if isinstance(self.model, nn.DataParallel):
                input = self.model.module.process_x(batched_x)
                labels = self.model.module.process_y(batched_y)
                # input, labels = self.model.module.process_batch(batched_x, batched_y, train=False)
            else:
                input = self.model.process_x(batched_x)
                labels = self.model.process_y(batched_y)

            if input.size == 0:
                continue

            # print(input.shape) [64,3,32,32]
            input_tensor = torch.from_numpy(input).type(torch.FloatTensor).to(self.device)
            labels_tensor = torch.LongTensor(labels).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                test_loss += F.cross_entropy(outputs, labels_tensor, reduction='sum').item()
                _, predicted = torch.max(outputs.data, 1)  # same as torch.argmax()
                # print(predicted, labels_tensor)
                total += labels_tensor.size(0)
                correct += (predicted == labels_tensor).sum().item()
        if total == 0:
            accuracy = 0
            test_loss = 0
        else:
            accuracy = 100 * correct / total
            test_loss /= total
        return {ACCURACY_KEY: accuracy, 'loss': test_loss}

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0
        if self.eval_data is not None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
