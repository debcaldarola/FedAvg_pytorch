import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from PIL import Image
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from torchvision import transforms

IMAGES_DIR = os.path.join('.', 'data', 'raw', 'train')
IMAGES_DIR_TEST = os.path.join('.', 'data', 'raw', 'test')
TRAIN_FILE_DIR = os.path.join('.', 'landmarks-user-160k')
TEST_FILE_DIR = os.path.join('.', 'landmarks-user-160k')
IMAGE_SIZE = 224

class myResidualModel(nn.Module):
    def __init__(self, general_model, models, n_classes, device):
        super(myResidualModel, self).__init__()
        self.alpha = Parameter(torch.zeros(1))
        self.general = general_model
        self.models = models
        self.n_classes = n_classes
        self.device = device

    def forward(self, x, domains):
        # output size: (batch_size, n_classes) = (x.size()[0], 2028)
        output = torch.zeros((x.size()[0]), self.n_classes).to(self.device)
        for i, img in enumerate(x):
            output[i] = self.general(img.unsqueeze(0)) + self.alpha * self.models[domains[i]](img.unsqueeze(0))
        return output

    def start_training(self):
        self.general.train()
        for m in self.models:
            m.train()

    def start_eval(self):
        self.general.eval()
        for m in self.models:
            m.eval()

def parse_args():
    parser = argparse.ArgumentParser()
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
    img_name = img_name + ".jpg"
    img = Image.open(os.path.join(path, img_name))
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

def get_countries_and_domains(df):
    domains = sorted(set(df['domain'].tolist()))
    countries = []
    cleaned_domains = []
    removed_countries = []
    for d in domains:
        country = df[df['domain'] == d]['country'].tolist()[0]
        n_classes = len(set(df[df['country'] == country]['class']))
        if n_classes == 1:
            removed_countries.append(country)
            continue
        countries.append(country)
        cleaned_domains.append(d)
    print("Removed countries ({:d}):".format(len(removed_countries)), removed_countries)
    return countries, cleaned_domains

def train(model, domains, train_dataset, labels, **kwargs):
    criterion = kwargs['criterion']
    optimizer = kwargs['optimizer']
    n_epochs = kwargs['n_epochs']
    batch_size = kwargs['batch_size']
    device = kwargs['device']

    losses = []

    for epoch in range(n_epochs):
        print(f'\tEpoch [{epoch + 1}/{n_epochs}]', end='')
        train_loss = 0
        i = 0
        for k in range(0, len(train_dataset), batch_size):
            batched_x = process_x(train_dataset[k:k + batch_size])
            batched_y = labels[k:k + batch_size]
            batched_d = domains[k:k + batch_size]
            input_data_tensor = torch.from_numpy(batched_x).type(torch.FloatTensor).to(device)
            target_data_tensor = torch.LongTensor(batched_y).to(device)
            optimizer.zero_grad()
            outputs = model(input_data_tensor, batched_d)
            loss = criterion(outputs, target_data_tensor)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            i += 1
        print(f'\t train_loss: {train_loss/i:.2f}')
        losses.append(train_loss/i)

    return losses

def test(model, domain, data, labels, **kwargs):
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
        domains = [domain] * batch_size
        with torch.no_grad():
            outputs = model(input_data_tensor, domains)
            test_loss += F.cross_entropy(outputs, target_data_tensor, reduction='sum').item()
            _, predicted = torch.max(outputs.data, 1)  # same as torch.argmax()
            total += target_data_tensor.size(0)
            correct += (predicted == target_data_tensor).sum().item()

    test_loss = test_loss / len(data)
    print(correct, total)
    return (100 * correct / total), test_loss

def main():
    args = parse_args()
    df_train = pd.read_csv(os.path.join(TRAIN_FILE_DIR, args.train_file))
    df_test = pd.read_csv(os.path.join(TEST_FILE_DIR, args.test_file))

    print("--- Loading models ---")
    n_models = len(set(df_train['country'].tolist()))
    n_classes = len(set(df_train['class'].tolist()))
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')
    batch_size = 32
    n_epochs = 20

    countries, domains = get_countries_and_domains(df_train)

    models = []
    general_model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
    general_model.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True).to(device)
    general_model = nn.DataParallel(general_model, device_ids=[0, 1, 2, 3])

    for i in range(n_models):
        m = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
        m.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True).to(device)
        m = nn.DataParallel(m, device_ids=[0, 1, 2, 3])
        m.load_state_dict(general_model.state_dict())
        models.append(m)

    model = myResidualModel(general_model, models, n_classes, device).to(device)

    results = pd.DataFrame(columns=['country', 'test_accuracy', 'test_loss', 'test_n_images', 'eval_accuracy',
                                    'eval_loss', 'eval_n_images'])
    results['country'] = pd.Series(countries)

    # Train
    model.train()
    model.start_training()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)
    # avg_losses = np.zeros(n_epochs)
    print("--- Training with residuals ---")
    input_data = df_train['image_id'].tolist()
    labels = df_train['class'].tolist()
    train_domains = df_train['domain'].tolist()

    z = list(zip(input_data, labels, train_domains))
    random.shuffle(z)
    input_data, labels, train_domains = zip(*z)

    train(model, train_domains, input_data, labels, criterion=criterion, optimizer=optimizer, n_epochs=n_epochs,
         batch_size=batch_size, device=device)

    print("--- Evaluating model --- ")
    model.eval()
    model.start_eval()

    for i, domain in enumerate(domains):
        print("Model {:d} ({:s})".format(i, countries[i]))
        input_data = df_train[df_train['domain'] == domain]['image_id'].tolist()
        labels = df_train[df_train['domain'] == domain]['class'].to_numpy()
        eval_acc, eval_loss = test(model, i, input_data, labels, batch_size=batch_size, device=device)
        results.loc[(results['country'] == countries[i]), 'eval_accuracy'] = eval_acc
        results.loc[(results['country'] == countries[i]), 'eval_loss'] = eval_loss
        results.loc[(results['country'] == countries[i]), 'eval_n_images'] = len(labels)
        print("\tEvaluation accuracy: {:.2f}. Test loss: {:.2f}".format(eval_acc, eval_loss))

    # for i, domain in enumerate(domains):
    #     print("Model {:d} ({:s})".format(i, countries[i]))
    #     input_data = df_train[df_train['domain'] == domain]['image_id'].tolist()
    #     labels = df_train[df_train['domain'] == domain]['class'].to_numpy()
    #     losses = train(model=model, domain=i, train_dataset=input_data, labels=labels, criterion=criterion,
    #                 optimizer=optimizer, n_epochs=n_epochs, batch_size=batch_size, device=device)
    #     for k in range(n_epochs):
    #         avg_losses[k] += losses[k]
    #
    #     model.eval()
    #     model.start_eval()
    #     eval_acc, eval_loss = test(model, i, input_data, labels, batch_size=batch_size, device=device)
    #     print("\tEvaluation accuracy: {:.2f}. Test loss: {:.2f}".format(eval_acc, eval_loss))
    #     results.loc[(results['country'] == countries[i]), 'eval_accuracy'] = eval_acc
    #     results.loc[(results['country'] == countries[i]), 'eval_loss'] = eval_loss
    #     results.loc[(results['country'] == countries[i]), 'eval_n_images'] = len(labels)
    #
    #     model.train()
    #     model.start_training()

    # for k in range(n_epochs):
    #     avg_losses[k] = avg_losses[k]/n_models

    print("--- Saving model ---")
    path = os.path.join('.', 'checkpoints', 'residuals')
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model, os.path.join(path, 'model_with_residuals_e1'))

    # plot_score(avg_losses, n_epochs, title='Average of country-specific models with residuals train losses',
    #            ylabel='avg_train_loss', fig_name='avg_train_loss')

    # Test
    tot_acc = 0
    tot_loss = 0
    print("--- Testing with residuals ---")
    model.eval()
    model.start_eval()
    for i, domain in enumerate(domains):
        if df_test[df_test['domain'] == domain].empty:
            print("No data available for domain {:d} ({:s})".format(domain, countries[i]))
            results.loc[(results['country'] == countries[i]), 'accuracy'] = None
            results.loc[(results['country'] == countries[i]), 'loss'] = None
            continue
        print(countries[i], "({:d})".format(domain))
        test_data = df_test[df_test['domain'] == domain]['image_id'].tolist()
        test_labels = df_test[df_test['domain'] == domain]['class'].tolist()
        accuracy, loss = test(model, i, test_data, test_labels, batch_size=batch_size, device=device)
        tot_acc += accuracy
        tot_loss += loss
        print("\tTest accuracy: {:.2f}. Test loss: {:.2f}".format(accuracy, loss))
        results.loc[(results['country'] == countries[i]), 'test_accuracy'] = accuracy
        results.loc[(results['country'] == countries[i]), 'test_loss'] = loss
        results.loc[(results['country'] == countries[i]), 'test_n_images'] = len(test_labels)

    print("Average accuracy: {:.2f}. Average loss: {:.2f}".format(tot_acc / len(models), tot_loss / len(models)))
    print("Alpha: ", model.alpha)

    results.to_csv('offline_test_by_country_with_residuals_e1.csv', index=False)

if __name__ == '__main__':
    main()