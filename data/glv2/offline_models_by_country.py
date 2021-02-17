import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

IMAGES_DIR = os.path.join('.', 'data', 'raw', 'train')
TRAIN_FILE_DIR = os.path.join('.', 'landmarks-user-160k')
TEST_FILE_DIR = os.path.join('.', 'landmarks-user-160k')
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
    parser.add_argument('--only-eval',
                        type=bool,
                        default=False)
    return parser.parse_args()

def load_image(img_name):
    path = os.path.join(IMAGES_DIR, img_name[0], img_name[1], img_name[2])
    img_name = img_name + ".jpg"
    img = Image.open(os.path.join(path, img_name))
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

def get_countries_and_domains(df):
    domains = sorted(set(df['domain'].tolist()))
    countries = []
    for d in domains:
        country = df[df['domain'] == d]['country'].tolist()[0]
        countries.append(country)
    return countries, domains

def train(model, train_dataset, labels, **kwargs):
    criterion = kwargs['criterion']
    optimizer = kwargs['optimizer']
    n_epochs = kwargs['n_epochs']
    batch_size = kwargs['batch_size']
    device = kwargs['device']

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
    print(correct, total)
    test_loss = test_loss / len(data)
    test_acc = 100 * correct / total
    return test_acc, test_loss

def save_models(models, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for i, model in enumerate(models):
        name = 'model' + '_' + str(i)
        model_path = os.path.join(path, name)
        torch.save(model, model_path)
    return

def main():
    args = parse_args()
    df = pd.read_csv(os.path.join(TRAIN_FILE_DIR, args.train_file))

    n_classes = len(set(df['class'].tolist()))
    n_models = len(set(df['country'].tolist()))
    batch_size = 32
    n_epochs = 20
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')

    countries, domains = get_countries_and_domains(df)

    if args.only_eval:
        df_results = pd.read_csv('offline_test_by_country.csv')
        df_results['eval_accuracy'] = ""
        df_results['eval_loss'] = ""
        df_results['eval_n_images'] = ""
        # Load trained models
        print("--- Loading {:d} models ---".format(n_models))
        models = []
        load_path = os.path.join('preprocess', 'checkpoints')
        for i in range(n_models):
            chkpt = 'model' + '_' + str(i)
            model_path = os.path.join(load_path, chkpt)
            m = torch.load(model_path).to(device)
            models.append(m)

        print("--- Evaluating models ---")
        tot_acc = 0
        tot_loss = 0
        for i, domain in enumerate(domains):
            print(countries[i], "({:d})".format(domain))
            input_data = df[df['domain'] == domain]['image_id'].tolist()
            labels = df[df['domain'] == domain]['class'].to_numpy()
            models[i].eval()
            accuracy, loss = test(models[i], input_data, labels, batch_size=batch_size, device=device)
            tot_acc += accuracy
            tot_loss += loss
            print("\tEval accuracy: {:.2f}. Eval loss: {:.2f}".format(accuracy, loss))
            df_results.loc[(df_results['country'] == countries[i]), 'eval_accuracy'] = accuracy
            df_results.loc[(df_results['country'] == countries[i]), 'eval_loss'] = loss
            df_results.loc[(df_results['country'] == countries[i]), 'eval_n_images'] = len(labels)

        df_results.to_csv('offline_models_by_country_with_eval.csv')
        return

    models = []
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.module.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True).to(device)

    for i in range(n_models):
        m = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
        m = nn.DataParallel(m, device_ids=[0, 1, 2, 3])
        m.module.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True).to(device)
        m.load_state_dict(model.state_dict())
        models.append(m)

    # Train
    for i, domain in enumerate(domains):
        print("Model {:d} ({:s})".format(i, countries[i]))
        models[i].train()
        input_data = df[df['domain'] == domain]['image_id'].tolist()
        labels = df[df['domain'] == domain]['class'].to_numpy()
        optimizer = optim.SGD(models[i].parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss().to(device)
        train(model=models[i], train_dataset=input_data, labels=labels, criterion=criterion,
              optimizer=optimizer, n_epochs=n_epochs, batch_size=batch_size, device=device)

    # Test
    df_test = pd.read_csv(os.path.join(TEST_FILE_DIR, args.test_file))
    tot_acc = 0
    tot_loss = 0
    results = pd.DataFrame(columns=['country', 'accuracy', 'loss', 'n_images'])
    results['country'] = pd.Series(countries)
    print("--- Testing ---")
    for i, domain in enumerate(domains):
        if df_test[df_test['domain'] == domain].empty:
            print("No data available for domain {:d} ({:s})".format(domain, countries[i]))
            results.loc[(results['country'] == countries[i]), 'accuracy'] = None
            results.loc[(results['country'] == countries[i]), 'loss'] = None
            continue
        print(countries[i], "({:d})".format(domain))
        test_data = df_test[df_test['domain'] == domain]['image_id'].tolist()
        test_labels = df_test[df_test['domain'] == domain]['class'].tolist()
        models[i].eval()
        accuracy, loss = test(models[i], test_data, test_labels, batch_size=batch_size, device=device)
        tot_acc += accuracy
        tot_loss += loss
        print("\tTest accuracy: {:.2f}. Test loss: {:.2f}".format(accuracy, loss))
        results.loc[(results['country'] == countries[i]), 'accuracy'] = accuracy
        results.loc[(results['country'] == countries[i]), 'loss'] = loss
        results.loc[(results['country'] == countries[i]), 'n_images'] = len(test_labels)

    print("Average accuracy: {:.2f}. Average loss: {:.2f}".format(tot_acc/len(models), tot_loss/len(models)))

    path = os.path.join('preprocess', 'checkpoints')
    save_models(models, path)
    results.to_csv('offline_test_by_country.csv', index=False)

if __name__ == '__main__':
    main()