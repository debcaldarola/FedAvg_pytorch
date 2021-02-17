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
from .offline_models_by_country import load_image, process_x, get_countries_and_domains, train, test, save_models

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
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(os.path.join(TRAIN_FILE_DIR, args.train_file))

    n_classes = len(set(df['class'].tolist()))
    n_models = len(set(df['country'].tolist()))
    batch_size = 32
    n_epochs = 20
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')

    countries, domains = get_countries_and_domains(df)

    models = []
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    for c in countries:
        m = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
        m.load_state_dict(model.state_dict())
        country_classes = len(set(df[df['country'] == c]['class']))
        m.module.classifier[1] = nn.Linear(in_features=1280, out_features=country_classes, bias=True).to(device)
        m = nn.DataParallel(m, device_ids=list(range(torch.cuda.device_count())))
        models.append(m)

    results = pd.DataFrame(columns=['country', 'eval_accuracy', 'eval_loss', 'eval_n_images', 'test_accuracy',
                                    'test_loss', 'test_n_images', ])
    results['country'] = pd.Series(countries)

    # Train
    for i, domain in enumerate(domains):
        print("Model {:d} ({:s})".format(i, countries[i]))
        print("\tTraining...")
        models[i].train()
        input_data = df[df['domain'] == domain]['image_id'].tolist()
        labels = df[df['domain'] == domain]['class'].to_numpy()
        optimizer = optim.SGD(models[i].parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss().to(device)
        train(model=models[i], train_dataset=input_data, labels=labels, criterion=criterion,
              optimizer=optimizer, n_epochs=n_epochs, batch_size=batch_size, device=device)
        print("\tEvaluating...")
        models[i].eval()
        accuracy, loss = test(models[i], input_data, labels, batch_size=batch_size, device=device)
        print("\tEval accuracy: {:.2f}. Eval loss: {:.2f}".format(accuracy, loss))
        results.loc[(results['country'] == countries[i]), 'eval_accuracy'] = accuracy
        results.loc[(results['country'] == countries[i]), 'eval_loss'] = loss
        results.loc[(results['country'] == countries[i]), 'eval_n_images'] = len(labels)

    # Test
    df_test = pd.read_csv(os.path.join(TEST_FILE_DIR, args.test_file))
    tot_acc = 0
    tot_loss = 0
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
        results.loc[(results['country'] == countries[i]), 'test_accuracy'] = accuracy
        results.loc[(results['country'] == countries[i]), 'test_loss'] = loss
        results.loc[(results['country'] == countries[i]), 'test_n_images'] = len(test_labels)

    print("Average accuracy: {:.2f}. Average loss: {:.2f}".format(tot_acc / len(models), tot_loss / len(models)))

    path = os.path.join('preprocess', 'checkpoints')
    save_models(models, path)
    results.to_csv('offline_test_by_country.csv', index=False)

if __name__ == '__main__':
    main()