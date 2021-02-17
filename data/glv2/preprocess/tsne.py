import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

IMAGES_DIR = os.path.join('..', 'data', 'raw', 'train')
TRAIN_FILE_DIR = os.path.join('..', 'landmarks-user-160k')
IMAGE_SIZE = 224
REMOVE_NOISE = True
COMMON_WEIGHTS_ONLY = False
ONLY_SPECIFIC = False

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
    parser.add_argument('--load-models',
                        type=bool,
                        default=False)
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

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + .02, point['y'], str(point['val']))

def plot_tsne(X_embedded, hue, title, path, countries):
    df_tsne = pd.DataFrame()
    df_tsne['tsne-2d-one'] = X_embedded[:, 0]
    df_tsne['tsne-2d-two'] = X_embedded[:, 1]
    plt.figure(figsize=(30, 30))
    ax = sns.scatterplot(x="tsne-2d-one", y="tsne-2d-two", hue=hue, data=df_tsne, palette="Set2", legend=False,
                    sizes=(40,40))
    label_point(df_tsne['tsne-2d-one'], df_tsne['tsne-2d-two'], pd.Series(countries), ax)

    title = title + '.png'
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join('.', path, title)
    plt.savefig(path)

def save_models(models, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for i, model in enumerate(models):
        name = 'model' + '_' + str(i)
        model_path = os.path.join(path, name)
        torch.save(model, model_path)
    return

def get_countries_and_domains(df):
    domains = sorted(set(df['domain'].tolist()))
    countries = []
    for d in domains:
        country = df[df['domain'] == d]['country'].tolist()[0]
        countries.append(country)
    return countries, domains

def get_models_embeddings(models, domains, df):
    models_embeddings = []
    # concatenate model params in one tensor
    for domain, m in zip(domains, models):
        model_tensors = []
        for key, tensor in m.state_dict().items():
            if ONLY_SPECIFIC and key != 'module.classifier.1.weight' and key != 'module.classifier.1.bias':
                continue
            if COMMON_WEIGHTS_ONLY and (key == 'module.classifier.1.weight' or key == 'module.classifier.1.bias'):
                continue    # delete domain-specific weights
            if (ONLY_SPECIFIC or REMOVE_NOISE) and key == 'module.classifier.1.weight':
                classes = list(set(df[df['domain'] == domain]['class']))
                mask = list(range(0, 2028))
                for c in classes:
                    mask.remove(c)
                # tensor = torch.index_select(tensor, dim=0, index=torch.tensor(classes).to('cuda')) # remove noise from classifier
                tensor[mask, :] = 0
            model_tensors.append(tensor.view(-1))  # reshape tensors
        models_embeddings.append(torch.cat(model_tensors, dim=0).cpu().detach().numpy())
    return models_embeddings

def main():
    args = parse_args()
    df = pd.read_csv(os.path.join(TRAIN_FILE_DIR, args.train_file))

    # Set number of models = number of distinct countries
    n_models = len(set(df['country'].tolist()))
    n_classes = len(set(df['class'].tolist()))
    print("Tot models: ", n_models)
    print("Tot domains: ", len(set(df['domain'].tolist())))

    batch_size = 32
    n_epochs = 5
    countries, domains = get_countries_and_domains(df)
    print("Domains: ", domains)
    print("Countries: ", countries)
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')
    # print(args.load_models)
    models = []
    if args.load_models:
        print("--- Loading {:d} models ---".format(n_models))
        load_path = os.path.join('.', 'checkpoints')
        for i in range(n_models):
            chkpt = 'model' + '_' + str(i)
            model_path = os.path.join(load_path, chkpt)
            m = torch.load(model_path).to(device)
            models.append(m)
    else:
        # Init models (in the same way)
        model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model.module.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True).to(device)

        for i in range(n_models):
            m = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True).to(device)
            m = nn.DataParallel(m, device_ids=[0, 1, 2, 3])
            m.module.classifier[1] = nn.Linear(in_features=1280, out_features=n_classes, bias=True).to(device)
            m.load_state_dict(model.state_dict())
            models.append(m)

        # Training
        for i, domain in enumerate(domains):
            print("Model {:d} ({:s})".format(i, countries[i]))
            models[i].train()
            input_data = df[df['domain'] == domain]['image_id'].tolist()
            labels = df[df['domain'] == domain]['class'].to_numpy()
            optimizer = optim.SGD(models[i].parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss().to(device)
            train(model=models[i], train_dataset=input_data, labels=labels, criterion=criterion,
                  optimizer=optimizer, n_epochs=n_epochs, batch_size=batch_size, device=device)

    # tSNE
    print("--- Embedding models and computing tSNE ---")
    models_embeddings = get_models_embeddings(models, domains, df)
    X_embedded = TSNE(n_components=2, verbose=1, perplexity=5, learning_rate=200).fit_transform(models_embeddings)
    plot_tsne(X_embedded, hue=domains, title='NEWtsne_lr200_p5_wonoise_e20', path='tSNE_plots', countries=countries)

    # Save trained models
    if not args.load_models:
        save_path = os.path.join('.', 'models')
        save_models(models, save_path)

    # Init tensorboard
    # writer = SummaryWriter(comment='glv2_tsne')  # https://github.com/lanpa/tensorboardX/blob/master/examples/demo_embedding.py
    # writer.add_embedding() -> vedi link sito
    # writer.close()

if __name__ == '__main__':
    main()