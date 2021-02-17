import importlib
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import argparse
import numpy as np
import json
import torch
from baseline_constants import MODEL_PARAMS
from main import setup_clients
from sklearn.cluster import KMeans

def main():
    args = parse_args()

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    device = torch.device(args.device if torch.cuda.is_available else 'cpu')

    print("--- Loading general model ---")
    client_model = ClientModel(*model_params, device).to(device)
    # Load model trained offline
    load_path = os.path.join('checkpoints', args.dataset, '{}.ckpt'.format('offline_'+args.model))
    client_model.load_state_dict(torch.load(load_path))
    # Create clients
    clients, test_clients = setup_clients(args.dataset, client_model, args.use_val_set, args.seed, device,
                                                args.lr)
    train_ids, test_ids = get_clients_ids(clients, test_clients)

    if set(train_ids) == set(test_ids):
        print("Equal train and test clients")
    else:
        print("Different train and test clients")
        clients.extend(test_clients)

    # Fine tune clients models on their local data
    model_params = []
    clients_weight = np.zeros(len(clients))
    print("--- Starting Training ---")
    for i, c in enumerate(clients):
        if i%1000 == 0:
            print("Trained {:d}/{:d} clients".format(i, len(clients)))
        num_samples, update = c.train(args.num_epochs, args.batch_size, None)   # update is a state_dict
        c_params = client_params(c.model.state_dict())
        model_params.append(torch.cat(c_params, dim=0))
        clients_weight[i] = num_samples

    clients_models = torch.stack(model_params).cpu().detach().numpy()

    # Cluster clients according to models params
    print("--- Clustering clients ---")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0, verbose=1).fit(clients_models, sample_weight=clients_weight)
    print("Number of iterations:", kmeans.n_iter_)
    print("Assignments:", kmeans.labels_)

    # Save assignments
    clients_assignments = dict.fromkeys(range(args.num_clusters), [])
    labels = kmeans.labels_
    for l, c in zip(labels, clients):
        clients_assignments[l].append(c.id)

    filename = args.dataset + '_kmeans_' + str(args.num_clusters) + '.json'
    with open(filename, "w") as write_file:
        json.dump(clients_assignments, write_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset',
                        type=str,
                        required=True)
    parser.add_argument('-model',
                        required=True,
                        type=str)
    parser.add_argument('-lr',
                        type=float)
    parser.add_argument('--use-val-set',
                        action='store_true')
    parser.add_argument('-device',
                        type=str,
                        default='cuda:0')
    parser.add_argument('--num-clusters',
                        type=int,
                        required=True)
    parser.add_argument('--batch-size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)
    parser.add_argument('--seed',
                        help='seed for random client sampling and batch splitting',
                        type=int,
                        default=0)
    return parser.parse_args()

def get_clients_ids(train_clients, test_clients):
    train_ids = []
    test_ids = []
    for tr_c, test_c in zip(train_clients, test_clients):
        train_ids.append(tr_c.id)
        test_ids.append(test_c.id)
    return train_ids, test_ids

def client_params(c_model):
    all_params = []
    for p in c_model.values():
        all_params.append(p.view(-1))
    return all_params

if __name__ == '__main__':
    main()