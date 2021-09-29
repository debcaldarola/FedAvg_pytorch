"""Script to run the baselines."""
import importlib
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import random
import torch
import torch.nn as nn
import metrics.writer as metrics_writer
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from utils.args import parse_args
from utils.model_utils import read_data
from datetime import datetime
import matplotlib.pyplot as plt

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'
CHECKPOINT_NAME = "mobilenet_fedavg_N50000_K10_090921_16:50:53"


def main():
    args = parse_args()

    # Set the random seed if provided (affects client sampling and batching)
    random.seed(1 + args.seed)
    np.random.seed(123 + args.seed)
    torch.manual_seed(args.seed)

    # Obtain the path to client's model (e.g. celeba/cnn.py)
    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    # Load model
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    # CIFAR: obtain info on parameter alpha (Dirichlet's distribution)
    if 'cifar' in args.dataset:
        data_dir = os.path.join('..', 'data', args.dataset, 'data', 'train')
        train_file = os.listdir(data_dir)
        alpha = train_file[0].split('train_')[1][:-5]
        print("Alpha:", alpha)

    # Experiment parameters (e.g. num rounds, clients per round, lr, etc)
    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Setup GPU
    device = torch.device(args.device if torch.cuda.is_available else 'cpu')

    # Create client model, and share params with server model
    client_model = ClientModel(*model_params, device)
    if args.load:  # load model from checkpoint
        print("--- Loading model", CHECKPOINT_NAME, "from checkpoint ---")
        load_path = os.path.join('.', 'checkpoints', args.dataset, '{}.ckpt'.format(CHECKPOINT_NAME))
        checkpoint = torch.load(load_path)
        # print(checkpoint.keys())
        client_model.load_state_dict(checkpoint, strict=False)
    elif args.multigpu:
        client_model = nn.DataParallel(client_model)  # multiple GPUs usage if more memory is required
    client_model = client_model.to(device)

    # Create server
    server = Server(client_model)

    # Create and set up clients
    train_clients, test_clients = setup_clients(args.dataset, args.lr, args.weight_decay, client_model,
                                                args.use_val_set, args.seed, device)
    train_client_ids, train_client_groups, train_client_num_samples = server.get_clients_info(train_clients)
    test_client_ids, test_client_groups, test_client_num_samples = server.get_clients_info(test_clients)
    if set(train_client_ids) == set(test_client_ids):
        print('Clients in Total: %d' % len(train_clients))
    else:
        print('Clients in Total: %d' % int(len(train_clients) + len(test_clients)))

    # Initial status
    print('--- Random Initialization ---')

    stat_writer_fn = get_stat_writer_function(test_client_ids, test_client_groups, test_client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)

    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    val_accuracies = []
    val_loss = []
    test_accuracies = []
    test_loss = []
    val_rounds = []

    # Create file for storing results
    res_path = os.path.join('results', args.dataset, args.model)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    start_time = datetime.now()
    current_time = start_time.strftime("%m%d%y_%H:%M:%S")

    if 'cifar' in args.dataset:
        file = os.path.join(res_path, 'results_' + str(alpha) + '_N' + str(num_rounds) + '_K' + str(
            clients_per_round) + '_' + current_time + '.txt')
    else:
        file = os.path.join(res_path, 'results_N' + str(num_rounds) + '_K' + str(
            clients_per_round) + '_' + current_time + '.txt')

    fp = open(file, "w")

    print_stats(0, server, train_clients, train_client_num_samples, test_clients, test_client_num_samples, args,
               stat_writer_fn, args.use_val_set, fp)

    # Initialize checkpoint path
    if args.load:
        save_path = os.path.join(ckpt_path,  '{}.ckpt'.format(CHECKPOINT_NAME))
    else:
        if 'cifar' in args.dataset:
            save_path = os.path.join(ckpt_path, '{}.ckpt'.format(args.model + '_fedavg_' + str(alpha) +
                                                         '_N' + str(num_rounds) + '_K' + str(
                    clients_per_round) + '_' + current_time))
        else:
            save_path = os.path.join(ckpt_path, '{}.ckpt'.format(args.model + '_fedavg_N' + str(num_rounds) + '_K' + str(
                    clients_per_round) + '_' + current_time))

    # Start training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))
        fp.write('--- Round %d of %d: Training %d Clients ---\n' % (i + 1, num_rounds, clients_per_round))

        start_round = datetime.now()

        # Select clients to train during this round
        server.select_clients(i, online(train_clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)
        print("Selected clients:", c_ids)

        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size,
                                         minibatch=args.minibatch)
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)

        # Update server model (FedAvg)
        print("--- Updating central model ---")
        server.update_model()

        print("\tCommunication round duration: ", datetime.now() - start_round)

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            val_metrics, test_metrics = print_stats(i + 1, server, train_clients, train_client_num_samples,
                                                    test_clients, test_client_num_samples,
                                                    args, stat_writer_fn, args.use_val_set, fp)
            val_accuracies.append(val_metrics[0])
            test_accuracies.append(test_metrics[0])
            val_loss.append(val_metrics[1])
            test_loss.append(test_metrics[1])
            val_rounds.append(i)

        print('Model saved in path: %s' % save_path)

    end_time = datetime.now()
    tot_time = end_time - start_time
    print("Total time for experiment: ", tot_time)
    fp.write("Total time for experiment: %d seconds, %d microseconds " % (tot_time.seconds, tot_time.microseconds))

    # Save server model
    print('Model saved in path: %s' % save_path)

    fp.close()
    print("File saved in path: %s" % res_path)

    # Save plots
    img_path = os.path.join('plots', args.dataset)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if 'cifar' in args.dataset:
        img_name_val = str(alpha) + '_val_N' + str(num_rounds) + '_K' + str(
            clients_per_round) + '_lr' + str(args.lr) + current_time
        img_name_test = str(alpha) + '_test_N' + str(num_rounds) + '_K' + str(
            clients_per_round) + '_lr' + str(args.lr) + current_time
    else:
        img_name_val = 'val_N' + str(num_rounds) + '_K' + str(clients_per_round) + '_lr' + str(args.lr) + current_time
        img_name_test = 'test_N' + str(num_rounds) + '_K' + str(clients_per_round) + '_lr' + str(args.lr) + current_time

    plot_metrics(val_accuracies, val_loss, val_rounds, img_name_val, img_path, 'Validation on ' + args.dataset)
    plot_metrics(test_accuracies, test_loss, val_rounds, img_name_test, img_path, 'Test on ' + args.dataset,
                 prefix='test_')
    print("Plots saved in path: %s" % img_path)


def online(clients):
    """We assume all users are always online."""
    return clients


def create_clients(users, groups, train_data, test_data, model, seed, device, lr, weight_decay):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(seed, u, lr, weight_decay, g, train_data[u], test_data[u], model, device) for u, g in zip(users, groups)]
    return clients


def setup_clients(dataset, lr, weight_decay, model=None, use_val_set=False, seed=None, device=None):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)

    train_users, train_groups, test_users, test_groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    train_clients = create_clients(train_users, train_groups, train_data, test_data, model, seed, device, lr, weight_decay)
    test_clients = create_clients(test_users, test_groups, train_data, test_data, model, seed, device, lr, weight_decay)

    return train_clients, test_clients


def get_stat_writer_function(ids, groups, num_samples, args):
    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir,
            '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):
    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir,
            '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def print_stats(
        num_round, server, train_clients, train_num_samples, test_clients, test_num_samples, args, writer, use_val_set,
        fp):
    train_stat_metrics = server.test_model(train_clients, args.batch_size, set_to_use='train')
    val_metrics = print_metrics(train_stat_metrics, train_num_samples, fp, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(test_clients, args.batch_size, set_to_use=eval_set)
    test_metrics = print_metrics(test_stat_metrics, test_num_samples, fp, prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)

    return val_metrics, test_metrics


def print_metrics(metrics, weights, fp, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    metrics_values = []
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))
        fp.write('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g\n' \
                 % (prefix + metric,
                    np.average(ordered_metric, weights=ordered_weights),
                    np.percentile(ordered_metric, 10),
                    np.percentile(ordered_metric, 50),
                    np.percentile(ordered_metric, 90)))
        # fp.write("Clients losses:", ordered_metric)
        metrics_values.append(np.average(ordered_metric, weights=ordered_weights))
    return metrics_values


def plot_metrics(accuracy, loss, n_rounds, figname, figpath, title, prefix='val_'):
    name = os.path.join(figpath, figname)

    plt.plot(n_rounds, loss, '-b', label=prefix + 'loss')
    plt.plot(n_rounds, accuracy, '-r', label=prefix + 'accuracy')

    plt.xlabel("# rounds")
    plt.ylabel("Average accuracy")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(name + '.png')  # should before show method
    plt.close()


if __name__ == '__main__':
    main()
