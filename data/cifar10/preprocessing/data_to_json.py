import os
import argparse
from csv import DictReader
import json

# ONE_TRAIN_FILE = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-alpha',
                        required=True,
                        type=float,
                        choices=[0.00, 0.05, 0.10, 0.20, 0.50, 1.00, 10.00, 10.00, 100.00])
    return parser.parse_args()

def save_as_json_file(filename, data):
    with open(filename, 'w+') as fp:
        json.dump(data, fp)

def parse_file_information(file_path, train=True):
    dictionary = {"users": [], "num_samples": [], "user_data": {}}
    user_data = {}
    n_users = 100   # start assigning user_id from 100 for test users
    with open(file_path, 'r') as file:
        reader = DictReader(file)
        next(reader)  # skip header
        for line in reader:
            if train:
                user_id = int(line['user_id'])
            else:
                user_id = n_users
            filename = line['filename']
            class_id = int(line['class'])
            if user_id not in dictionary["users"]:
                dictionary["users"].append(user_id)
            if user_id not in user_data:
                user_data[user_id] = {'x': [], 'y': []}
            user_data[user_id]['x'].append(filename)
            user_data[user_id]['y'].append(class_id)

    for user in dictionary["users"]:
        dictionary["user_data"][user] = user_data[user]
        dictionary["num_samples"].append(len(user_data[user]['y']))

    return dictionary

def parse_file(csv_path, dir_path, train=True, alpha=None):
    if train and alpha is not None:
        if alpha == 0.05:
            f = 'federated_train_alpha_' + str(alpha) + '.csv'
        else:
            f = 'federated_train_alpha_' + str(alpha) + '0.csv'
    elif not train:
        f = 'test.csv'
    print("Parsing file ", f)
    file_path = os.path.join(csv_path, f)
    dictionary = parse_file_information(file_path, train)
    f_name = f[:-4] + '.json'
    json_path = os.path.join(dir_path, f_name)
    save_as_json_file(json_path, dictionary)


def main():
    args = parse_args()
    alpha = args.alpha
    assert alpha in [0.00, 0.05, 0.10, 0.20, 0.50, 1.00, 10.00, 10.00, 100.00]
    csv_path = os.path.join('.', 'cifar10_with_name')
    if not os.path.exists(csv_path):
        print("Launch program in /FedAvg_pytorch/data/cifar10/preprocessing/")
        exit(0)

    # Create train and test directories to store json files
    train_data_dir = os.path.join('..', 'data', 'train')
    test_data_dir = os.path.join('..', 'data', 'test')
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    # Read files and save them as json in the required format
    parse_file(csv_path, train_data_dir, alpha=alpha)
    # parse_file(csv_path, test_data_dir, train=False)


if __name__ == '__main__':
    main()