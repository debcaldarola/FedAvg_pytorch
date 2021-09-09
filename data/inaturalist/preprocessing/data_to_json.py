import os
from csv import DictReader
import json


def save_as_json_file(filename, data):
    with open(filename, 'w+') as fp:
        json.dump(data, fp)


def parse_file_information(file_path, train=True, train_users=None):
    dictionary = {"users": [], "num_samples": [], "user_data": {}}
    user_data = {}
    if train:
        n_users = 0
    else:
        if train_users == -1:
            print("Parse training file first!")
            exit(-1)
        n_users = train_users
    with open(file_path, 'r') as file:
        reader = DictReader(file)
        for line in reader:
            if train:
                user_id = line['user_id']
            else:
                user_id = n_users
            image_id = line['image_id']
            class_id = int(line['class'])
            if user_id not in dictionary["users"]:
                dictionary["users"].append(user_id)
            if user_id not in user_data:
                user_data[user_id] = {'x': [], 'y': []}
            user_data[user_id]['x'].append(image_id)
            user_data[user_id]['y'].append(class_id)

    for user in dictionary["users"]:
        dictionary["user_data"][user] = user_data[user]
        dictionary["num_samples"].append(len(user_data[user]['y']))

    print("\tTotal images: ", sum(dictionary["num_samples"]))

    if train:
        n_users = len(dictionary["users"])
        return dictionary, n_users
    else:
        return dictionary, None


def parse_file(csv_path, dir_path, train=True, n_train_users=-1):
    if train:
        f = 'federated_train_user_120k.csv'
    else:
        f = 'test.csv'
    print("\tParsing file ", f)
    file_path = os.path.join(csv_path, f)

    if train:
        dictionary, n_train_users = parse_file_information(file_path, train)
        print("Total train users: ", n_train_users)
    else:
        dictionary, _ = parse_file_information(file_path, train, train_users=n_train_users)
    f_name = f[:-4] + '.json'
    json_path = os.path.join(dir_path, f_name)
    save_as_json_file(json_path, dictionary)
    return n_train_users


def main():
    csv_path = os.path.join('..', 'data', 'raw', 'inaturalist-user-120k')
    if not os.path.exists(csv_path):
        print("Launch program in /FedAvg_pytorch/data/inaturalist/preprocessing/")
        exit(0)

    # Create train and test directories to store json files
    print("Retrieving folders for json storage...")
    train_data_dir = os.path.join('..', 'data', 'train')
    test_data_dir = os.path.join('..', 'data', 'test')
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    # Read files and save them as json in the required format
    print("Parsing file from csv to json...")
    n_train_users = parse_file(csv_path, train_data_dir)
    parse_file(csv_path, test_data_dir, train=False, n_train_users=n_train_users)


if __name__ == '__main__':
    main()
