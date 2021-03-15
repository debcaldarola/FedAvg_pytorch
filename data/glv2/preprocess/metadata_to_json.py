import os
import csv
from csv import DictReader
import json

ONE_TRAIN_FILE = True
IMGS_PER_TEST_USER = 30

def save_as_json_file(filename, data):
    with open(filename, 'w+') as fp:
        json.dump(data, fp)

def parse_file_information(file_path, train=True):
    dictionary = {"users": [], "num_samples": [], "user_data": {}}
    user_data = {}
    n_users = 1262
    assigned = 0
    with open(file_path, 'r') as file:
        reader = DictReader(file)
        next(reader)  # skip header
        for line in reader:
            if train:
                user_id = int(line['user_id'])
            else:
                user_id = n_users
                # assigned += 1 PROVA CON UN SOLO UTENTE
                if assigned == IMGS_PER_TEST_USER:
                    n_users += 1
                    assigned = 0
            image_id = line['image_id']
            class_id = int(line['class'])
            # domain = int(line['domain'])
            if user_id not in dictionary["users"]:
                dictionary["users"].append(user_id)
            if user_id not in user_data:
                user_data[user_id] = {'x': [], 'y': []}
                # user_data[user_id] = {'x': [], 'y': [], 'domain': []}
            user_data[user_id]['x'].append(image_id)
            user_data[user_id]['y'].append(class_id)
#            user_data[user_id]['domain'].append(domain)

    for user in dictionary["users"]:
        dictionary["user_data"][user] = user_data[user]
        dictionary["num_samples"].append(len(user_data[user]['y']))

    return dictionary

def parse_file(csv_path, dir_path, train=True):
    if train:
        file = 'federated_train.csv'
    else:
        file = 'test.csv'

    file_path = os.path.join(csv_path, file)
    dictionary = parse_file_information(file_path, train)
    f_name = file[:-4] + '.json'
    json_path = os.path.join(dir_path, f_name)
    save_as_json_file(json_path, dictionary)

def main():
    csv_path = os.path.join('..', 'landmarks-user-160k')
    if not os.path.exists(csv_path):
        print("Launch program in /leaf_pytorch/data/glv2/preprocess/")
        exit(0)

    # Create train and test directories to store json files
    train_data_dir = os.path.join('..', 'data', 'train')
    test_data_dir = os.path.join('..', 'data', 'test')
    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    # Read files and save them as json in the required format
    parse_file(csv_path, train_data_dir)
    parse_file(csv_path, test_data_dir, train=False)


if __name__ == '__main__':
    main()
