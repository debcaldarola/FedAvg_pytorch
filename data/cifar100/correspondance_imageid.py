import pandas as pd
import json
import numpy as np
import pickle
from scipy import misc
from tqdm import tqdm

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

train = unpickle('cifar-100-python/train')

filenames = [t.decode('utf8') for t in train[b'filenames']]
fine_labels = train[b'fine_labels']

class_info = dict.fromkeys(set(fine_labels))

for index in range(len(filenames)):
    filename = filenames[index]
    label = fine_labels[index]
    if class_info[label] is None:
        class_info[label] = {'filenames': [], 'img_ids': []}
    class_info[label]['filenames'].append(filename)

with open('../train/federated_train_alpha_0.00.json') as f:
    train_json = json.load(f)
    user_data = train_json["user_data"]

    for u, u_data in user_data.items():
        c = u_data['y'][0]
        # class_info[c]['img_ids'] = u_data['x']
        train_json["user_data"][u]['x'] = class_info[c]['filenames']

with open('federated_train_alpha_0.00.json', 'w+') as f:
    json.dump(train_json, f)

test = unpickle('cifar-100-python/test')

filenames_test = [t.decode('utf8') for t in test[b'filenames']]
fine_labels_test = test[b'fine_labels']

for index in range(len(filenames_test)):
    filename = filenames[index]
    label = fine_labels[index]
    if class_info[label] is None:
        class_info[label] = {'filenames': []}
    class_info[label]['filenames'].append(filename)

with open('../test/test.json') as f:
    test_json = json.load(f)
    user_data = test_json["user_data"]

for u, u_data in user_data.items():
    user_data[u]['x'] = []
    for c in u_data['y']:
        img_name = class_info[c]['filenames'].pop(0)
        user_data[u]['x'].append(img_name)

with open('test.json', 'w+') as f:
    json.dump(test_json, f)
