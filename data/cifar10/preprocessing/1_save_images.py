import os
import numpy as np
import pickle
from tqdm import tqdm
import imageio
import pandas as pd


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

meta = unpickle('../data/raw/cifar-10-batches-py/batches.meta')
label_names = [t.decode('utf8') for t in meta[b'label_names']]

print("Label names:", label_names)

images = []
labels = []

for i in range(1, 6):
    print("Batch ", i)
    batch = unpickle('../data/raw/cifar-10-batches-py/data_batch_'+str(i))
    data = batch[b'data']
    batch_labels = batch[b'labels']
    labels.extend(batch_labels)
    for d in data:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
        image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
        image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
        images.append(image)

print("Saving train images")
with open('cifar10.csv', 'w+') as f:
    for index, image in tqdm(enumerate(images)):
        label = labels[index]
        label_name = label_names[label]
        filename = 'img_' + str(index) + '_label_' + str(label) + '.png'
        imageio.imwrite('../data/raw/img/train/%s' % filename, image)
        f.write('img/train/%s,%s,%s\n' % (filename, label_name, label))

test = unpickle('../data/raw/cifar-10-batches-py/test_batch')
test_data = test[b'data']
test_labels = test[b'labels']
images = []

for d in test_data:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
    image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
    image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
    images.append(image)

print("Saving test images")
with open('cifar10.csv', 'a') as f:
    for index, image in tqdm(enumerate(images)):
        label = test_labels[index]
        label_name = label_names[label]
        filename = 'img_' + str(index) + '_label_' + str(label) + '.png'
        imageio.imwrite('../data/raw/img/test/%s' % filename, image)
        f.write('img/test/%s,%s,%s\n' % (filename, label_name, label))

df = pd.read_csv('cifar10.csv', header=None)
df2 = df.sort_values(by=[1])
df2.to_csv('orderedcifar10_labelname.csv')