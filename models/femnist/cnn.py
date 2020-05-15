import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
#import os
#from PIL import Image

IMAGE_SIZE = 28


class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        super(ClientModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes)
        # nn.Linear equivalent to tf.layers.dense()
        self.size = self.model_size()

    def forward(self, x):
        x = self.layer1(x.float())
        x = self.layer2(x)
        x = torch.reshape(x,(x.shape[0], -1))
        x = F.dropout(x, p=0.5, training=self.training)
        #print(x.shape)
        x = self.fc1(x)
        logits = self.fc2(x)
        return logits

    # def create_model(self):
    #     """Model function for CNN."""
    #     features = tf.placeholder(
    #         tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
    #     labels = tf.placeholder(tf.int64, shape=[None], name='labels')
    #     input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    #     conv1 = tf.layers.conv2d(
    #       inputs=input_layer,
    #       filters=32,
    #       kernel_size=[5, 5],
    #       padding="same",
    #       activation=tf.nn.relu)
    #     pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #     conv2 = tf.layers.conv2d(
    #         inputs=pool1,
    #         filters=64,
    #         kernel_size=[5, 5],
    #         padding="same",
    #         activation=tf.nn.relu)
    #     pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #     pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    #     dense = tf.layers.dense(inputs=pool2_flat, units=2048, activation=tf.nn.relu)
    #     logits = tf.layers.dense(inputs=dense, units=self.num_classes)
    #     predictions = {
    #       "classes": tf.argmax(input=logits, axis=1),
    #       "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    #     }
    #     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    #     # TODO: Confirm that opt initialized once is ok?
    #     train_op = self.optimizer.minimize(
    #         loss=loss,
    #         global_step=tf.train.get_global_step())
    #     eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
    #     return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch):
        #x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(raw_x_batch)
        #print(x_batch.shape)
        #print(x_batch.shape[0])
        x_batch = np.reshape(x_batch, (x_batch.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1))
        return x_batch

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size

    def _load_image(self, img_name):
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        return np.array(img)
