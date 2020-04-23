import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image
#from model import Model

IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('..', 'data', 'celeba', 'data', 'raw', 'img_align_celeba')


class ClientModel(nn.Module):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer2_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(64*4*4, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes) #4 filters => 4 feature maps
        # nn.Linear equivalent to tf.layers.dense()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2_3(x)
        x = self.layer2_3(x)
        x = self.layer4(x)
        x = x.view(x.shape[0], -1)
        x = F.dropout(x, p=0.5, training=self.training)
        #FC layer
        x = self.fc1(x)
        logits = self.fc2(x)
#        loss = nn.CrossEntropyLoss()
        return logits

    # def create_model(self):
    #     input_ph = tf.placeholder(
    #         tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    #     out = input_ph
    #     for _ in range(4):
    #         out = tf.layers.conv2d(out, 32, 3, padding='same')
    #         out = tf.layers.batch_normalization(out, training=True)
    #         out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
    #         out = tf.nn.relu(out)
    #     out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
    #     logits = tf.layers.dense(out, self.num_classes)
    #     label_ph = tf.placeholder(tf.int64, shape=(None,))
    #     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #         labels=label_ph,
    #         logits=logits)
    #     predictions = tf.argmax(logits, axis=-1)
    #     minimize_op = self.optimizer.minimize(
    #         loss=loss, global_step=tf.train.get_global_step())
    #     eval_metric_ops = tf.count_nonzero(
    #         tf.equal(label_ph, tf.argmax(input=logits, axis=1)))
    #     return input_ph, label_ph, minimize_op, eval_metric_ops, tf.math.reduce_mean(loss)
    #   input_ph, label_ph sono placeholders, non necessari in pytorch

    def process_x(self, raw_x_batch):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        return raw_y_batch

    def _load_image(self, img_name):
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        return np.array(img)
