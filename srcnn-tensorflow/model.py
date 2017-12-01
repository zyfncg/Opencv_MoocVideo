from utils import (
    read_data,
    input_setup,
    imsave,
    merge
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import scipy.io


class SRCNN(object):

    def __init__(self,
                 sess,
                 image_size=33,
                 label_size=21,
                 batch_size=128,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')

        init_param = scipy.io.loadmat('model/9-1-5(ImageNet)/x3.mat')
        weights_conv1 = init_param['weights_conv1']
        conv1_patchsize2, conv1_filters = weights_conv1.shape
        conv1_patchsize = int(np.sqrt(conv1_patchsize2))
        weights_conv2 = init_param['weights_conv2']
        conv2_channels, conv2_patchsize2, conv2_filters = weights_conv2.shape
        conv2_patchsize = int(np.sqrt(conv2_patchsize2))
        weights_conv3 = init_param['weights_conv3']
        conv3_channels, conv3_patchsize2 = weights_conv3.shape
        conv3_patchsize = int(np.sqrt(conv3_patchsize2))

        weights_conv1 = weights_conv1.reshape(conv1_patchsize, conv1_patchsize, self.c_dim, conv1_filters)
        weights_conv2 = weights_conv2.reshape(conv2_patchsize, conv2_patchsize, conv2_channels, conv2_filters)
        weights_conv3 = weights_conv3.reshape(conv3_patchsize, conv3_patchsize, conv3_channels, 1)

        self.weights = {
            'w1': tf.Variable(initial_value=weights_conv1, name='w1',dtype=tf.float32),
            'w2': tf.Variable(initial_value=weights_conv2, name='w2',dtype=tf.float32),
            'w3': tf.Variable(initial_value=weights_conv3, name='w3',dtype=tf.float32)
        }

        biases_conv1 = init_param['biases_conv1']
        biases_conv2 = init_param['biases_conv2']
        biases_conv3 = init_param['biases_conv3']

        biases_conv1=biases_conv1.reshape(biases_conv1.shape[0])
        biases_conv2 = biases_conv2.reshape(biases_conv2.shape[0])
        biases_conv3 = biases_conv3.reshape(biases_conv3.shape[0])

        self.biases = {
            'b1': tf.Variable(initial_value=biases_conv1, name='b1',dtype=tf.float32),
            'b2': tf.Variable(initial_value=biases_conv2, name='b2',dtype=tf.float32),
            'b3': tf.Variable(initial_value=biases_conv3, name='b3',dtype=tf.float32)
        }

        self.pred = self.model()

        # Loss function (MSE)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver()

    def train(self, config):
        if config.is_train:
            input_setup(self.sess, config)
        else:
            test_input, test_label = input_setup(self.sess, config)

        if config.is_train:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
            train_data, train_label = read_data(data_dir)

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

        tf.initialize_all_variables().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.is_train:
            print("Training...")

            for ep in range(config.epoch):
                # Run by batch images
                batch_idxs = len(train_data) // config.batch_size
                for idx in range(0, batch_idxs):
                    batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
                    batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]

                    counter += 1
                    _, err = self.sess.run([self.train_op, self.loss],
                                           feed_dict={self.images: batch_images, self.labels: batch_labels})

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                              % ((ep + 1), counter, time.time() - start_time, err))

                    if counter % 500 == 0:
                        self.save(config.checkpoint_dir, counter)

        else:
            print("Testing...")

            result = self.pred.eval({self.images: test_input, self.labels: test_label})

            image_path = os.path.join(os.getcwd(), config.sample_dir)
            image_path = os.path.join(image_path, "test_image.png")
            bicbic_image_path = os.path.join(image_path, "bicbic_image.png")
            imsave(result, image_path)
            imsave(test_input, bicbic_image_path)

    def model(self):
        conv1 = tf.nn.relu(
            tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b1'])
        conv2 = tf.nn.relu(
            tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
        return conv3

    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
