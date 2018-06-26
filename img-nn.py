"""Required Imports"""

import os
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import zipfile
import requests
from pathlib import Path
from tiny_imagenet import tiny_imagenet
from tqdm import tqdm


""" Global Constants for Image Classifier """

HOME_DIR = str(Path.home()) #portable function to locate home directory on  a computer
NUM_EPOCHS = 3 # number of passes through data
DATASET_DIR = os.path.join(HOME_DIR, 'dev/datasets/') #directory of the tiny-imagenet-200 database
IMG_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' # url for downloading tiny-imagenet-200
TRAIN_BATCH_SIZE = 50
VAL_BATCH_SIZE = 50

""" Functions for Image Classifier """

def plot_object(data): # function that uses matplotlib to plot a single image to the screen (for data visualization)
    np.resize(image,(img_size, img_size, num_channels))
    plt.figure(figsize=(1,1))
    plt.imshow(data, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def plot_objects(instances, images_per_row = 10, **options): # function that uses matplotlib to display multiple images (for data visualization)
    size = img_size
    images_per_row = min(len(instances), images_per_row)
    images = [np.resize(instance,(img_size, img_size, num_channels)) for instance in instances]
    n_rows = (len(instances)-1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        if(row == len(instances)/images_per_row):
            break
        rimages = images[row * images_per_row : (row+1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
        image = np.concatenate(row_images, axis=0)
    plt.imshow(image, **options)
    plt.axis("off")
    plt.ion()
    plt.show()
    plt.pause(1)
    plt.close("all")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2D(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

tiny_imagenet200 = tiny_imagenet(IMG_URL, DATASET_DIR)
tiny_imagenet200.load_training_names()
tiny_imagenet200.load_validation_names()

num_classes = tiny_imagenet200.num_classes
num_images = tiny_imagenet200.num_images
num_test_images = tiny_imagenet200.num_test
img_size = tiny_imagenet200.img_size
num_channels = tiny_imagenet200.num_channels
num_pool_layers = 4
num_filters_conv1 = 32
num_filters_conv2 = 64
num_filters_conv3 = 128
num_filters_conv4 = 256
num_nodes_fc1 = 512

x = tf.placeholder(tf.float32, shape=[None, num_channels*img_size*img_size])
y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

"""First Conv+Pool"""
W_conv1 = weight_variable([5,5,num_channels,num_filters_conv1])
b_conv1 = bias_variable([num_filters_conv1])

h_conv1 = tf.nn.relu(conv2D(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

"""Second Conv+Pool"""
W_conv2 = weight_variable([5,5,num_filters_conv1,num_filters_conv2])
b_conv2 = bias_variable([num_filters_conv2])

h_conv2 = tf.nn.relu(conv2D(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

"""Third Conv+Pool"""
W_conv3 = weight_variable([3,3,num_filters_conv2,num_filters_conv3])
b_conv3 = bias_variable([num_filters_conv3])

h_conv3 = tf.nn.relu(conv2D(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

"""Fourth Conv+Pool"""
W_conv4 = weight_variable([3,3,num_filters_conv3,num_filters_conv4])
b_conv4 = bias_variable([num_filters_conv4])

h_conv4 = tf.nn.relu(conv2D(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

"""Dense Layer"""
flat_res_4_layer = int((img_size/pow(num_pool_layers,2))*(img_size/pow(num_pool_layers,2))*num_filters_conv4)

h_pool4_flat = tf.reshape(h_pool4, [-1, flat_res_4_layer])

W_fc1 = weight_variable([flat_res_4_layer, num_nodes_fc1])
b_fc1 = bias_variable([num_nodes_fc1])

h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([num_nodes_fc1, num_classes])
b_fc2 = bias_variable([num_classes])

y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2
#y_conv = tf.Print(y_conv_out, [y_conv_out])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = y_conv))
#cross_entropy = tf.Print(cross_entropy_out, [cross_entropy_out])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv + 1e-9), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
val_best_acc = 0
val_threshold = 100
stop_iterator = 0
val_accumulated_accuracy = 0
stop_training = False
val_iterations = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    try:
        saver.restore(sess, "./tmp/best/model.ckpt")
    except:
        pass

    print("Model restored.")
    tiny_imagenet200.shuffle_val_data()
    for i in range(NUM_EPOCHS):
        print("Epoch #%d" % (i))
        tiny_imagenet200.shuffle_train_data()
        for j in tqdm(range(int(num_images/TRAIN_BATCH_SIZE))):
            train_batch = tiny_imagenet200.next_train_batch(VAL_BATCH_SIZE)

            """
            if j % 100 == 0 and j > 0:
                val_batch = tiny_imagenet200.next_val_batch(VAL_BATCH_SIZE)
                train_accuracy = accuracy.eval(feed_dict = {x: train_batch[0], y_: train_batch[1], keep_prob: 1.0})
                val_accuracy = accuracy.eval(feed_dict = {x: val_batch[0], y_: val_batch[1], keep_prob: 1.0})
                if (val_accuracy > val_best_acc):
                    val_best_acc = val_accuracy
                    stop_iterator = 0
                    save_path = saver.save(sess, "./tmp/best/model.ckpt")
                    print("Best model saved in path: %s" % save_path)

                stop_iterator += 1
                val_iterations += 1
                val_accumulated_accuracy += val_accuracy
                val_avg_accuracy = val_accumulated_accuracy/val_iterations

                print("step: %d" % (j))
                print("train accuracy: %g" % (train_accuracy))
                print("validation average accuracy: %g" % (val_avg_accuracy))

                save_path = saver.save(sess, "./tmp/model.ckpt")
                print("Model saved in path: %s" % save_path)
                if stop_iterator >= val_threshold:
                    print("Model has converged. Stopping training.")
                    stop_training = True

            if (stop_training):
                break
            """
                #print(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            #train_step.run(feed_dict = {x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})
            a, b, c = sess.run([train_step, cross_entropy, accuracy], feed_dict = {x: train_batch[0], y_: train_batch[1], keep_prob: 0.5})
            print("Cross Entropy: ", b)
            print("Train Accuracy: ", c)
