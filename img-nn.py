"""Required Imports"""

import os
import random
import matplotlib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import zipfile
import requests
import io
from sklearn import preprocessing
from pathlib import Path

""" Global Constants for Image Classifier """

HOME_DIR = str(Path.home()) #portable function to locate home directory on  a computer

NUM_EPOCHS = 1 # number of passes through data
BATCH_SIZE = 50 # size of processing batch
NUM_CLASSES = 200 # number of classes in tiny imagenet data set
NUM_IMAGES_PER_CLASS = 500 # number of images corresponding to each class
DATASET_DIR = HOME_DIR + '/dev/datasets/' #directory of the tiny-imagenet-200 database
TRAIN_IMG_DIR = DATASET_DIR + 'tiny-imagenet-200/train/' # directory of training images in tiny-imagenet-200 database
VAL_IMG_DIR = DATASET_DIR + 'tiny-imagenet-200/val/' # directory of validation images in tiny-imagenet-200 database

IMG_SIZE = 64 # resolution for images in database
NUM_CHANNELS = 3 # rgb channels for each image
IMG_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' # url for downloading tiny-imagenet-200

""" Relevant Classes for Image Classifier """

class dbMember: # class for holding image file path and corresponding classification
    def __init__(self, file_path, classification):
        self.file_path = file_path # file path for image
        self.classification = classification # corresponding classification

""" Functions for Image Classifier """

def download_images(url): # function for downloading tiny-imagenet-200, should take IMG_URL as argument
        if(os.path.isdir(TRAIN_IMG_DIR)): # if there is a location for the training directory, function is not necessary
            print('Images already downloaded...')
        else:
            print('Downloading ' + url)
            r = requests.get(url, stream = True) # returns response object from download link, setting stream = true prevents partial download
            zip_ref = zipfile.ZipFile(io.BytesIO(r.content)) # downloads and zips file
            zip_ref.extractall(DATASET_DIR) # extracts tiny-imagenet-200 zip file to database directory
            print("Dowload Complete.")
            zip_ref.close()

def plot_object(data): # function that uses matplotlib to plot a single image to the screen (for data visualization)
    np.resize(image,(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    plt.figure(figsize=(1,1))
    plt.imshow(data, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def load_training_names(image_dir):
    """
    Function that returns an array of class objects containing the image file path and classification
    for all 100,000 images in training folder. Prevents having to work with all 100,000 images directly.
    """
    training_data = []
    print("Loading Training Data...")
    for type in os.listdir(image_dir):
        type_images = os.listdir(image_dir + type + '/images')
        for image in type_images:
            image_file = os.path.join(image_dir, type + '/images/', image)
            training_file = dbMember(image_file, type)
            training_data.append(training_file)

    return np.asarray(training_data)


def load_validation_names(image_dir, label_file):
    """
    Function that returns an array of class objects containing the image file path and classification
    for all 10,000 images in training folder. Prevents having to work with all 10,000 images directly.
    """
    validation_data = []
    print("Loading Validation Data...")
    val_images = os.listdir(image_dir + '/images/')
    for element in os.listdir(image_dir):
        if (not os.path.isdir(os.path.join(image_dir, element))):
            labels = [x.split('\t')[1] for x in open(os.path.join(image_dir, element)).readlines()]
    for image in val_images:
        image_file = os.path.join(image_dir, 'images/', image)
        label_index = int(image[4:-5])
        val_member = dbMember(image_file, labels[label_index])
        validation_data.append(val_member)

    return np.asarray(validation_data)

def load_batch_index(list_size, batch_size=20): # function that returns a list of random indexs from the length of the list of training objects
    return np.random.choice(range(0,list_size), batch_size, replace=False)

def load_training_batches(batch_indexes, file_list, batch_size=20): # function that returns a list of training objects at the random indexes returned by the load_batch_index function
    return np.asarray([file_list[i] for i in batch_indexes])

def get_batch_images(file_list): # function that returns batch images (as matrices) from the training objects returned by load_training_batches function
    return np.asarray([mpimg.imread(i.file_path) for i in training_batch])

def plot_objects(instances, images_per_row = 10, **options): # function that uses matplotlib to display multiple images (for data visualization)
    size = IMG_SIZE
    images_per_row = min(len(instances), images_per_row)
    images = [np.resize(instance,(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)) for instance in instances]
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

def reset_graph(seed=42): #resets tensorflow graph
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


height = IMG_SIZE
width = IMG_SIZE
channels = NUM_CHANNELS
n_inputs = height * width * channels
n_outputs = 20

reset_graph()

X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
y = tf.placeholder(tf.int32, shape=[None], name="y")

le = preprocessing.LabelEncoder()
conv1 = tf.layers.conv2d(X_reshaped, filters = 32, kernel_size=[5,5], padding='SAME', activation=tf.nn.relu, name="conv1")
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
conv2 = tf.layers.conv2d(pool1, filters = 64, kernel_size=[5,5], padding='SAME', activation=tf.nn.relu, name="conv2")
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4)
dropout_reshape = tf.reshape(dropout, [-1, 8 * 8 * 64])
logits = tf.layers.dense(inputs=dropout_reshape, units=200, name='output')
Y_proba = tf.nn.softmax(logits, name="Y_proba")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

download_images(IMG_URL)
training_files = load_training_names(TRAIN_IMG_DIR)
class_labels_le = le.fit(np.asarray([i.classification for i in training_files]))
val_data = pd.read_csv(VAL_IMG_DIR + 'val_annotations.txt', sep='\t', header=None, names=['File', 'Class', 'X', 'Y', 'H', 'W'])
val_files = load_validation_names(VAL_IMG_DIR, val_data)
val_labels = class_labels_le.transform(np.asarray([i.classification for i in val_files][0:1000]))
val_images = np.asarray([np.resize(mpimg.imread(i.file_path).flatten(),(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)).flatten() for i in val_files][0:1000])


with tf.Session() as sess:
    init.run()
    for i in range(0, NUM_EPOCHS):
        temp_training_files = training_files
        while (len(temp_training_files) != 0):
            batch_indexes = load_batch_index(len(temp_training_files))
            training_batch = load_training_batches(batch_indexes, temp_training_files, BATCH_SIZE)
            batch_labels = class_labels_le.transform(np.asarray([i.classification for i in training_batch]))
            batch_images = np.asarray([np.resize(mpimg.imread(i.file_path).flatten(),(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)).flatten() for i in training_batch])
            print('Training Set', batch_images.shape, batch_labels.shape)
            sess.run(training_op, feed_dict={X: batch_images, y: batch_labels})
            plot_objects(batch_images)
            acc_train = accuracy.eval(feed_dict = {X: batch_images, y: batch_labels})
            acc_test = accuracy.eval(feed_dict={X: val_images, y: val_labels})
            print(i, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            save_path = saver.save(sess, "./tiny_imagenet")
            print(len(temp_training_files))
            temp_training_files = np.delete(temp_training_files, batch_indexes)
