import os
import random
import numpy as np
import zipfile
import requests
import io
import matplotlib.image as mpimg

class dbMember: # class for holding image file path and corresponding classification
    def __init__(self, file_path, classification):
        self.file_path = file_path # file path for image
        self.classification = classification # corresponding classification

class tiny_imagenet:
    def __init__(self, url, data_set_dir):
        self.url = url
        self.num_classes = 200
        self.num_images_per_class = 50
        self.num_images = 100000
        self.num_test = 10000
        self.img_size = 64
        self.num_channels = 3
        self.data_dir = data_set_dir
        self.train_img_dir = os.path.join(data_set_dir, 'tiny-imagenet-200/train/')
        self.val_img_dir = os.path.join(data_set_dir, 'tiny-imagenet-200/val/')
        self.download_images()
        self.labels, self.encoded_labels = self.encode_labels()
        self.train_batch_size = 50
        self.val_batch_size = 50
        self.train_batch_index = 0
        self.val_batch_index = 0

    def download_images(self): # function for downloading tiny-imagenet-200, should take IMG_URL as argument
        if(os.path.isdir(self.train_img_dir)): # if there is a location for the training directory, function is not necessary
            print('Images already downloaded...')
        else:
            print('Downloading ' + self.url)
            r = requests.get(self.url, stream = True) # returns response object from download link, setting stream = true prevents partial download
            zip_ref = zipfile.ZipFile(io.BytesIO(r.content)) # downloads and zips file
            zip_ref.extractall(self.data_dir) # extracts tiny-imagenet-200 zip file to database directory
            print("Download Complete.")
            zip_ref.close()

    def encode_labels(self):
        with open(os.path.join(self.data_dir, "tiny-imagenet-200/wnids.txt")) as f:
            labels = [i[:-1] for i in f.readlines()]
            encoded_labels = np.eye(200, dtype=float)
            f.close()
            return labels, encoded_labels

    def encode_single_label(self, label):
        index = 0
        for l in self.labels:
            if l == label:
                return self.encoded_labels[index]
            index+=1
        return -1

    def decode_single_label(self, encoded_label):
        index = 0
        for el in self.encoded_labels:
            if el == encoded_label:
                return self.labels[index]
            index+=1
        return -1

    def load_training_names(self):
        """
        Function that returns an array of class objects containing the image file path and classification
        for all 100,000 images in training folder. Prevents having to work with all 100,000 images directly.
        """
        training_data = []
        image_dir = self.train_img_dir
        print("Loading Training Data...")
        for type in os.listdir(image_dir):
            type_images = os.listdir(image_dir + type + '/images')
            for image in type_images:
                image_file = os.path.join(image_dir, type + '/images/', image)
                training_file = dbMember(image_file, self.encode_single_label(type))
                training_data.append(training_file)

        self.training_data = np.asarray(training_data)


    def load_validation_names(self):
        """
        Function that returns an array of class objects containing the image file path and classification
        for all 10,000 images in training folder. Prevents having to work with all 10,000 images directly.
        """
        validation_data = []
        image_dir = self.val_img_dir
        label_file = os.path.join(self.val_img_dir, 'val_annotations.txt')
        print("Loading Validation Data...")
        val_images = os.listdir(image_dir + '/images/')
        labels = [x.split('\t')[1] for x in open(label_file).readlines()]
        for image in val_images:
            image_file = os.path.join(image_dir, 'images/', image)
            label_index = int(image[4:-5])
            val_member = dbMember(image_file, self.encode_single_label(labels[label_index]))
            validation_data.append(val_member)

        self.validation_data = np.asarray(validation_data)

    def next_val_batch(self, batch_size):
        self.val_batch_size = batch_size
        if (self.val_batch_index + batch_size > self.num_test):
            self.val_batch_index = 0
            try:
                self.shuffle_val_data()
            except AttributeError:
                self.load_validation_names()
                self.shuffle_val_data()

        try:
            batch_names = self.validation_data[self.val_batch_index:self.val_batch_index+batch_size]
        except AttributeError:
            self.load_validation_names()
            batch_names = self.validation_data[self.val_batch_index:self.val_batch_index+batch_size]

        batch_images = np.zeros((batch_size, self.img_size*self.img_size*self.num_channels))
        batch_labels = np.zeros((batch_size, self.num_classes))

        i = 0
        for image in batch_names:
            batch_image_v = mpimg.imread(image.file_path)
            batch_image = np.divide(batch_image_v.astype(float), 255)
            if batch_image.shape == (64,64,3):
                batch_images[i] = batch_image.flatten()
            else:
                batch_image = np.dstack((batch_image, batch_image, batch_image))
                batch_images[i] = batch_image.flatten()
            batch_labels[i] = image.classification
            i+=1

        #batch_images = np.asarray([np.resize(mpimg.imread(i.file_path).flatten(),(self.img_size, self.img_size, self.num_channels)).flatten() for i in batch_names])
        #batch_labels = np.asarray([i.classification for i in batch_names])
        batch = [batch_images, batch_labels]
        self.val_batch_index+=batch_size

        return batch


    def next_train_batch(self, batch_size):
        self.train_batch_size = batch_size
        if (self.train_batch_index + batch_size > self.num_images):
            self.train_batch_index = 0
        try:
            batch_names = self.training_data[self.train_batch_index:self.train_batch_index+batch_size]
        except AttributeError:
            self.load_training_names()
            batch_names = self.training_data[self.train_batch_index:self.train_batch_index+batch_size]

        batch_images = np.zeros((batch_size, self.img_size*self.img_size*self.num_channels))
        batch_labels = np.zeros((batch_size, self.num_classes))

        i = 0
        for image in batch_names:
            batch_image_t = mpimg.imread(image.file_path)
            batch_image = np.divide(batch_image_t.astype(float), 255)
            if batch_image.shape == (64,64,3):
                batch_images[i] = batch_image.flatten()
            else:
                batch_image = np.dstack((batch_image, batch_image, batch_image))
                batch_images[i] = batch_image.flatten()
            batch_labels[i] = image.classification
            i+=1

        #batch_images = np.asarray([np.resize(mpimg.imread(i.file_path).flatten(),(self.img_size, self.img_size, self.num_channels)).flatten() for i in batch_names])
        #batch_labels = np.asarray([i.classification for i in batch_names])
        batch = [batch_images, batch_labels]
        self.train_batch_index+=batch_size

        return batch

    def shuffle_train_data(self):
        try:
            np.random.shuffle(self.training_data)
        except AttributeError:
            self.load_training_names()
            np.random.shuffle(self.training.data)

    def shuffle_val_data(self):
        try:
            np.random.shuffle(self.validation_data)
        except AttributeError:
            self.load_validation_names()
            np.random.shuffle(self.validation_data)
