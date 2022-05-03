##############################Importing modules
import re
import os 
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import glob
import warnings 
import tqdm

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img



#####taking note of NON GPU ENABLED DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
    
warnings.filterwarnings("ignore")

class Preprocessing():
    def __init__(self):
        ################## DATA INFORMATION
        # Data directory of the project
        self.DATA_DIR =  '/mnt/d6cf3633-27d8-4655-8a8c-7fc0daf3bb07/1/Documents/machine_learning/Pneumonia detection/Dataset/data-task1'

        # dataset respective directory
        self.TRAIN_DIR = os.path.join(self.DATA_DIR, "train")
        self.TEST_DIR = os.path.join(self.DATA_DIR, "test")
        self.VAL_DIR = os.path.join(self.DATA_DIR, "val")


        ################# HYPERPARAMETER TUNING
        self.IMAGE_MIN_DIM = 224
        self.IMAGE_MAX_DIM = 224
        self.EPOCHS = 1
        self.BATCH_SIZE = 30

    def plot_random_dataset_images(self) -> None:
        '''
        Arguments:None
        Purpose: helps plot our classes of images for our train,test and val datasets
        Return : None
        '''
        fig, ax = plt.subplots(2, 3, figsize=(15, 7))
        ax = ax.ravel()
        plt.tight_layout()

        #plotting random non-pneumonia and pneumonia images from our train, test, and val to see how much different they look to our eye
        for i, dir_ in enumerate(glob.glob(f'{self.DATA_DIR}/*')):
            ax[i].imshow(plt.imread(dir_ + '/no_pneumonia/' +next(os.walk(dir_+'/no_pneumonia/'))[2][0]))
            ax[i].set_title(f'Set: {dir_.split("/")[-1]}, Condition: no_pneumonia')

            ax[i+3].imshow(plt.imread(dir_ + '/pneumonia/' +next(os.walk(dir_+'/pneumonia/'))[2][0]))
            ax[i+3].set_title(f'Set: {dir_.split("/")[-1]}, Condition: pneumonia')


    def basic_descriptive_of_images(self):
        '''
        '''
        #parsing and storing datasets into variable
        non_pneumonia_train = glob.glob(f'{self.TRAIN_DIR}/no_pneumonia/*.png')
        pneumonia_train = glob.glob(f'{self.TRAIN_DIR}/no_pneumonia/*.png')


        #Basic EDA -Descriptive
        for i in glob.glob(f'{self.DATA_DIR}/*'):
            non_pneumonia_images_count = len(glob.glob(i+'/no_pneumonia/*.png'))
            pneumonia_images_count = len(glob.glob(i+'/pneumonia/*.png'))
            # print(f'{i.split("/")[-1]} Dataset \n Non-pneumonia images: {non_pneumonia_images_count} \n pneumonia images: {pneumonia_images_count} \n')

            x = ['pneumonia', 'Non-pneumonia']
            y = [pneumonia_images_count, non_pneumonia_images_count]
            plt.bar(x, y)
            plt.title(f'Set:{i.split("/")[-1]}  pneumonia vs non pneumonia')
            plt.ylabel('Count')
            plt.xlabel('Classes')
            plt.show()

    def data_augmentation(self):
        '''
        argument:None
        purpose:Augmentation expands the size of the dataset by creating a modified version of the existing training set images that helps to increase dataset variation and ultimately improve the ability of the model to predict new images.
        return: train_image_gen, test_image_gen
        '''
        train_image_gen = ImageDataGenerator(
            horizontal_flip = True,
            vertical_flip = False, 
            rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.05,
            height_shift_range = 0.02,
            width_shift_range = 0.02,
            rotation_range = 3,
            fill_mode = 'nearest'
        )
        test_image_gen = ImageDataGenerator(
            rescale = 1./255
        )

        return train_image_gen, test_image_gen

        
    def dataset_splitting(self, train_image_gen, test_image_gen):
        '''
        argument:augmented train images and augmented test images
        purpose:get images batches into respective variable sets
        return: train_images, test_images, validation_images
        '''
        train_set = train_image_gen.flow_from_directory(
            self.TRAIN_DIR,
            target_size=(self.IMAGE_MIN_DIM, self.IMAGE_MAX_DIM),
            batch_size = self.BATCH_SIZE,
            class_mode = 'binary',
            # shuffle=True,
            # color_mode="grayscale"
        )

        test_set = test_image_gen.flow_from_directory(
            self.TEST_DIR,
            target_size=(self.IMAGE_MIN_DIM, self.IMAGE_MAX_DIM),
            batch_size = self.BATCH_SIZE-10,
            class_mode = 'binary',
            # shuffle=True,
            # color_mode="grayscale"
        )
        valid_set = train_image_gen.flow_from_directory(
            self.VAL_DIR,
            target_size=(self.IMAGE_MIN_DIM, self.IMAGE_MAX_DIM),
            batch_size = self.BATCH_SIZE,
            class_mode = 'binary',
            # shuffle=True,
            # color_mode="grayscale"
        )
        return train_set, test_set, valid_set

    