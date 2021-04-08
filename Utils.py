#!/usr/bin/env python
# title           :Utils.py
# description     :Have helper functions to process images and plot images
# author          :Deepak Birla
# date            :2018/10/30
# usage           :imported in other files
# python_version  :3.5.4

from tensorflow.keras.layers import Lambda
import tensorflow as tf
from skimage import data, io, filters
import numpy as np
from numpy import array
from numpy.random import randint
# from scipy.misc import imresize # duplicated since scipy 1.3
from extraction import extract_patches # added by Furat
from patching_utils import overlap_patching
from sklearn.feature_extraction.image import extract_patches_2d as extract_2d
#from config_keras import general_configuration as gen_conf
#from config_keras import training_configuration as train_test_conf
from config_keras import general_configuration as gen_conf
from config_keras import training_configuration as train_conf
from config_keras import test_configuration as test_conf

from PIL import Image
import os
import sys
import cv2

import matplotlib.pyplot as plt

plt.switch_backend('agg')

# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
# change to 1D --> upsample from h w to h/r w/r
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0], input_shape[1] * scale, input_shape[2] * scale, int(input_shape[3] / (scale ** 2))] #commented by Furat
        #dims = [input_shape[0], input_shape[1] * scale, input_shape[2] * scale] # changed by Furat
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)


# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    #images_hr = np.array(images, dtype="object")
    images_hr = array(images)
    print (images_hr.shape)
    #images_hr = np.int(images)
    #images_hr, this_lr_patch = overlap_patching(gen_conf, train_conf, images)
    images_hr = extract_2d (images_hr, (32,32))
    print (images_hr.shape)
    return images_hr



# Takes list of images and provide LR images in form of numpy array
# Using imresize to down sampling images
def lr_images(images_real, downscale):
    images = []
    for img in range(len(images_real)):
        # images.append(imresize(images_real[img], [images_real[img].shape[0] // downscale, images_real[img].shape[1] // downscale], interp='bicubic', mode=None))
        images.append(np.array(Image.fromarray(images_real[img]).resize((images_real[img].shape[0] // downscale, images_real[img].shape[1] // downscale), resample=Image.BICUBIC)))
    images_lr = array(images)
    #images_lr, this_lr_patch = overlap_patching(gen_conf, train_conf, images_real)
    #images_lr = extract_2d(images_lr, (32,32))
    return images_lr


#  normalize images to range [-1, 1]
def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5
    #input_data = (input_data -127.5)/127.5
    #return input_data


def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)
    #return input_data

def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path, elem)):
            directories = directories + load_path(os.path.join(path, elem))
            directories.append(os.path.join(path, elem))
    return directories


def load_data_from_dirs(dirs, ext, image_shape): #change to loading grayscale images
    files = []
    file_names = []
    count = 0
    (width, height, n) = image_shape
    #(width, height) = image_shape #changed to grayscale by Furat
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                # image = data.imread(os.path.join(d, f)) #duplicated
                image = io.imread(os.path.join(d,f))
                if len(image.shape) > 2: #commented by Furat
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) #commented by Furat
                files.append(image) # index was changed
                file_names.append(os.path.join(d, f)) #index was changed
                #else: #commented by Furat
                #    file_name = os.path.join(d, f) #commented by Furat
                 #   print('Image', file_name, 'is not 3 dimension') #commented by Furat
                count = count + 1
    
    return files


def load_data(directory, ext):
    files = load_data_from_dirs(load_path(directory), ext)
    return files


def load_training_data(directory, ext, image_shape, number_of_images=300, train_test_ratio=0.8): #was 1000
    print("========= Start loading data ==========")
    number_of_train_images = int(number_of_images * train_test_ratio)

    print(number_of_train_images)

    files = load_data_from_dirs(load_path(directory), ext, image_shape)

    print(len(files))

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    #test_array = array(files)
    #test_array = np.array(files,dtype="object")
    #test_array = np.matrix(files)
    #if len(test_array.shape) < 3: # commented by Furat
        #files= tf.image.rgb_to_grayscale()
    #    print("Images are of not same shape") # commented by Furat
        #print("Please provide same shape images") # commented by Furat
    #    sys.exit()


    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]
    

    # Re-scale image to x4 of image shape was defined

    x_train_hr = hr_images(x_train)
    x_train_hr = normalize(x_train_hr)

    x_train_lr = lr_images(x_train, 4)
    x_train_lr = normalize(x_train_lr)

    x_test_hr = hr_images(x_test)
    x_test_hr = normalize(x_test_hr)

    x_test_lr = lr_images(x_test, 4)
    x_test_lr = normalize(x_test_lr)

    print("========= End loading data ==========")

    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


def load_test_data_for_model(directory, ext, image_shape, number_of_images=30):
    files = load_data_from_dirs(load_path(directory), ext, image_shape)

    print("Load image from ", directory, "successfully")

    for file in files:
        print(file.shape)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    x_test_hr = hr_images(files)
    x_test_hr = normalize(x_test_hr)

    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr, x_test_hr


def load_test_data(directory, ext, number_of_images=30):
    files = load_data_from_dirs(load_path(directory), ext)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        sys.exit()

    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr


# While training save generated image(in form LR, SR, HR)
# Save only one image as sample  
def plot_generated_images(output_dir, epoch, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    print(examples)
    value = randint(0, examples)
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    plt.figure(figsize=figsize)

    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[value], interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(image_batch_hr[value], interpolation='nearest')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir + 'generated_image_%d.png' % epoch)

    # plt.show()


# Plots and save generated images(in form LR, SR, HR) from model to test the model 
# Save output for all images given for testing  
def plot_test_generated_images_for_model(output_dir, generator, x_test_hr, x_test_lr, dim=(1, 3), figsize=(15, 5)):
    examples = x_test_hr.shape[0]
    image_batch_hr = denormalize(x_test_hr)
    image_batch_lr = x_test_lr
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)
    image_batch_lr = denormalize(image_batch_lr)

    for index in range(examples):
        plt.figure(figsize=figsize)

        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(image_batch_lr[index], interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(image_batch_hr[index], interpolation='nearest')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir + 'test_generated_image_%d.png' % index)

        # plt.show()


# Takes LR images and save respective HR images
def plot_test_generated_images(output_dir, generator, x_test_lr, figsize=(5, 5)):
    examples = x_test_lr.shape[0]
    image_batch_lr = denormalize(x_test_lr)
    gen_img = generator.predict(image_batch_lr)
    generated_image = denormalize(gen_img)

    for index in range(examples):
        # plt.figure(figsize=figsize)

        plt.imshow(generated_image[index], interpolation='nearest')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir + 'high_res_result_image_%d.png' % index)

        # plt.show()
