#!/usr/bin/env python
# title           :train.py
# description     :to train the model
# author          :Deepak Birla
# date            :2018/10/30
# usage           :python train.py --options
# python_version  :3.5.4

from Network import Generator, Discriminator
import Utils_model, Utils
from Utils_model import VGG_LOSS
from patching_utils import overlap_patching # added by Furat
from extraction import extract_patches # added by Furat
from sklearn.feature_extraction.image import extract_patches_2d as extract_2d
#from keras.layers import * #added by Furat
from keras import backend as k
from skimage.color import rgb2gray

from config_keras import general_configuration as gen_conf
from config_keras import training_configuration as train_conf
from config_keras import test_configuration as test_conf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tqdm import tqdm
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import os
import argparse

# To fix error Initializing libiomp5.dylib
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.random.seed(10)
# Better to use downscale factor as 4
downscale_factor = 4
# Remember to change image shape if you are having different size of images
#image_shape = (256, 256, 3) # commented by Furat
image_shape = (768, 768 , 3) #was 128 by 128
original_image_shape= image_shape


## Added by Furat
#patch_shape = train_conf['patch_shape']

patch_shape = (32,32,1)
    
## done
# Combined network
def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


# default values for all parameters are given, if want defferent values you can give via commandline
# for more info use $python train.py -h
def train(epochs, batch_size, input_dir, output_dir, model_save_dir, number_of_images, train_test_ratio, image_extension):

    # Loading images
    
    x_train_lr, x_train_hr, x_test_lr, x_test_hr = \
        Utils.load_training_data(input_dir, image_extension, image_shape, number_of_images, train_test_ratio)
        
    # convert to loading PATCHES
    #num_samples = dataset_info['num_samples'][1]
    

    print('======= Loading VGG_loss ========')
    # Loading VGG loss
    
    # convert to 3 channels
    #img_input = Input(shape=original_image_shape)
    #image_shape_gray = Concatenate()([img_input, img_input, img_input])
    #image_shape_gray = Concatenate()([original_image_shape, original_image_shape])
    #image_shape_gray = Concatenate()([image_shape_gray,original_image_shape])
    #image_shape = patch_shape
    #experimental_run_tf_function=False
    
    loss = VGG_LOSS(image_shape) # was image_shape
    
    print('====== VGG_LOSS =======', loss)
    
    # 1 channel
    #image_shape= original_image_shape
    batch_count = int(x_train_hr.shape[0] / batch_size)
    #batch_count = int(x_train_hr_patch.shape[0] / batch_size) # for patch
    
    print('====== Batch_count =======', batch_count)

    shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, image_shape[2]) # commented by Furat
    #shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor)
    print('====== Shape =======', shape)
    
    # Generator description
    generator = Generator(shape).generator()

    # Discriminator description
    discriminator = Discriminator(image_shape).discriminator()

    optimizer = Utils_model.get_optimizer()

    generator.compile(loss=loss.vgg_loss, optimizer=optimizer)

    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan = get_gan_network(discriminator, shape, generator, optimizer, loss.vgg_loss)

    loss_file = open(model_save_dir + 'losses.txt', 'w+')

    loss_file.close()
    
    ## restore the patches into 1 image:
    # x_train_hr should have a whole image insted of patches?
    
    ######
   # input_data= x_train_hr
   # patch_shape = train_conf['patch_shape']
   # output_shape = train_conf['output_shape']
   # num_chs = num_modalities * dataset_info['RGBch'] # number of total channels
    
   # if input_data.ndim == 6: # augmentation case
   #     num_samples = dataset_info['num_samples'][1]
    
    #num_samples = 3
    ######
    
    #lr_data= x_train_lr
    #hr_data= x_train_hr
    
    #for patch_idx in range (num_patches):
        #this_input_data = np.reshape(input_data[:,:,patch_idx], input_data.shape[:2]+input_data.shape[3:])
    #this_hr_patch, this_lr_patch = overlap_patching(gen_conf, train_conf, x_train_hr)
    #this_output_patch, out = overlap_patching(gen_conf, train_conf, output_data)
    
    # take patches:
    this_hr_patch, = extract_2d (x_train_hr, (32,32))
    this_lr_patch= extract_2d (x_train_lr, (32,32))
    
    x_train_lr = this_lr_patch
    x_train_hr = this_hr_patch
    
    #convert to grayscale
    #x_train_hr= tf.image.rgb_to_grayscale(x_train_hr)
    #x_train_hr= rgb2gray(x_train_hr)
    #x_train_hr= np.concatenate(x_train_hr,1)
    #x_train_hr= np.array(x_train_hr)
    #x_train_lr= tf.image.rgb_to_grayscale(x_train_lr)
    #x_train_lr= np.array(x_train_lr)
    #x_train_lr= rgb2gray(x_train_lr)
    #x_train_lr= np.concatenate(x_train_lr,1)
    
    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        for _ in tqdm(range(batch_count)):
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            fake_data_Y = np.random.random_sample(batch_size) * 0.2

            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums]
            image_batch_lr = x_train_lr[rand_nums]

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr, gan_Y])

        print("discriminator_loss : %f" % discriminator_loss)
        print("gan_loss :", gan_loss)
        gan_loss = str(gan_loss)

        loss_file = open(model_save_dir + 'losses.txt', 'a')
        loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' % (e, gan_loss, discriminator_loss))
        loss_file.close()

        if e == 1 or e % 5 == 0:
            Utils.plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr)
        if e % 200 == 0:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)
            discriminator.save(model_save_dir + 'dis_model%d.h5' % e)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('-i', '--input_dir', action='store', dest='input_dir', default='./data_hr/',
#                         help='Path for input images')
#
#     parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default='./output/',
#                         help='Path for Output images')
#
#     parser.add_argument('-m', '--model_save_dir', action='store', dest='model_save_dir', default='./model/',
#                         help='Path for model')
#
#     parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', default=64,
#                         help='Batch Size', type=int)
#
#     parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=1000,
#                         help='number of iteratios for trainig', type=int)
#
#     parser.add_argument('-n', '--number_of_images', action='store', dest='number_of_images', default=1000,
#                         help='Number of Images', type=int)
#
#     parser.add_argument('-r', '--train_test_ratio', action='store', dest='train_test_ratio', default=0.8,
#                         help='Ratio of train and test Images', type=float)
#
#     values = parser.parse_args()
#
#     train(values.epochs, values.batch_size, values.input_dir, values.output_dir, values.model_save_dir,
#           values.number_of_images, values.train_test_ratio)

# Parameter
param_epochs = 1000
param_batch = 20
param_input_folder = './VN_dataset/'
param_out_folder = './output/'
param_model_out_folder = './model/'
param_number_images = 300 #was 3000
param_train_test_ratio = 0.8 #was 0.8
param_image_extension = '.png'

train(param_epochs,
      param_batch,
      param_input_folder,
      param_out_folder,
      param_model_out_folder,
      param_number_images,
      param_train_test_ratio,
      param_image_extension)


    #for p in range(this_hr_patch):
    #x_train_hr= this_hr_patch[p]
    #x_train_lr= this_lr_patch[p]
    
    ########
    #x_train_lr, x_train_hr, x_test_lr, x_test_hr
    # get overlap patches of size 256x256?
    #if input_data.ndim == 6: # the case of augmenting simulation
    #    x_train = np.zeros((0, num_chs) + patch_shape)
    #    y_train = np.zeros((0, num_chs) + output_shape)
    #    set_trace()
    #    for smpl_idx in range(num_samples):
    #        this_input_data = np.reshape(input_data[:,:,0], input_data.shape[:2]+input_data.shape[3:])
    #        np.delete(input_data, 0, 2)
    #        this_x_train, this_y_train = overlap_patching(
    #            gen_conf, train_conf, this_input_data[train_index], labels[train_index])
    #        x_train = np.vstack((x_train, this_x_train))
    #        y_train = np.vstack((y_train, this_y_train))
    #else:
    #    x_train, y_train = overlap_patching(
    #        gen_conf, train_conf, input_data[train_index], labels[train_index])
        
    #x_train=x_train_hr
    #y_train=x_train_lr
    
    ########
    
    
    #num_patches = 3 #3 parts of each images, size 256*256
    #input_data= x_train_lr
    #output_data= x_train_hr
    
    #for patch_idx in range (num_patches):
        #this_input_data = np.reshape(input_data[:,:,patch_idx], input_data.shape[:2]+input_data.shape[3:])
    #this_input_patch, out = overlap_patching(gen_conf, train_conf, input_data)
    #this_output_patch, out = overlap_patching(gen_conf, train_conf, output_data)
        #patched_input = np.vstack((patched_input, this_input_patch))
        #patched_output = np.vstack((patched_output, this_output_patch))
        
    #x_train_lr = this_input_patch
    #x_train_hr = this_output_patch
    
    #########
    
    #my_input_image = x_train_lr # shape = ( batch , size_x, size_y)
    #my_input_image = np.asarray(my_input_image).astype('float32')
    #my_input_image = tf.expand_dims(my_input_image ,-1) #add 1 more "depth" channel as the last axis
    #ksizes = [1, 512, 512, 1] #size of output patch
    #strides = [1, 256, 256, 1] # Stride
    #rates = [1, 1, 1, 1] #Rate
    #padding='SAME' # I want to have zero padding when the stride go out of my_input_image
    #image_patches = tf.image.extract_patches(my_input_image, ksizes, strides, rates, padding)
    #image_patches.shape # => TensorShape([125, 5, 5, 262144]) . Why we have 5 pictures in a row?
    #patch1 = image_patches[0,0,0,] # Get the 1st patch
    #patch1 = tf.reshape(patch1, [512, 512, 1]) # Reshape to the correct shape
    #patch1 = tf.squeeze(patch1) # Remove the depth channel
    
    #my_input_image2 = x_train_hr # shape = ( batch , size_x, size_y)
    #my_input_image2 = np.asarray(my_input_image2).astype('float32')
    #my_input_image2 = tf.expand_dims(my_input_image2 ,-1) #add 1 more "depth" channel as the last axis
    #image_patches2 = tf.image.extract_patches(my_input_image2, ksizes, strides, rates, padding)
    #image_patches2.shape # => TensorShape([125, 5, 5, 262144]) . Why we have 5 pictures in a row?
    #patch2 = image_patches2[0,0,0,] # Get the 1st patch
    #patch2 = tf.reshape(patch2, [512, 512, 1]) # Reshape to the correct shape
    #patch2 = tf.squeeze(patch2) # Remove the depth channel
    
    #parch1= image_batch_lr
    #patch2= image_batch_hr
    
    #########
    # everything was shifted here!
