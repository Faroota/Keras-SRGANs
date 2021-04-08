#!/usr/bin/env python
#title           :Utils_model.py
#description     :Have functions to get optimizer and loss
#author          :Deepak Birla
#date            :2018/10/30
#usage           :imported in other files
#python_version  :3.5.4

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import VGG16
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
#tf.compat.v1.disable_eager_execution()

class VGG_LOSS(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
        #self.image_shape= Concatenate()([image_shape,image_shape,image_shape])
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape) # was VGG19
        vgg19.trainable = False
        # Make trainable as False
        for l in vgg19.layers:
            l.trainable = False
        self.model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        self.model.trainable = False

    # computes VGG loss or content loss
    def vgg_loss(self, y_true, y_pred):
    
        return K.mean(K.square(self.model(y_true) - self.model(y_pred)))
    
def get_optimizer():
 
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam
