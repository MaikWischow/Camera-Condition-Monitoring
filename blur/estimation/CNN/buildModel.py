import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Input
import cv2

def bn(x, name, zero_init=False):
    return BatchNormalization(
        axis=1, name=name, fused=True,
        momentum=0.9, epsilon=1e-5,
        gamma_initializer='zeros' if zero_init else 'ones')(x)


def conv(x, filters, kernel, strides=1, name=None):
    return Conv2D(filters, kernel, name=name,
                  strides=strides, use_bias=True, padding='same')(x)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv(input_tensor, filters1, 1, name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b')
    x = Activation('relu')(x)

    x = conv(x, filters3, (1, 1), name=conv_name_base + '2c')
    x = bn(x, name=bn_name_base + '2c', zero_init=True)

    x = tf.keras.layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def residual_block(input_tensor, kernel_size, filters, block):
    filters1, filters2 = filters
    conv_name_base =  block + "/" + "conv"

    x = conv(input_tensor, filters1, kernel_size, name=conv_name_base + '1')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, name=conv_name_base + '2')
    x = Activation('relu')(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def residual_block_change_dim(input_tensor, kernel_size, filters, strides, block):
    filters1, filters2, filters3 = filters
    stride1, stride2, stride3 = strides
    conv_name_base =  block + "/" + "conv"

    x = conv(input_tensor, filters1, kernel_size, stride1, name=conv_name_base + '1')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, stride2, name=conv_name_base + '2')
    x = Activation('relu')(x)
    
    shortcut = conv(input_tensor, filters3, 1, stride3, name=conv_name_base + 'shortcut')
    shortcut = Activation('relu')(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv(input_tensor, filters1, (1, 1), name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, strides=strides, name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b')
    x = Activation('relu')(x)

    x = conv(x, filters3, (1, 1), name=conv_name_base + '2c')
    x = bn(x, name=bn_name_base + '2c', zero_init=True)

    shortcut = conv(
        input_tensor,
        filters3, (1, 1), strides=strides,
        name=conv_name_base + '1')
    shortcut = bn(shortcut, name=bn_name_base + '1')

    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def MTFNet(inputShape):
    inputs = Input(shape=inputShape)

    # Conv01
    x = conv(inputs, 128, (5, 5), strides=(1, 1), name='conv01')
    x = Activation('relu')(x)
    
    # Residual Blocks 1-7
    x = residual_block(x, 5, [128, 128], block="block0")
    x = residual_block(x, 3, [128, 128], block="block1")
    x = residual_block_change_dim(x, 3, [256, 256, 256], [2, 1, 2], block="block2")
    x = residual_block_change_dim(x, 3, [256, 256, 256], [2, 1, 2], block="block3")
    x = residual_block_change_dim(x, 3, [256, 256, 256], [2, 1, 2], block="block4")
    x = residual_block_change_dim(x, 3, [256, 256, 256], [2, 1, 2], block="block5")
    x = residual_block_change_dim(x, 2, [256, 256, 256], [2, 1, 2], block="block6")
    
    # Fully connected layers 1-3
    x = Dense(256, activation='relu', name="fc1")(x)
    x = Dense(256, activation='relu', name="fc2")(x)
    x = Dense(128, activation='relu', name="fc3")(x)
    x = Dense(8, activation='sigmoid', name="fc_out")(x)
    
    M = tf.keras.models.Model(inputs, x, name='MTFNet')
    return M