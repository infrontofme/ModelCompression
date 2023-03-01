'''
group convolution
learned group convolution
group Lasso Regularizer
learned group convolution for inference
'''


import sys
import math
import time
import numpy as np
import tensorflow as tf
sys.path.append('..')
from hp_condensenet import *


def cosine_learning_rate(learning_rate, n_epochs, epoch, n_batches, batch):
    """
    cosine decay learning rate from 0.1~0, during training phase
    :param learning_rate: 0.1, initial learning rate
    :param n_epochs: 300, total epochs
    :param epoch: current epoch
    :param n_batches: num_examples // batch_size
    :param batch: current batch
    :return: cosine_learning_rate
    """
    t_total = n_epochs * n_batches
    t_cur = (epoch -1 )* n_batches + batch
    learning_rate_cosine = 0.5 * learning_rate * (1 + math.cos(math.pi * t_cur / t_total))
    return learning_rate_cosine


def create_variable(name, shape, initializer,
                    dtype=tf.float32, trainable=True):
    """
    create variable for filters, weight, bias etc.
    :param name: str, variable name
    :param shape: list, variable shape
    :param initializer: ways to initilize
    :param dtype: default tf.float32
    :param trainable: bool, default true
    :return: tf.get_variable
    """
    return tf.get_variable(name=name,
                           shape=shape,
                           dtype=dtype,
                           initializer=initializer,
                           trainable=trainable)


def weight_variable_xavier(shape, name):
    """
    Initilize W for softmax layer
    :param shape: [features_total,n_classes]
    :param name: W
    :return: Initilized W
    """
    return tf.get_variable(
        name,
        shape=shape,
        initializer=tf.contrib.layers.xavier_initializer())


def bias_variable(shape, name='bias'):
    """
    Initilize b for softmax layer
    :param shape: [n_classes]
    :param name: bias
    :return: Initilized bias
    """
    return tf.get_variable(name, initializer=tf.constant(0.0, shape=shape))


def conv2d(_input, out_features, kernel_size, strides=[1, 1, 1, 1], padding='SAME'):
    """
    Convolutional
    :param _input: feature map
    :param out_features: output features nums
    :param kernel_size:  1 or 3
    :param strides: [1,1,1,1]
    :param padding: 'SAME'
    :return: feature map
    """
    in_features = int(_input.get_shape()[-1])
    kernel = create_variable(name='kernel', shape=[kernel_size, kernel_size, in_features, out_features],
                             initializer=tf.contrib.layers.variance_scaling_initializer())
    output = tf.nn.conv2d(_input, kernel, strides, padding)
    return output


def avg_pool(_input, k):
    """
    Average Pooling
    :param _input: feature map
    :param k: kernel size
    :return: feature map
    """
    ksize = [1, k, k, 1]
    strides = [1, k, k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(_input, ksize, strides, padding)
    return output


def batch_norm(_input,is_training):
    """
    Batch Normalization with tf.cond
    :param _input: feature map
    :param is_training: bool
    :return: feature map
    """
    return tf.contrib.layers.batch_norm(_input,
                                        scale=True,
                                        center=True,
                                        zero_debias_moving_mean=True,
                                        decay=0.9,
                                        updates_collections=None,
                                        is_training=is_training)


def dropout(_input, is_training):
    """
    dropout layer
    :param _input: feature map
    :param is_training: bool
    :return: feature map
    """
    if keep_prob < 1:
        output = tf.cond(
            is_training,
            lambda : tf.nn.dropout(_input,keep_prob),
            lambda : _input)

    else:
        output = _input
    return output


def groupconv2d(name, _input, out_features, kernel_size, strides):
    """
    group convolution for 3x3 or 1x1 G-Conv
    :param _input: feature map
    :param out_features: feature map
    :param kernel_size: 3x3
    :param strides: 1
    :return: grouped convolution for 3x3
    """
    feature_num = int(_input.get_shape()[-1])
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=strides, padding='SAME')
    with tf.variable_scope(name):
        weight = create_variable('weights', shape=[kernel_size, kernel_size, feature_num//group, out_features],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
        input_groups = tf.split(axis=3, num_or_size_splits=group, value=_input)
        weight_groups = tf.split(axis=3, num_or_size_splits=group, value=weight)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
        output = tf.concat(axis=3, values=output_groups)

        return output


def learned_group_conv2d(name, _input, kernel_size, out_channels):
    """
    Learned group convolution for 1x1 L-Conv, implement pruning during training
    :param _input: feature map
    :param kernel_size: 1x1
    :param out_channels: out features num
    :return: learned group feature map
    """
    in_channels = int(_input.get_shape()[-1])
    # check group and condense_factor
    assert _input.get_shape()[-1] % group == 0, "group number cannot be divided by input channels"
    assert _input.get_shape()[-1] % condense_factor == 0, "condensation factor cannot be divided by input channels"
    with tf.variable_scope(name):
        weight = create_variable('weight', shape=[kernel_size, kernel_size, in_channels, out_channels],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
        tf.add_to_collection('learned_group_layer', weight)
        print("weight")
        print(weight)
        mask = tf.get_variable('mask', shape=[kernel_size, kernel_size, in_channels, out_channels],
                               initializer=tf.constant_initializer(1),
                               trainable=False)
        tf.add_to_collection('mask', mask)
        # group-lasso regularizer
        # tf.add_to_collection('lasso', lasso_loss_regularizer(weight, mask))
        output = tf.nn.conv2d(_input, tf.multiply(weight, mask), strides=[1, 1, 1, 1], padding='SAME')
        assert output.get_shape()[-1] % group == 0, "group number cannot be divided by output channels"

        return output


def lasso_loss_regularizer(weight, mask):
    """
    group-lasso regularizer for learend group convolution
    :param weight: tensor, weight in learned
    :param mask:
    :return:
    """
    in_channels = int(weight.get_shape()[-2])
    out_channels = int(weight.get_shape()[-1])
    d_out = out_channels // group
    assert weight.get_shape()[0] == 1
    variable = tf.multiply(weight, mask)
    variable = tf.square(tf.squeeze(variable))
    print(variable)

    # shuffle weight
    variable = tf.reshape(variable, [d_out, group, in_channels])
    variable = tf.sqrt(tf.maximum(tf.reduce_sum(variable, axis=0),
                                  1e-10))  # add a small constant 1e-10 to solve tf.sqrt() numerical instability
    print(variable)

    variable = tf.reduce_sum(variable)
    print("lasso_loss", end=' ')
    print(variable)

    return variable

def LearnedGroupConvTest(name, _input, kernel_size, out_channels):
    """
    Learned group convolution for testing, inference phase
    :param _input: feature map
    :param kernel_size: 1x1
    :param out_channels: out features num
    :return: learned group feature map
    """
    in_channels = int(_input.get_shape()[-1])
    #check group and condense_factor
    assert _input.get_shape()[-1] % group == 0, "group number cannot be divided by input channels"
    assert _input.get_shape()[-1] % condense_factor == 0, "condensation factor cannot be divided by input channels"
    with tf.variable_scope(name):
        weight = create_variable('weight', shape=[kernel_size, kernel_size, in_channels, out_channels],
                                 initializer=tf.contrib.layers.variance_scaling_initializer())
        tf.add_to_collection('learned_group_layer', weight)
        mask = tf.get_variable("mask", shape=[kernel_size, kernel_size, in_channels, out_channels],
                               initializer=tf.constant_initializer(1),
                               trainable=False)
        tf.add_to_collection('mask', mask)

        #inference phase
        # masktrans = tf.transpose(mask, perm=[0, 1, 3, 2])
        # weighttrans = tf.transpose(weight, perm=[0, 1, 3, 2])

        idx = tf.where(tf.not_equal(tf.transpose(mask, perm=[0, 1, 3, 2]), 0)) # get non zero index

        weight_new = tf.gather_nd(tf.transpose(weight, perm=[0, 1, 3, 2]), idx) # get the remaining weights
        weight_new = tf.reshape(weight_new, shape=[1, 1, out_channels, int(in_channels // condense_factor)])
        weight_new = tf.transpose(weight_new, perm=[0, 1, 3, 2])

        idx_1 = tf.reshape(idx, [-1]) # reshape to a 1-d tensor
        idx_2 = idx_1[3::4] #get index of the remaining weights

        index = 0
        control_ops = []
        # conv and concat
        for i in range(group):
            idx2_1 = idx_2[index:index+in_channels//group]
            idx2_2 = tf.convert_to_tensor(np.array([j for j in range(0, out_channels)]))[i::group]
            groupout = tf.nn.conv2d(input=tf.gather(_input, idx2_1, axis=3),
                                    filter=tf.gather(weight_new, idx2_2, axis=3),
                                    strides=[1, 1, 1, 1], padding='SAME')
            control_ops.append(groupout)
            index += in_channels // group
        output = tf.concat(axis=3, values=control_ops)

        # shuffle
        output = tf.reshape(output, shape=[-1, int(_input.get_shape()[1]), int(_input.get_shape()[2]), group, out_channels//group])
        output = tf.transpose(output, perm=[0, 1, 2, 4, 3])
        output = tf.reshape(output, shape=[-1, int(_input.get_shape()[1]), int(_input.get_shape()[2]), out_channels])

        return output
