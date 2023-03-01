'''
CondenseNet-light-94
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import math
import time
import numpy as np
import tensorflow as tf

from datasets.cifar100 import *
from utils.utils_con import *
from hp_condensenet import *


def Evaluate(sess):
    total_loss = []
    total_acc = []
    test_pre_index = 0
    add = batch_size

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index:test_pre_index+add]
        test_batch_y = test_y[test_pre_index:test_pre_index+add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: 0,
            training_flag: False
        }

        loss_, acc_ = sess.run([cross_entropy, accuracy], feed_dict=test_feed_dict)
        total_loss.append(loss_)
        total_acc.append(acc_)
    test_loss = np.mean(total_loss)
    test_acc = np.mean(total_acc)
    return test_acc, test_loss


class ConDenseNet():
    def __init__(self, _input, is_training, growth_rate=16, layers_per_block=15, expansion=4):
        """
        Initilize CondenseNet
        :param _input: input image
        :param is_training: bool
        :param growth_rate: int , it controls output of each block
        :param layers_per_block:
        :param expansion:
        """
        self.is_training = is_training
        self.growth_rate = growth_rate
        self.layers_per_block = layers_per_block
        self.expansion = expansion
        self.model = self.ConDense_net(_input)

    def composite_function(self, _input, out_features, kernel_size=3):
        """
        BN-Relu-3x3 Conv-Dropout
        :param _input: feature map
        :param out_features: feature map nums = growrth rate
        :param kernel_size: 3x3
        :return: feature map
        """
        with tf.variable_scope("composite_function"):
            output = batch_norm(_input, is_training=self.is_training)
            output = tf.nn.relu(output)
            output = groupconv2d(name='gc', _input=output, out_features=out_features,
                                 kernel_size=kernel_size, strides=[1, 1, 1, 1])
            output = dropout(output, self.is_training)
        return output

    def bottleneck_condense(self, _input, out_features):
        """
        1x1 L-Conv
        :param _input: feature map
        :param out_features: feature map
        :return: L-Conv
        """
        with tf.variable_scope("Bottleneck"):
            output = batch_norm(_input, is_training=self.is_training)
            output = tf.nn.relu(output)
            output = LearnedGroupConvTest(name='lgc', _input=output, kernel_size=1, out_channels=out_features)
            output = dropout(output, is_training=self.is_training)
        return output

    def add_internal_layer(self, _input, t):
        """
        1x1 and 3x3 Conv and Concat
        :param _input: feature map
        :param growth_rate: growth rate
        :return: feature map
        """
        bottleneck_out = self.bottleneck_condense(_input=_input, out_features=t * self.growth_rate * self.expansion)
        comp_out = self.composite_function(_input=bottleneck_out, out_features=t * self.growth_rate, kernel_size=3)
        output = tf.concat(axis=3, values=(_input, comp_out))

        return output

    def add_block(self, _input, t, layers_per_block):
        """
        Dense Block
        :param _input: feature map
        :param growth_rate: growth rate
        :param layers_per_block: 12
        :return: feature map
        """
        output = _input #necessary,if removed,every _input will be the same
        print(output)
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, t)
                print("output:")
                print(output)
        return output

    def transition_layer(self, _input):
        """
        1x1 L-Conv and 2x2 Avg pool
        :param _input: feature map
        :return: down sample feature map
        """
        output = _input
        if reduction != 1:
            out_features = int(int(_input.get_shape()[-1])*reduction)
            output = LearnedGroupConvTest(name='lgc', _input=_input, kernel_size=1, out_channels=out_features)
        output = avg_pool(output, k=2)
        return output

    def transition_layer_to_classes(self, _input):
        """
        last transition to get probabilites by classes
        BN-Relu-Global_Avg_Pool-FC
        :param _input: feature map
        :return: logits
        """
        output = batch_norm(_input, is_training=self.is_training)
        output = tf.nn.relu(output)
        last_pool_kernel = int(output.get_shape()[-2])
        output = avg_pool(output, k=last_pool_kernel)
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = weight_variable_xavier([features_total, class_num], name='W')
        print("W")
        print(W)
        bias = bias_variable([class_num])
        logits = tf.matmul(output, W) + bias
        return logits

    def ConDense_net(self, _input):
        """
        ConDense_net Compute graph
        :param _input: feature map
        :return: prediction
        """
        #initial 3x3 Conv
        with tf.variable_scope("Initial_convolution"):
            output = conv2d(_input, out_features=2 * self.growth_rate, kernel_size=3)
        with tf.variable_scope("Block_1"):
            output = self.add_block(output, igr[0], self.layers_per_block)
        with tf.variable_scope("Transition_layer_after_block1"):
            output = self.transition_layer(output)
        with tf.variable_scope("Block_2"):
            output = self.add_block(output, igr[1], self.layers_per_block)
        with tf.variable_scope("Transition_layer_after_block2"):
            output = self.transition_layer(output)
        with tf.variable_scope("Block_3"):
            output = self.add_block(output, igr[2], self.layers_per_block)
        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output)
        return logits


if __name__ == "__main__":

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = noralize_image(train_x, test_x)
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels], name='input_images')
    label = tf.placeholder(tf.float32, shape=[None, class_num], name='labels')
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    logits = ConDenseNet(_input=x, growth_rate=growthRate,
                         is_training=training_flag,
                         layers_per_block=layerPerBlock,
                         expansion=exp).model
    prediction = tf.nn.softmax(logits)

    # Losses
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    # lasso_loss = tf.add_n(tf.get_collection('lasso'))

    # optimizer and train step
    optimizer = tf.train.MomentumOptimizer(learning_rate, nesterov_momentum, use_nesterov=True)
    train_step = optimizer.minimize(cross_entropy + l2_loss * weight_decay)

    correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(tf.global_variables())

    config = tf.ConfigProto()
    # restrict model GPU memory utilization to min required
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore model from ckpt...")
        else:
            sess.run(tf.global_variables_initializer())
            print("Initialize model...")

        """
        compute FLOPs and params
        """
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(tf.Session().graph, run_meta=run_meta, cmd='op', options=opts)
        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(tf.Session().graph, run_meta=run_meta, cmd='op', options=opts)
        print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

        # inference
        start_time = time.time()
        test_acc, test_loss = Evaluate(sess)
        end_time = time.time()
        line = "test_loss: %.4f, test_acc: %.4f\n" % (float(test_loss), float(test_acc))
        print(line)
        print("Time cost: %f" % (end_time-start_time))