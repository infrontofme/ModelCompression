"""
mobilenetv2.py on CIFAR-100 dataset
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from datasets.cifar100 import *
from utils.utils_mob import *
from hp_mobilenet import *


def Evaluate(sess):
    """
    inference phase
    :param sess: TensorFlow session
    :return: test acc, test loss
    """
    total_loss = []
    total_acc = []
    test_pre_index = 0
    add = 1000

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


class MobileNet(object):
    def __init__(self, inputs, num_classes=100, is_training=True, width_multiplier=1.0, exp=6, scope="MobileNet"):
        """
        Bulid mobilenet ref paper https://arxiv.org/abs/1704.04861v1,
        MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
        :param inputs: tensor, input_image
        :param num_classes: int, number of classes
        :param is_training: bool, whether or not it is in training mode
        :param width_multiplier: float, control the size of model
        :param exp: int, expansion ratio
        :param scope: str, optional scope for variables
        """
        self.inputs = inputs #[N, 32, 32, 3]
        self.num_classes = num_classes
        self.is_training = is_training
        self.width_multiplier = width_multiplier
        self.exp = exp

        #build model
        with tf.variable_scope(scope):
            #init_conv1-bn-relu6
            net = conv2d(inputs, "conv1_1", round(32 * self.width_multiplier), filter_size=3, strides=1) #[N, 32 32, 32]
            net = batchnorm(net, "conv1_1/bn", is_training=self.is_training)
            net = tf.nn.relu6(net)
            print(net)

            net = self._res_block(net, 1, 16, 1, name='res2_1') #[N, 32, 32, 16]

            net = self._res_block(net, self.exp, 24, 1, name='res3_1') #[N, 32, 32, 24]
            net = self._res_block(net, self.exp, 24, 1, name='res3_2') #[N, 32, 32, 24]

            net = self._res_block(net, self.exp, 32, 1, name='res4_1') #[N, 32, 32, 32]
            net = self._res_block(net, self.exp, 32, 1, name='res4_2') #[N, 32, 32, 32]
            net = self._res_block(net, self.exp, 32, 1, name='res4_3') #[N, 32, 32, 32]

            net = self._res_block(net, self.exp, 64, 2, name='res5_1', shortcut=False) #[N, 16, 16, 64]
            net = self._res_block(net, self.exp, 64, 1, name='res5_2') #[N, 16, 16, 64]
            net = self._res_block(net, self.exp, 64, 1, name='res5_3') #[N, 16, 16, 64]
            net = self._res_block(net, self.exp, 64, 1, name='res5_4') #[N, 16, 16, 64]

            net = self._res_block(net, self.exp, 96, 1, name='res6_1') #[N, 16, 16, 96]
            net = self._res_block(net, self.exp, 96, 1, name='res6_2') #[N, 16, 16, 96]
            net = self._res_block(net, self.exp, 96, 1, name='res6_3') #[N, 16, 16, 96]

            net = self._res_block(net, self.exp, 160, 2, name='res7_1', shortcut=False) #[N, 8, 8, 160]
            net = self._res_block(net, self.exp, 160, 1, name='res7_2') #[N, 8, 8, 160]
            net = self._res_block(net, self.exp, 160, 1, name='res7_3') #[N, 8, 8, 160]

            net = self._res_block(net, self.exp, 320, 1, name='res8_1', shortcut=False) #[N, 8, 8, 320]

            net = conv2d(net, 'conv9_1', 1280) #[N, 8, 8, 1280]
            net = batchnorm(net, 'conv9_1/bn', is_training=self.is_training)
            net = tf.nn.relu6(net)

            net = avg_pool(net, 8, "avg_pool_10") #[N, 1, 1, 1280]
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            self.logits = fc(net, self.num_classes, "fc_11")
            self.predictions = tf.nn.softmax(self.logits)

    def _res_block(self, inputs, expansion_ratio, output_channels, stride, name, shortcut=True):
        """
        bottleneck residual block 1x1 expansion(bn,relu6) + 3x3 dw(bn,relu6) + 1x1 pj(bn)
        :param inputs: tensor, feature map
        :param expansion_ratio: int, default=6
        :param output_channels: output features num
        :param stride: 2 downsample true else 1
        :param name: str
        :param shortcut: bool, if true element wise add for stride==1
        :return: tensor, feature map
        """
        with tf.variable_scope(name):
            #pw-bn-relu6
            expansion_channels = round(expansion_ratio * inputs.get_shape().as_list()[-1])
            print(expansion_channels)
            # net = conv2dblock(inputs=inputs, scope='pw', num_filters=expansion_channels)
            # net = groupconv2d(name="pw", inputs=inputs, out_features=expansion_channels, kernel_size=1, strides=[1, 1, 1, 1])
            # net = self._shufflelayers(net)
            net = LearnedGroupConvTest(name='pw', _input=inputs, kernel_size=1, out_channels=expansion_channels)
            net = batchnorm(net, 'pw_bn', is_training=self.is_training)
            net = tf.nn.relu6(net)
            # net = dropout(net, self.is_training)
            print(net)

            #dw-bn-relu6
            net = depthwise_conv2d(net, 'dw', strides=stride)
            net = batchnorm(net, 'dw_bn', is_training=self.is_training)
            net = tf.nn.relu6(net)
            # net = dropout(net, self.is_training)
            print(net)

            #pw-bn
            # net = conv2dblock(inputs=net, scope='pw_linear', num_filters=output_channels)
            # net = groupconv2d(name="pw_linear", inputs=net, out_features=output_channels, kernel_size=1, strides=[1, 1, 1, 1])
            # net = self._shufflelayers(net)
            net = LearnedGroupConvTest(name='pw_linear', _input=net, kernel_size=1, out_channels=output_channels)
            net = batchnorm(net, 'pw_linear_bn', is_training=self.is_training)

            #element wise add, only for stride==1
            if shortcut and stride==1:
                in_channels = int(inputs.get_shape()[-1])
                if in_channels != output_channels:
                    ins = conv2d(inputs, 'ex_dim', output_channels)
                    # ins = groupconv2d(name='ex_dim', inputs=inputs, out_features=output_channels, kernel_size=1, strides=[1, 1, 1, 1])
                    # ins = self._shufflelayers(ins)
                    # ins = learned_group_conv2d(name='ex_dim', _input=inputs, kernel_size=1, out_channels=output_channels)
                    net = ins + net
                else:
                    net = inputs + net
            print(net)

            return net

    def _shufflelayers(self, inputs):
        """
        channel shuffle after 1x1 G-Conv
        :param inputs: tensor, feature map
        :return: feature map
        """
        height = int(inputs.get_shape()[1])
        width = int(inputs.get_shape()[2])
        features_num = int(inputs.get_shape()[3])
        features_per_group = features_num // group
        # shuffle
        inputs = tf.reshape(inputs, shape=[-1, height, width, features_per_group, group])
        inputs = tf.transpose(inputs, perm=[0, 1, 2, 4, 3])
        output = tf.reshape(inputs, shape=[-1, height, width, features_num])
        return output


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = noralize_image(train_x, test_x)
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels], name='input_image')
    label = tf.placeholder(tf.float32, shape=[None, class_num], name='labels')
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    mobileNet = MobileNet(inputs=x, num_classes=class_num,
                          is_training=training_flag, width_multiplier=widthMultiple,
                          exp=expansionRatio)
    logits = mobileNet.logits
    prediction = mobileNet.predictions

    #Losses
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    # lasso_loss = tf.add_n(tf.get_collection('lasso'))

    #optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate, nesterov_momentum, use_nesterov=True)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=momentum, name='RMSProp')
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
        print("Time cost: %f" % (end_time - start_time))