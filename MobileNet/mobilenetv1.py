"""
Tensorflow implementation of MobileNetV1 on cifar100 datasets

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


def Evaluate(sess):
    total_loss = []
    total_acc = []
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index:test_pre_index+add]
        test_batch_y = test_y[test_pre_index:test_pre_index+add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x:test_batch_x,
            label:test_batch_y,
            learning_rate:epoch_learning_rate,
            training_flag:False
        }

        loss_,acc_ = sess.run([cross_entropy,accuracy],feed_dict=test_feed_dict)
        total_loss.append(loss_)
        total_acc.append(acc_)
    test_loss = np.mean(total_loss)
    test_acc = np.mean(total_acc)
    return test_acc,test_loss


class MobileNet(object):
    def __init__(self, inputs, num_classes=100, is_training=True, width_multiplier=1.0, scope="MobileNet"):
        """
        Bulid mobilenet ref paper https://arxiv.org/abs/1704.04861v1,
        MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
        :param inputs: tensor, input_image
        :param num_classes: int, number of classes
        :param is_training: bool, whether or not it is in training mode
        :param width_multiplier: float, control the size of model
        :param scope: str, optional scope for variables
        """
        self.inputs = inputs
        self.num_classes = num_classes
        self.is_training = is_training
        self.width_multiplier = width_multiplier

        #build model
        with tf.variable_scope(scope):
            #init_conv1
            net = conv2d(inputs, "conv_1", round(32 * self.width_multiplier), filter_size=3, strides=1) #[N, 32 32, 3]
            net = batchnorm(net, "conv_1/bn", is_training=self.is_training)
            net = tf.nn.relu6(net)

            net = self._depthwise_separable_conv2d(net, 64, self.width_multiplier, "ds_conv_2") #[N, 32, 32, 64]
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier, "ds_conv_3", downsample=False) #[N, 32, 32, 128]
            net = self._depthwise_separable_conv2d(net, 128, self.width_multiplier, "ds_conv_4") #[N, 56, 56, 128]
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier, "ds_conv_5", downsample=False) #[N, 32, 32, 256]
            net = self._depthwise_separable_conv2d(net, 256, self.width_multiplier, "ds_conv_6") #[N, 28, 28, 256]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_7", downsample=True) #[N, 16, 16, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_8") #[N, 16, 16, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_9") #[N, 16, 16, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_10") # [N, 16, 16, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_11") # [N, 16, 16, 512]
            net = self._depthwise_separable_conv2d(net, 512, self.width_multiplier, "ds_conv_12") # [N, 16, 16, 512]
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier, "ds_conv_13", downsample=True) #[N, 8, 8, 1024]
            net = self._depthwise_separable_conv2d(net, 1024, self.width_multiplier, "ds_conv_14") #[N, 8, 8, 1024]

            net = avg_pool(net, 8, "avg_pool_15")
            net = tf.squeeze(net, [1, 2], name="SpatialSqueeze")
            self.logits = fc(net, self.num_classes, "fc_16")
            self.predictions = tf.nn.softmax(self.logits)

    def _depthwise_separable_conv2d(self, inputs, num_filters, width_multiplier, scope, downsample=False):
        """
        depthwise separable convolution 2D function
        :param inputs: tensor, feature map
        :param num_filters: int, output feature nums
        :param width_multiplier: float, control the nums of output channels
        :param scope: str, context manager
        :param downsample: bool, whether or not the strides =1 or 2. True for 2, False for 1
        :return: tensor, feature map
        """
        num_filters = round(num_filters * width_multiplier)
        strides = 2 if downsample else 1

        with tf.variable_scope(scope):
            #depthwise conv2d - BN -relu6
            dw_conv = depthwise_conv2d(inputs, "depthwise_conv", strides=strides)
            bn = batchnorm(dw_conv, "dw_bn", is_training=self.is_training)
            relu = tf.nn.relu6(bn)

            #pointwise conv2d(1x1) - BN - relu6
            pw_conv = conv2d(relu, "pointwise_conv", num_filters)
            bn = batchnorm(pw_conv, "pw_bn", is_training=self.is_training)
            output = tf.nn.relu6(bn)

            return output

if __name__ == "__main__":
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = noralize_image(train_x, test_x)
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels], name='input_image')
    label = tf.placeholder(tf.float32, shape=[None, class_num], name='labels')
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    mobileNet = MobileNet(inputs=x, num_classes=class_num, is_training=training_flag, width_multiplier=widthMultiple)
    logits = mobileNet.logits
    prediction = mobileNet.predictions

    #Losses
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

    #optimizer
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

        """
        compute FLOPs and params
        """
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(tf.Session().graph, run_meta=run_meta, cmd='op', options=opts)
        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(tf.Session().graph, run_meta=run_meta, cmd='op', options=opts)
        print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

        train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        for epoch in range(1, total_epochs+1):
            epoch_learning_rate = cosine_learning_rate(init_learning_rate, total_epochs, epoch, iteration, iteration-1)
            print("shuffle training date every epoch...")
            indices = np.random.permutation(len(train_x))
            train_x = train_x[indices]
            train_y = train_y[indices]
            print("shuffle finished...")

            pre_index = 0
            total_loss = []
            total_acc = []
            start_time = time.time()
            for step in range(1, iteration+1):
                if pre_index + batch_size < 50000:
                    batch_x = train_x[pre_index: pre_index + batch_size]
                    batch_y = train_y[pre_index: pre_index + batch_size]
                else:
                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]

                #data augmentation
                batch_x = data_augmentation(batch_x)

                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }
                _, batch_loss, batch_acc = sess.run([train_step, cross_entropy, accuracy], feed_dict=train_feed_dict)
                total_loss.append(batch_loss)
                total_acc.append(batch_acc)
                pre_index += batch_size

            train_loss = np.mean(total_loss)
            train_acc = np.mean(total_acc)
            test_acc, test_loss = Evaluate(sess)
            end_time = time.time()

            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=train_acc),
                                              tf.Summary.Value(tag='learning_rate', simple_value=epoch_learning_rate),
                                              tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                              tf.Summary.Value(tag='test_acc', simple_value=test_acc)])
            train_summary_writer.add_summary(summary=train_summary, global_step=epoch)
            train_summary_writer.flush()

            line = "epoch: %d/%d, train_loss: %.4f, train_acc:%.4f, test_loss: %.4f, test_acc:%.4f\n" % (
                epoch, total_epochs, float(train_loss), float(train_acc), float(test_loss), float(test_acc))
            print(line)
            print("Time cost: %f" % (end_time - start_time))

            with open('logs.txt', 'a') as f:
                f.write(line)
            saver.save(sess=sess, save_path=ckpt_dir + 'mobilenetv1.ckpt')