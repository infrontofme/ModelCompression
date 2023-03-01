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
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cross_entropy, accuracy], feed_dict=test_feed_dict)
        total_loss.append(loss_)
        total_acc.append(acc_)
    test_loss = np.mean(total_loss)
    test_acc = np.mean(total_acc)
    return test_acc, test_loss


def getparams(epoch):
    """
    record weight in expansion and projection layer
    :param epoch: global epoch
    :return: txt record file
    """
    print("rcording weight in expansion and projection layer....")
    weights = tf.get_collection('weights_in_block')
    with open('modifyLearningStrategy_weight_at_epoch_%d.txt' % epoch, 'a') as outfile:
        for w in zip(weights):
            np.savetxt(outfile, sess.run(tf.squeeze(w)), fmt='%7.4f')
            outfile.write("New weight\n")
        outfile.write('\n')
        outfile.write('recording end...\n')
        outfile.close()
    print("recording finished at epoch: %d" % epoch)


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
            net = conv2dblock(inputs=inputs, scope='pw', num_filters=expansion_channels)
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
            net = conv2dblock(inputs=net, scope='pw_linear', num_filters=output_channels)
            net = batchnorm(net, 'pw_linear_bn', is_training=self.is_training)

            #element wise add, only for stride==1
            if shortcut and stride==1:
                in_channels = int(inputs.get_shape()[-1])
                if in_channels != output_channels:
                    ins = conv2d(inputs, 'ex_dim', output_channels)
                    net = ins + net
                else:
                    net = inputs + net
            print(net)

            return net


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = noralize_image(train_x, test_x)
    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels], name='input_image')
    label = tf.placeholder(tf.float32, shape=[None, class_num], name='labels')
    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    mobileNet = MobileNet(inputs=x, num_classes=class_num, is_training=training_flag,
                          width_multiplier=widthMultiple, exp=expansionRatio)
    logits = mobileNet.logits
    prediction = mobileNet.predictions

    #Losses
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

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

        train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        for epoch in range(1, total_epochs+1):
            epoch_learning_rate = cosine_learning_rate(init_learning_rate, total_epochs, epoch, iteration, iteration-1)
            # epoch_learning_rate = epoch_learning_rate * learning_rate_decay
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

            if epoch == total_epochs:
                getparams(epoch)

            with open('logs.txt', 'a') as f:
                f.write(line)
            saver.save(sess=sess, save_path=ckpt_dir + 'mobilenetv2.ckpt')