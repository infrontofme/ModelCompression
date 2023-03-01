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


def dropping(stage):
    """
    Pruning 1x1 L-CONV layer weight during training
    :param stage: int, which stage
    :return: masked_weight
    """
    print("Dropping...")
    weights = tf.get_collection('learned_group_layer')
    masks = tf.get_collection('mask')
    for w, m in zip(weights, masks):
        _dropping(w, m, stage)


def _dropping(weight, mask, stage):
    """
    Pruning weight in 1x1 learned group layer
    :param weight: weight
    :param mask: mask, shape=weight.shape
    :param stage: int, which stage
    :return: masked, masked_weight
    """
    s_time = time.time()
    out_channels = int(weight.get_shape()[-1])
    in_channels = int(weight.get_shape()[-2])
    delta = in_channels // condense_factor
    d_out = out_channels // group

    weight_s = tf.abs(tf.squeeze(tf.multiply(weight, mask)))

    #shuffle
    weight_s = tf.reshape(weight_s, [in_channels, d_out, group])
    weight_s = tf.transpose(weight_s, [0, 2, 1])
    weight_s = tf.reshape(weight_s, [in_channels, out_channels])

    k = delta * stage
    print("k: %d" % k)

    control_ops = []
    for i in range(group):
        wi = weight_s[:, i*d_out:(i+1)*d_out] #(, out_channels)
        _, index = tf.nn.top_k(tf.reduce_sum(wi, axis=1)*(-1), k=k, sorted=True)
        for j in range(delta):
            idx = index[tf.cast((stage-1)*delta + j, tf.int32)]
            op = mask[:, :, idx, i::group].assign(np.zeros([1, 1, d_out]))
            control_ops.append(op)
    with tf.control_dependencies(control_ops):
        sess.run(tf.identity(mask))
    e_time = time.time()
    print("Time cost in dropping one layer: %f" % (e_time - s_time))


def getparams(epoch):
    """
    output all weight or mask or masked_weight in learned group layer at specific epoch
    :param epoch: int, training epoch
    :return: txt record file
    """
    print("recording weight and mask in learned group layer...")
    weights = tf.get_collection('learned_group_layer')
    masks = tf.get_collection('mask')
    with open('weight_at_epoch_%d.txt' % epoch, 'a') as weightfile:
        with open('mask_at_epoch_%d.txt' % epoch, 'a') as maskfile:
            for w, m in zip(weights, masks):
                np.savetxt(weightfile, sess.run(tf.squeeze(w)), fmt='%7.4f')
                weightfile.write('#New weight\n')

                np.savetxt(maskfile, sess.run(tf.squeeze(m)), fmt='%7.4f')
                maskfile.write("#New mask\n")

            weightfile.write('\n')
            weightfile.write('recording end...\n')

            maskfile.write('\n')
            maskfile.write('recording end...\n')

        weightfile.close()
        maskfile.close()
    print("recording finished at epoch: %d" % epoch)


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
            net = learned_group_conv2d(name='pw', _input=inputs, kernel_size=1, out_channels=expansion_channels)
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
            net = learned_group_conv2d(name='pw_linear', _input=net, kernel_size=1, out_channels=output_channels)
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

        train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

        check_drop = int(total_epochs // (2 * (condense_factor - 1)))
        for epoch in range(1, total_epochs+1):
            epoch_learning_rate = cosine_learning_rate(init_learning_rate, total_epochs, epoch, iteration, iteration-1)
            # epoch_learning_rate = epoch_learning_rate * learning_rate_decay
            print("shuffle training date every epoch...")
            indices = np.random.permutation(len(train_x))
            train_x = train_x[indices]
            train_y = train_y[indices]
            print("shuffle finished...")

            # pruning
            if epoch == int(check_drop * 1) and epoch <= int(total_epochs // 2):
            # if epoch == 1:
                stage = 1
                print("stage: %d" % stage)
                dropping(stage)
                getparams(epoch)

            if epoch == int(check_drop * 2) and epoch <= int(total_epochs // 2):
                stage = 2
                print("stage: %d" % stage)
                dropping(stage)
                getparams(epoch)

            if epoch == int(check_drop * 3) and epoch <= int(total_epochs // 2):
                stage = 3
                print("stage: %d" % stage)
                dropping(stage)
                getparams(epoch)

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