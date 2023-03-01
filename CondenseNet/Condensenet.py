'''
CondenseNet-light-94
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import math
import time
import numpy as np
import tensorflow as tf

# import cifar-100
from datasets.cifar100 import *
from utils.utils_con import *


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
    return test_acc,test_loss


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
            output = learned_group_conv2d(name='lgc', _input=output, kernel_size=1, out_channels=out_features)
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
            output = learned_group_conv2d(name='lgc', _input=_input, kernel_size=1, out_channels=out_features)
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
    optimizer = tf.train.MomentumOptimizer(learning_rate, nesterov_momentum,use_nesterov=True)
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

        train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        check_drop = int(total_epochs // (2 * (condense_factor - 1)))
        for epoch in range(1, total_epochs+1):
            epoch_learning_rate = cosine_learning_rate(init_learning_rate, total_epochs, epoch, iteration, iteration-1)
            print("shuffle training data every epoch...")
            indices = np.random.permutation(len(train_x))
            train_x = train_x[indices]
            train_y = train_y[indices]
            print("shuffle finished...")

            #pruning
            if epoch == int(check_drop*1) and epoch <= int(total_epochs // 2):
                stage = 1
                dropping(stage)
                getparams(epoch)
            if epoch == int(check_drop*2) and epoch <= int(total_epochs // 2):
                stage = 2
                dropping(stage)
                getparams(epoch)
            if epoch == int(check_drop*3) and epoch <= int(total_epochs // 2):
                stage = 3
                dropping(stage)
                getparams(epoch)

            pre_index = 0
            total_loss = []
            total_acc = []
            start_time = time.time()
            for step in range(1, iteration+1):
                if pre_index+batch_size < 50000:
                    batch_x = train_x[pre_index:pre_index+batch_size]
                    batch_y = train_y[pre_index:pre_index+batch_size]
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
            test_acc,test_loss = Evaluate(sess)
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
            print("Time cost: %f" % (end_time-start_time))

            with open('logs.txt', 'a') as f:
                f.write(line)
            saver.save(sess=sess, save_path=ckpt_dir + 'Condensenetl94k16.ckpt')