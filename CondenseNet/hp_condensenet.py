'''
Hyper Parameters of CondenseNet
'''

# growth rate
growthRate = 16 # control each layer output channels
layerPerBlock = 15 # how many layer in one block
exp = 4 # control 1x1 bottleneck output channels
igr = [1, 1, 1] # increasing growth rate parameters, if set [1, 2, 4], means exponentially growth

# learning rate
init_learning_rate = 0.1

# drop out rate
keep_prob = 0.9

# lgc / gc
group = 4
condense_factor = 4

# for 1x1 Conv at transition layer, only CondenseNet-light have 1x1 Conv at transition layer,
# if set to 1, means, no 1x1 Conv at Transition layer.
reduction = 0.5

# Momentum Optimizer
nesterov_momentum = 0.9

# l2 regularizer
weight_decay = 1e-4

# batch_size, train_set_number = batch_size * iteration
batch_size = 64
iteration = 782
test_iteration = 10
total_epochs = 300

# logs dir
train_log_dir = './CondenseNet/logs/run1/'

# ckpt model dir
ckpt_dir = './CondenseNet/model/run1/'

