'''
Hyper Parameters of MobileNetV1 and MobileNetV2
'''


# learning rate
init_learning_rate = 0.1

# width multiple
widthMultiple = 0.5

# expansion ratio only for V2
expansionRatio = 1

# Momentum Optimizer
nesterov_momentum = 0.9

# RMSProp Optimizer
decay = 0.9
momentum = 0.9

# l2 regularizer
weight_decay = 1e-4

# dropout rate
keep_prob = 0.9

# lgc / gc
group = 4
condense_factor = 4
lasso_decay = 1e-5

# batch size, train_set_number = batch_size * iteration
batch_size = 64
iteration = 782
test_iteration = 10
total_epochs = 400

# logs dir
train_log_dir = './MobileNet/logs/run1/'

# check point model dir
ckpt_dir = './MobileNet/model/run1/'

