#
# pydeepdelta demo for CIFAR-10 ResNet18
# Custom file pydeepdelta_bn.py is a modified version of the master branch's pydeepdelta.py; it contains support for 
# batch noramlization layers and is used just in this example.
# author: Geir K. Nilsen (geir.kjetil.nilsen@gmail.com), 2021
#

import tqdm
import numpy as np
from utils import *
import os
import random

import pdb

import tensorflow.compat.v1 as tf
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.disable_v2_behavior()

# Import CIFAR-10 data & convert 2D images to scaled 1D vectors, and labels to one hot format
cifar10 = tf.keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
N_train = X_train.shape[0]
N_test = X_test.shape[0]
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_one_hot(np.squeeze((y_train)))
y_test = to_one_hot(np.squeeze((y_test)))


# Close current graph session if any is already open
if('sess' in locals()):
    sess.close()
tf.reset_default_graph()

# Parameters
learning_rate = tf.placeholder(tf.float32)
dropout_rate = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool, name="is_training")
reg_lambda_val = 0.01
num_steps = 115000
TL = 10

# Set random seed for repeatability.
rseed=0
np.random.seed(rseed)
random.seed(rseed)
tf.random.set_random_seed(seed=rseed)

# Random weight init
weights = {
'wc1': tf.Variable(tf.random.uniform((3, 3, 3, 32*2),    minval=-np.sqrt(2)*np.sqrt(6/(32*32*3 + 32*32*64)), maxval=np.sqrt(2)*np.sqrt(6/(32*32*3 + 32*32*64))), dtype='float32'),
'wc2': tf.Variable(tf.random.uniform((3, 3, 32*2, 32*2), minval=-np.sqrt(2)*np.sqrt(6/(32*32*64 + 32*32*64)), maxval=np.sqrt(2)*np.sqrt(6/(32*32*64 + 32*32*64))), dtype='float32'),
'wc3': tf.Variable(tf.random.uniform((3, 3, 32*2, 32*2), minval=-np.sqrt(2)*np.sqrt(6/(32*32*64 + 32*32*64)), maxval=np.sqrt(2)*np.sqrt(6/(32*32*64 + 32*32*64))), dtype='float32'),
'wc4': tf.Variable(tf.random.uniform((3, 3, 32*2, 32*2), minval=-np.sqrt(2)*np.sqrt(6/(32*32*64 + 32*32*64)), maxval=np.sqrt(2)*np.sqrt(6/(32*32*64 + 32*32*64))), dtype='float32'),
'wc5': tf.Variable(tf.random.uniform((3, 3, 32*2, 32*2), minval=-np.sqrt(2)*np.sqrt(6/(32*32*64 + 32*32*64)), maxval=np.sqrt(2)*np.sqrt(6/(32*32*64 + 32*32*64))), dtype='float32'),

'wc6': tf.Variable(tf.random.uniform((3, 3, 32*2, 64*2), minval=-np.sqrt(2)*np.sqrt(6/(32*32*64 + 16*16*128)), maxval=np.sqrt(2)*np.sqrt(6/(32*32*64 + 16*16*128))), dtype='float32'),
'wc6s': tf.Variable(tf.random.uniform((1, 1, 32*2, 64*2), minval=-np.sqrt(2)*np.sqrt(6/(32*32*64 + 16*16*128)), maxval=np.sqrt(2)*np.sqrt(6/(32*32*64 + 16*16*128))), dtype='float32'),
'wc7': tf.Variable(tf.random.uniform((3, 3, 64*2, 64*2), minval=-np.sqrt(2)*np.sqrt(6/(16*16*128 + 16*16*128)), maxval=np.sqrt(2)*np.sqrt(6/(16*16*128 + 16*16*128))), dtype='float32'),
'wc8': tf.Variable(tf.random.uniform((3, 3, 64*2, 64*2), minval=-np.sqrt(2)*np.sqrt(6/(16*16*128 + 16*16*128)), maxval=np.sqrt(2)*np.sqrt(6/(16*16*128 + 16*16*128))), dtype='float32'),
'wc9': tf.Variable(tf.random.uniform((3, 3, 64*2, 64*2), minval=-np.sqrt(2)*np.sqrt(6/(16*16*128 + 16*16*128)), maxval=np.sqrt(2)*np.sqrt(6/(16*16*128 + 16*16*128))), dtype='float32'),

'wc10': tf.Variable(tf.random.uniform((3, 3, 64*2, 128*2), minval=-np.sqrt(2)*np.sqrt(6/(16*16*128 + 8*8*256)), maxval=np.sqrt(2)*np.sqrt(6/(16*16*128 + 8*8*256))), dtype='float32'),
'wc10s': tf.Variable(tf.random.uniform((1, 1, 64*2, 128*2), minval=-np.sqrt(2)*np.sqrt(6/(16*16*128 + 8*8*256)), maxval=np.sqrt(2)*np.sqrt(6/(16*16*128 + 8*8*256))), dtype='float32'),
'wc11': tf.Variable(tf.random.uniform((3, 3, 128*2, 128*2), minval=-np.sqrt(2)*np.sqrt(6/(8*8*256 + 8*8*256)), maxval=np.sqrt(2)*np.sqrt(6/(8*8*256 + 8*8*256))), dtype='float32'),
'wc12': tf.Variable(tf.random.uniform((3, 3, 128*2, 128*2), minval=-np.sqrt(2)*np.sqrt(6/(8*8*256 + 8*8*256)), maxval=np.sqrt(2)*np.sqrt(6/(8*8*256 + 8*8*256))), dtype='float32'),
'wc13': tf.Variable(tf.random.uniform((3, 3, 128*2, 128*2), minval=-np.sqrt(2)*np.sqrt(6/(8*8*256 + 8*8*256)), maxval=np.sqrt(2)*np.sqrt(6/(8*8*256 + 8*8*256))), dtype='float32'),

'wc14': tf.Variable(tf.random.uniform((3, 3, 128*2, 256*2), minval=-np.sqrt(2)*np.sqrt(6/(8*8*256 + 4*4*512)), maxval=np.sqrt(2)*np.sqrt(6/(8*8*256 + 4*4*512))), dtype='float32'),
'wc14s': tf.Variable(tf.random.uniform((1, 1, 128*2, 256*2), minval=-np.sqrt(2)*np.sqrt(6/(8*8*256 + 4*4*512)), maxval=np.sqrt(2)*np.sqrt(6/(8*8*256 + 4*4*512))), dtype='float32'),
'wc15': tf.Variable(tf.random.uniform((3, 3, 256*2, 256*2), minval=-np.sqrt(2)*np.sqrt(6/(4*4*512 + 4*4*512)), maxval=np.sqrt(2)*np.sqrt(6/(4*4*512 + 4*4*512))), dtype='float32'),
'wc16': tf.Variable(tf.random.uniform((3, 3, 256*2, 256*2), minval=-np.sqrt(2)*np.sqrt(6/(4*4*512 + 4*4*512)), maxval=np.sqrt(2)*np.sqrt(6/(4*4*512 + 4*4*512))), dtype='float32'),
'wc17': tf.Variable(tf.random.uniform((3, 3, 256*2, 256*2), minval=-np.sqrt(2)*np.sqrt(6/(4*4*512 + 4*4*512)), maxval=np.sqrt(2)*np.sqrt(6/(4*4*512 + 4*4*512))), dtype='float32'),

'out': tf.Variable(tf.random.uniform((256*2, TL), minval=-np.sqrt(2)*np.sqrt(6/(4*4*512 + 10)), maxval=np.sqrt(2)*np.sqrt(6/(4*4*512 + 10))), dtype='float32')

}

# Biases init to zero
biases = {
'bc1': tf.Variable(tf.zeros((32*2)), dtype='float32'),
'bc2': tf.Variable(tf.zeros((32*2)), dtype='float32'),
'bc3': tf.Variable(tf.zeros((32*2)), dtype='float32'),
'bc4': tf.Variable(tf.zeros((32*2)), dtype='float32'),
'bc5': tf.Variable(tf.zeros((32*2)), dtype='float32'),

'bc6': tf.Variable(tf.zeros((64*2)), dtype='float32'),
'bc6s': tf.Variable(tf.zeros((64*2)), dtype='float32'),
'bc7': tf.Variable(tf.zeros((64*2)), dtype='float32'),
'bc8': tf.Variable(tf.zeros((64*2)), dtype='float32'),
'bc9': tf.Variable(tf.zeros((64*2)), dtype='float32'),

'bc10': tf.Variable(tf.zeros((128*2)), dtype='float32'),
'bc10s': tf.Variable(tf.zeros((128*2)), dtype='float32'),
'bc11': tf.Variable(tf.zeros((128*2)), dtype='float32'),
'bc12': tf.Variable(tf.zeros((128*2)), dtype='float32'),
'bc13': tf.Variable(tf.zeros((128*2)), dtype='float32'),

'bc14': tf.Variable(tf.zeros((256*2)), dtype='float32'),
'bc14s': tf.Variable(tf.zeros((256*2)), dtype='float32'),
'bc15': tf.Variable(tf.zeros((256*2)), dtype='float32'),
'bc16': tf.Variable(tf.zeros((256*2)), dtype='float32'),
'bc17': tf.Variable(tf.zeros((256*2)), dtype='float32'),

'out': tf.Variable(tf.zeros((TL)), dtype='float32')

}

# Stack parameters layer-wise, weights first then biases
params = [weights['wc1'], biases['bc1'], 
      weights['wc2'], biases['bc2'], 
      weights['wc3'], biases['bc3'],
      weights['wc4'], biases['bc4'],
      weights['wc5'], biases['bc5'],
      weights['wc6'], biases['bc6'],
      weights['wc6s'], biases['bc6s'],
      weights['wc7'], biases['bc7'],
      weights['wc8'], biases['bc8'],
      weights['wc9'], biases['bc9'],
      weights['wc10'], biases['bc10'],
      weights['wc10s'], biases['bc10s'],
      weights['wc11'], biases['bc11'],
      weights['wc12'], biases['bc12'],
      weights['wc13'], biases['bc13'],
      weights['wc14'], biases['bc14'],
      weights['wc14s'], biases['bc14s'],
      weights['wc15'], biases['bc15'],
      weights['wc16'], biases['bc16'],
      weights['wc17'], biases['bc17'],      
      weights['out'], biases['out']
      ]
      

# Model input/output
X = tf.placeholder(dtype='float32', shape=(None, 32, 32, 3), name="X-data")
y = tf.placeholder(dtype='float32', shape=(None, TL), name="y-data")

# L2-regularization parameter
reg_lambda = tf.placeholder(tf.float32)

# Cost function, data only
def cost_fun_data(y, yhat_logits):
    cost_data = tf.losses.softmax_cross_entropy(y, yhat_logits)
    return cost_data

# Cost function, reg. only
def cost_fun_reg(params):
    cost_reg = reg_lambda/2.0*tf.reduce_sum([tf.reduce_sum(tf.pow(params[x], 2.0)) \
                                             for x in range(len(params))])
    return cost_reg

# Cost function, data + reg.
def cost_fun(y, yhat_logits, params):
    return cost_fun_data(y, yhat_logits) + cost_fun_reg(params)


# Model function
def model_fun(x, params):
   
    conv1 = tf.nn.conv2d(x, params[0], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, params[1])

    print('Layer 1, Input: %s, Output: %s, Weights: %s, biases %s' % (x.shape, conv1.shape, params[0].shape, params[1].shape))

    # Resblock 1

    x_init_1 = conv1

    rb1_1 = tf.layers.batch_normalization(x_init_1, training=is_training)
    rb1_1 = tf.nn.relu(rb1_1)
    rb1_1 = tf.nn.conv2d(rb1_1, params[2], strides=[1, 1, 1, 1], padding='SAME')
    rb1_1 = tf.nn.bias_add(rb1_1, params[3])

    print('Layer 2, Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_1.shape, rb1_1.shape,  params[2].shape, params[3].shape))

    rb1_2 = tf.layers.batch_normalization(rb1_1, training=is_training)
    rb1_2 = tf.nn.relu(rb1_2)
    rb1_2 = tf.nn.conv2d(rb1_2, params[4], strides=[1, 1, 1, 1], padding='SAME')
    rb1_2 = tf.nn.bias_add(rb1_2, params[5])

    rb1_2 = rb1_2 + x_init_1 # Skip connection

    print('Layer 3, Input: %s, Output: %s, Weights: %s, biases %s' % (rb1_1.shape, rb1_2.shape,  params[4].shape, params[5].shape))

    # Resblock 2

    x_init_2 = rb1_2

    rb2_1 = tf.layers.batch_normalization(x_init_2, training=is_training)
    rb2_1 = tf.nn.relu(rb2_1)
    rb2_1 = tf.nn.conv2d(rb2_1, params[6], strides=[1, 1, 1, 1], padding='SAME')
    rb2_1 = tf.nn.bias_add(rb2_1, params[7])

    print('Layer 4, Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_2.shape, rb2_1.shape,  params[6].shape, params[7].shape))

    rb2_2 = tf.layers.batch_normalization(rb2_1, training=is_training)
    rb2_2 = tf.nn.relu(rb2_2)
    rb2_2 = tf.nn.conv2d(rb2_2, params[8], strides=[1, 1, 1, 1], padding='SAME')
    rb2_2 = tf.nn.bias_add(rb2_2, params[9])

    rb2_2 = rb2_2 + x_init_2 # Skip connection

    print('Layer 5, Input: %s, Output: %s, Weights: %s, biases %s' % (rb2_1.shape, rb2_2.shape,  params[8].shape, params[9].shape))

    # Resblock 3

    x_init_3_1 = rb2_2

    rb3_1 = tf.layers.batch_normalization(x_init_3_1, training=is_training)
    rb3_1 = tf.nn.relu(rb3_1)   
    rb3_1 = tf.nn.conv2d(rb3_1, params[10], strides=[1, 2, 2, 1], padding='SAME')
    rb3_1 = tf.nn.bias_add(rb3_1, params[11])

    print('Layer 6, Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_3_1.shape, rb3_1.shape,  params[10].shape, params[11].shape))

    x_init_3_2 = tf.nn.conv2d(x_init_3_1, params[12], strides=[1, 2, 2, 1], padding='SAME')
    x_init_3_2 = tf.nn.bias_add(x_init_3_2, params[13])

    print('Layer 6 (skip), Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_3_1.shape, x_init_3_2.shape,  params[12].shape, params[13].shape))
    
    rb3_2 = tf.layers.batch_normalization(rb3_1, training=is_training)
    rb3_2 = tf.nn.relu(rb3_2)
    rb3_2 = tf.nn.conv2d(rb3_2, params[14], strides=[1, 1, 1, 1], padding='SAME')
    rb3_2 = tf.nn.bias_add(rb3_2, params[15])

    print('Layer 7, Input: %s, Output: %s, Weights: %s, biases %s' % (rb3_1.shape, rb3_2.shape,  params[14].shape, params[15].shape))

    rb3_2 = rb3_2 + x_init_3_2 # Skip connection

    # Resblock 4

    x_init_4 = rb3_2

    rb4_1 = tf.layers.batch_normalization(x_init_4, training=is_training)
    rb4_1 = tf.nn.relu(rb4_1)
    rb4_1 = tf.nn.conv2d(rb4_1, params[16], strides=[1, 1, 1, 1], padding='SAME')
    rb4_1 = tf.nn.bias_add(rb4_1, params[17])

    print('Layer 8, Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_4.shape, rb4_1.shape,  params[16].shape, params[17].shape))

    rb4_2 = tf.layers.batch_normalization(rb4_1, training=is_training)
    rb4_2 = tf.nn.relu(rb4_2)
    rb4_2 = tf.nn.conv2d(rb4_2, params[18], strides=[1, 1, 1, 1], padding='SAME')
    rb4_2 = tf.nn.bias_add(rb4_2, params[19])

    print('Layer 9, Input: %s, Output: %s, Weights: %s, biases %s' % (rb4_1.shape, rb4_2.shape,  params[18].shape, params[19].shape))

    rb4_2 = rb4_2 + x_init_4 # Skip connection    

    # Resblock 5

    x_init_5_1 = rb4_2

    rb5_1 = tf.layers.batch_normalization(x_init_5_1, training=is_training)
    rb5_1 = tf.nn.relu(rb5_1)
    
    rb5_1 = tf.nn.conv2d(rb5_1, params[20], strides=[1, 2, 2, 1], padding='SAME')
    rb5_1 = tf.nn.bias_add(rb5_1, params[21])

    print('Layer 10, Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_5_1.shape, rb5_1.shape,  params[20].shape, params[21].shape))

    x_init_5_2 = tf.nn.conv2d(x_init_5_1, params[22], strides=[1, 2, 2, 1], padding='SAME')
    x_init_5_2 = tf.nn.bias_add(x_init_5_2, params[23])

    print('Layer 10 (skip), Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_5_1.shape, x_init_5_2.shape,  params[22].shape, params[23].shape))    
    
    rb5_2 = tf.layers.batch_normalization(rb5_1, training=is_training)
    rb5_2 = tf.nn.relu(rb5_2)
    rb5_2 = tf.nn.conv2d(rb5_2, params[24], strides=[1, 1, 1, 1], padding='SAME')
    rb5_2 = tf.nn.bias_add(rb5_2, params[25])

    print('Layer 11, Input: %s, Output: %s, Weights: %s, biases %s' % (rb5_1.shape, rb5_2.shape,  params[24].shape, params[25].shape))

    rb5_2 = rb5_2 + x_init_5_2 # Skip connection


    # Resblock 6

    x_init_6 = rb5_2

    rb6_1 = tf.layers.batch_normalization(x_init_6, training=is_training)
    rb6_1 = tf.nn.relu(rb6_1)
    rb6_1 = tf.nn.conv2d(rb6_1, params[26], strides=[1, 1, 1, 1], padding='SAME')
    rb6_1 = tf.nn.bias_add(rb6_1, params[27])

    print('Layer 12, Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_6.shape, rb6_1.shape,  params[26].shape, params[27].shape))

    rb6_2 = tf.layers.batch_normalization(rb6_1, training=is_training)
    rb6_2 = tf.nn.relu(rb6_2)
    rb6_2 = tf.nn.conv2d(rb6_2, params[28], strides=[1, 1, 1, 1], padding='SAME')
    rb6_2 = tf.nn.bias_add(rb6_2, params[29])

    print('Layer 13, Input: %s, Output: %s, Weights: %s, biases %s' % (rb6_1.shape, rb6_2.shape,  params[28].shape, params[29].shape))

    rb6_2 = rb6_2 + x_init_6 # Skip connection    


    # Resblock 7

    x_init_7_1 = rb6_2

    rb7_1 = tf.layers.batch_normalization(x_init_7_1, training=is_training)
    rb7_1 = tf.nn.relu(rb7_1)
    
    rb7_1 = tf.nn.conv2d(rb7_1, params[30], strides=[1, 2, 2, 1], padding='SAME')
    rb7_1 = tf.nn.bias_add(rb7_1, params[31])

    print('Layer 14, Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_7_1.shape, rb7_1.shape,  params[30].shape, params[31].shape))

    x_init_7_2 = tf.nn.conv2d(x_init_7_1, params[32], strides=[1, 2, 2, 1], padding='SAME')
    x_init_7_2 = tf.nn.bias_add(x_init_7_2, params[33])

    print('Layer 14 (skip), Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_7_1.shape, x_init_7_2.shape,  params[32].shape, params[33].shape))
    
    rb7_2 = tf.layers.batch_normalization(rb7_1, training=is_training)
    rb7_2 = tf.nn.relu(rb7_2)
    rb7_2 = tf.nn.conv2d(rb7_2, params[34], strides=[1, 1, 1, 1], padding='SAME')
    rb7_2 = tf.nn.bias_add(rb7_2, params[35])

    print('Layer 15, Input: %s, Output: %s, Weights: %s, biases %s' % (rb7_1.shape, rb7_2.shape,  params[34].shape, params[35].shape))

    rb7_2 = rb7_2 + x_init_7_2 # Skip connection

    # Resblock 8

    x_init_8 = rb7_2

    rb8_1 = tf.layers.batch_normalization(x_init_8, training=is_training)
    rb8_1 = tf.nn.relu(rb8_1)
    rb8_1 = tf.nn.conv2d(rb8_1, params[36], strides=[1, 1, 1, 1], padding='SAME')
    rb8_1 = tf.nn.bias_add(rb8_1, params[37])

    print('Layer 16, Input: %s, Output: %s, Weights: %s, biases %s' % (x_init_8.shape, rb8_1.shape,  params[36].shape, params[37].shape))

    rb8_2 = tf.layers.batch_normalization(rb8_1, training=is_training)
    rb8_2 = tf.nn.relu(rb8_2)
    rb8_2 = tf.nn.conv2d(rb8_2, params[38], strides=[1, 1, 1, 1], padding='SAME')
    rb8_2 = tf.nn.bias_add(rb8_2, params[39])

    print('Layer 17, Input: %s, Output: %s, Weights: %s, biases %s' % (rb8_1.shape, rb8_2.shape,  params[38].shape, params[39].shape))

    rb8_2 = rb8_2 + x_init_8 # Skip connection    


    ## Output layer (18)

    out = tf.layers.batch_normalization(rb8_2, training=is_training)
    out= tf.nn.relu(out)
    out = tf.reduce_mean(out, axis=[1, 2], keepdims=True)

    out = tf.reshape(out, [-1, 256*2])
    out = tf.add(tf.matmul(out, params[40]), params[41])

    print('Layer 18, Input: %s, Output: %s, Weights: %s, biases %s' % (rb8_2.shape, out.shape,  params[40].shape, params[41].shape))

    return out


# Construct model
yhat_logits = model_fun(X, params)
yhat = tf.nn.softmax(yhat_logits)


# Deal with batch normalization

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

bn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope='batch_normalization')


params.extend([bn_variables[i] for i in range(0, 68, 4)]) # Gammas
params.extend([bn_variables[i+1] for i in range(0, 68, 4)]) # Betas


# Define cost and optimizer
cost = cost_fun(y, yhat_logits, params)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) 
train_op = optimizer.minimize(cost)

# Include batch normalization parameters with 
train_op = tf.group([train_op, update_ops])

# For model evaluation
correct_pred = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize variables and session
init = tf.global_variables_initializer()

config = tf.ConfigProto(device_count = {'GPU': 1})
sess = tf.InteractiveSession(config=config)

#sess = tf.InteractiveSession()

# Run the initializer
sess.run(init)

step = 0
num_steps = 285000
_learning_rate=1e-3
batch_size = 100
display_step = 625
B = int(N_train / batch_size)

dCdw_op = tf.concat([tf.reshape(tf.gradients(cost, _params), [-1]) \
                     for _params in params], axis=0)

# # Training loop
# while step < num_steps:
#     batch_x = X_train[step%B*batch_size:(step%B+1)*batch_size]
#     batch_y = y_train[step%B*batch_size:(step%B+1)*batch_size]

#     sess.run(train_op, feed_dict={X: batch_x, y: batch_y, reg_lambda: reg_lambda_val, learning_rate:_learning_rate, is_training:True})
#     if step % display_step == 0:
#         print(str(step))
#         # Calculate batch loss and accuracy
#         # train_cost, train_acc = sess.run([cost, accuracy],
#         #                                       feed_dict={X: batch_x,
#         #                                                  y: batch_y,
#         #                                                  reg_lambda: reg_lambda_val, is_training:True})

#         #val_cost, val_acc = sess.run([cost, accuracy], feed_dict={X:X_test[:1000],
#         #                                                          y:y_test[:1000],
#         #                                                          reg_lambda: reg_lambda_val, is_training:False})
#         # dCdw = 0

#         # for b in range(10):
#         #     dCdw = dCdw + sess.run(dCdw_op, feed_dict={X:X_train[b*batch_size:(b+1)*batch_size], 
#         #                                                y:y_train[b*batch_size:(b+1)*batch_size], 
#         #                                                reg_lambda:reg_lambda_val, is_training:True})
#         # normgrad_train = np.linalg.norm(np.squeeze(dCdw / 10))

#         #print("Step " + str(step) + ", Train. Cost=" + "{:.4f}".format(train_cost) + \
#         #                    ", Train. Acc.= " + "{:.4f}".format(train_acc) + \
#         #                    ", Val. Cost=" + "{:.3f}".format(val_cost) + \
#         #                    ", Val. Acc.= " + "{: 4f}".format(val_acc))# + \
#         #                    #", Norm grad.= " + "{: 4f}".format(normgrad_train))


#     # lr-schedule
#     if step == 55000:
#         _learning_rate = 1e-4
#     if step == 85000:
#         _learning_rate = 1e-5
#     if step == 125000:
#         _learning_rate = 1e-6
#     if step == 155000:
#         _learning_rate = 1e-7
#     if step == 205000:
#         _learning_rate = 1e-8
#     if step == 255000:
#         _learning_rate = 1e-9
        

#     step = step + 1 

# print("Optimization finished.")

# save model
# saver = tf.train.Saver()
# saver.save(sess, 'model_rseed_%d_num_steps_%d_reg_lambda_%.4f.ckpt' % (rseed, num_steps, reg_lambda_val))

# Restore model
# saver = tf.train.Saver()
# saver.restore(sess, 'model_rseed_%d_num_steps_%d_reg_lambda_%.4f.ckpt' % (rseed, num_steps, reg_lambda_val))

# Evaluate model

# batch_size = 100
# B = int(N_train/batch_size)
# preds_train = np.concatenate([sess.run(yhat, 
#                                        feed_dict={X: X_train[b*batch_size:(b+1)*batch_size], is_training:True}) 
#                               for b in tqdm.tqdm(range(B), position=0, leave=True)])
# acc_train = np.mean([sess.run(accuracy, 
#                               feed_dict={X: X_train[b*batch_size:(b+1)*batch_size], 
#                                          y: y_train[b*batch_size:(b+1)*batch_size], is_training:True}) 
#                      for b in tqdm.tqdm(range(B), position=0, leave=True)])

# B = int(N_test/batch_size)
# preds_test = np.concatenate([sess.run(yhat, 
#                                      feed_dict = {X: X_test[b*batch_size:(b+1)*batch_size], is_training:False}) 
#                             for b in tqdm.tqdm(range(B), position=0, leave=True)])

# acc_test = np.mean([sess.run(accuracy, 
#                              feed_dict={X: X_test[b*batch_size:(b+1)*batch_size], 
#                                         y: y_test[b*batch_size:(b+1)*batch_size], is_training:False}) 
#                     for b in tqdm.tqdm(range(B), position=0, leave=True)])

# batch_size = 100
# B = int(N_train/batch_size)
# cost_val = 0
# for b in tqdm.tqdm(range(B), position=0, leave=True):
#     cost_val = cost_val + sess.run(cost, feed_dict={X:X_train[b*batch_size:(b+1)*batch_size], 
#                                                     y:y_train[b*batch_size:(b+1)*batch_size], 
#                                                     reg_lambda:reg_lambda_val, is_training:True})
# cost_val = np.squeeze(cost_val / B)

# cost_train = cost_val

# dCdw = 0

# for b in tqdm.tqdm(range(B), position=0, leave=True):
#     dCdw = dCdw + sess.run(dCdw_op, feed_dict={X:X_train[b*batch_size:(b+1)*batch_size], 
#                                                y:y_train[b*batch_size:(b+1)*batch_size], 
#                                                reg_lambda:reg_lambda_val, is_training:True})
# normgrad_train = np.linalg.norm(np.squeeze(dCdw / B))


# print('Tot. training cost: %.3f' % cost_train)
# print('Norm of gradient: %.3f' % normgrad_train)
# print('Training set accuracy: %.3f%%' % (100*acc_train))
# print('Test set accuracy: %.3f%%' % (100*acc_test))


import pydeepdelta_bn as pydd

_layers = ['0%dwb' % i for i in range(2, 10)] 
_layers.extend(['%dwb' % i for i in range(10, 40)])

dd = pydd.DeepDelta(params, cost, reg_lambda, reg_lambda_val, yhat, X, y,   
                    X_train, y_train, batch_size_H=250, K=200, tfsession=sess, 
                    layers=_layers, rseed=rseed,
                    batch_size_G=200, model_fun=model_fun, cost_fun=cost_fun, 
                    cost_fun_data=cost_fun_data, dropout_rate=dropout_rate, 
                    dropout_rate_val=0.0, batch_size_F=100, logits=False, is_training=is_training, is_training_val=True)

Lambda_H, Q_H, its = dd.compute_eig_H()

np.save('Lambda_H_rseed_%d_layers_ALL_K_%d_num_steps_%d_reg_lambda_%.4f_batch_size_H_%d_is_training_True.npy' 
         % (dd._rseed, 
            #'-'.join(dd.layers), 
            dd.K, 
            num_steps, 
            dd.reg_lambda_val,
            dd.batch_size_H), Lambda_H)

np.save('Q_H_rseed_%d_layers_ALL_K_%d_num_steps_%d_reg_lambda_%.4f_batch_size_H_%d_is_training_True.npy' 
         % (dd._rseed, 
            #'-'.join(dd.layers), 
            dd.K, 
            num_steps, 
            dd.reg_lambda_val,
            dd.batch_size_H), Q_H)

#Lambda_H = np.load('Lambda_H_rseed_%d_layers_ALL_K_%d_num_steps_%d_reg_lambda_%.4f_batch_size_H_%d_is_training_True.npy' 
#        % (dd._rseed, 
#           #'-'.join(dd.layers), 
#           dd.K, 
#           num_steps, 
#           dd.reg_lambda_val,
#           dd.batch_size_H))

# Q_H = np.load('Q_H_rseed_%d_layers_ALL_K_%d_num_steps_%d_reg_lambda_%.4f_batch_size_H_%d_is_training_True.npy' 
#        % (dd._rseed, 
#           #'-'.join(dd.layers), 
#           dd.K, 
#           num_steps, 
#           dd.reg_lambda_val,
#           50000))


dd.init_eig_H_op(Lambda_H, Q_H)

var_H_test = dd.eval_var_H_op(X_test, progress=True, returnError=True, is_training=False)
np.save('var_H_test_K_%d_H_is_training_True_F_is_training_False_H_full_rank_True.npy' % dd.K, var_H_test)
