"""
    pydeepboot.py

    Parallelized TensorFlow implementation of a Bootstrap ensemble of neural 
    network based MNIST classifiers. Currently supports both static and dynamic 
    random normal weight initialization specified by the RWI_MODE={SRWI|DRWI} 
    flag. The number of bootstrap replicates is set by the variable B. At the 
    end of the pipeline, the mean and standard deviation of the predictions for 
    the training and test set across the B-dimensional replicate space are 
    computed.
    
    This code is used for the Bootstrap calculations in the paper 
    "A Comparison of the Delta Method and the Bootstrap in Deep Learning Classification"
    found at http://arxiv...

    Copyright (c) 2018-2021 by Geir K. Nilsen (geir.kjetil.nilsen@gmail.com)
    and the University of Bergen.
 
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
 
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import tqdm
import numpy as np
from utils import *
import os
import random
from enum import Enum
import tensorflow.compat.v1 as tf
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.disable_v2_behavior()

# Define Random Weight Init Modes
class RWI_MODES(Enum):
    SRWI = 0
    DRWI = 1

# Function to vectorize tensors
def flatten(params):
    return tf.concat([tf.reshape(_params, [-1]) \
                      for _params in params], axis=0)


# Set number of bootstrap replicates
B = 5 

# Set current Random Weight Init Mode, SRWI = Static random weight init (same
# values across replicate space), DRWI = Dynamic random weight init (different
# values across replicate space).
RWI_MODE = RWI_MODES.DRWI

# Import MNIST data & convert 2D images to scaled 1D vectors, and labels to one 
# hot format
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
N_train = X_train.shape[0]
N_test = X_test.shape[0]
X_train = np.float32(np.reshape(X_train, (N_train, 784)) / 255.)
X_test = np.float32(np.reshape(X_test, (N_test, 784)) / 255.)
y_train = to_one_hot(y_train)
y_test = to_one_hot(y_test)

# Parameters
learning_rate = tf.placeholder(tf.float32) # used for lr-schedule
reg_lambda = tf.placeholder(tf.float32) # used for L2-regularization
reg_lambda_val = 0.01 # l2-reg rate factor
num_steps = 90000 # number of training steps
T1 = 784 # num input features
TL = 10 # num outputs

# Generated indexing scheme to avoid explicit formation of the bootstrapped 
# datasets, O(B*T1*N) + O(B*TL*N) => O(B*N) space
indices = np.zeros((B, N_train), dtype='int32')
print('Generating bootstrap replicate indexing scheme...')
for b in tqdm.tqdm(range(0, B)):
    if RWI_MODE is RWI_MODES.DRWI:
        rseed=b
    elif RWI_MODE is RWI_MODES.SRWI:
        rseed=0
    np.random.seed(rseed)
    random.seed(rseed)
    tf.random.set_random_seed(seed=rseed)
    indices[b] = np.random.choice(X_train.shape[0], size=N_train, replace=True)
    
        
# Init weights (and biases) according to current mode
weights = np.array([])
biases = np.array([])
params = []
print('Initializing weights and biases...')
for b in tqdm.tqdm(range(0, B)):
    # Random weight init
    if RWI_MODE is RWI_MODES.DRWI:
        rseed=b
    elif RWI_MODE is RWI_MODES.SRWI:
        rseed=0
    np.random.seed(rseed)
    random.seed(rseed)
    tf.random.set_random_seed(seed=rseed)

    weights = np.append(weights, {
        'wc1': tf.Variable(np.random.normal(size=(3, 3, 1, 32)), dtype='float32'),
        'wc2': tf.Variable(np.random.normal(size=(3, 3, 32, 64)), dtype='float32'),
        'wc3': tf.Variable(np.random.normal(size=(3, 3, 64, 64)), dtype='float32'),
        'wd1': tf.Variable(np.random.normal(size=(3 * 3 * 64, 64)), dtype='float32'),
        'out': tf.Variable(np.random.normal(size=(64, TL)), dtype='float32')
    })

    # Biases init to zero
    biases = np.append(biases, {
        'bc1': tf.Variable(tf.zeros((32)), dtype='float32'),
        'bc2': tf.Variable(tf.zeros((64)), dtype='float32'),
        'bc3': tf.Variable(tf.zeros((64)), dtype='float32'),
        'bd1': tf.Variable(tf.zeros((64)), dtype='float32'),
        'out': tf.Variable(tf.zeros((TL)), dtype='float32')
    })

    # Stack parameters layer-wise, weights first then biases
    params.append([weights[b]['wc1'], biases[b]['bc1'], 
                  weights[b]['wc2'], biases[b]['bc2'], 
                  weights[b]['wc3'], biases[b]['bc3'], 
                  weights[b]['wd1'], biases[b]['bd1'], 
                  weights[b]['out'], biases[b]['out']])

# Model input/output
X = [tf.placeholder(dtype='float32', shape=(None, T1), name="X-data") for b in range(0, B)]
y = [tf.placeholder(dtype='float32', shape=(None, TL), name="y-data") for b in range(0, B)]

# Cost function
def cost_fun(y, yhat_logits, params):
    cost = tf.losses.softmax_cross_entropy(y, yhat_logits) + \
    reg_lambda/2.0*tf.reduce_sum([tf.reduce_sum(tf.pow(params[x], 2.0)) \
                                  for x in range(len(params))])
    return cost

# Model function
def model_fun(x, params):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, params[0], params[1])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, params[2], params[3])
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, params[4], params[5])

    fc1 = tf.reshape(conv3, [-1, 3*3*64])
    fc1 = tf.add(tf.matmul(fc1, params[6]), params[7])
    fc1 = tf.nn.relu(fc1)

    out = tf.add(tf.matmul(fc1, params[8]), params[9])
    return out

# Setup parallel optimizers
yhat_logits = []
yhat = []
cost = []
optimizer = []
train_op = []
correct_pred = []
accuracy = []
print('Setting up parallel optimizers...')
for b in tqdm.tqdm(range(0, B)):
    # Construct models
    yhat_logits.append(model_fun(X[b], params[b]))
    yhat.append(tf.nn.softmax(yhat_logits[b]))
    # Construct cost functions and optimizers
    cost.append(cost_fun(y[b], yhat_logits[b], params[b]))
    optimizer.append(tf.train.AdamOptimizer(learning_rate=learning_rate))
    train_op.append(optimizer[b].minimize(cost[b]))
    # For model evaluation
    correct_pred.append(tf.equal(tf.argmax(yhat[b], 1), tf.argmax(y[b], 1)))
    accuracy.append(tf.reduce_mean(tf.cast(correct_pred[b], tf.float32)))

# Initialize variables and session
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

# Run the initializer
sess.run(init)

# Compute number of params per bootstrapped network
P = flatten(params[0]).get_shape().as_list()[0]

# Verify random weight init according to current mode
if RWI_MODE is RWI_MODES.SRWI:
    print('Verifying static random weight initialization...')
    w = np.array(sess.run([tf.concat([tf.reshape(__params, [-1]) for __params in _params], axis=0) for _params in params]))
    for b in range(1,B):
        assert np.equal(w[0], w[b]).sum() == P, 'Static random weight init. failed!'
elif RWI_MODE is RWI_MODES.DRWI:
    print('Verifying dynamic random weight initialization...')
    w = np.array(sess.run([tf.concat([tf.reshape(__params, [-1]) for __params in _params], axis=0) for _params in params]))
    for b in range(1,B):
        assert np.equal(w[0], w[b]).sum() != P, 'Dynamic random weight init. failed!'

# Train networks in parallel
step = 0
num_steps = 90000
_learning_rate=10e-4
batch_size = 100
display_step = 7500
B_train = int(N_train / batch_size)
B_test = int(N_test / batch_size)

while step < num_steps:  
    batch_x = [X_train[indices[b][step%B_train*batch_size:(step%B_train+1)*batch_size]] for b in range(0, B)]
    batch_y = [y_train[indices[b][step%B_train*batch_size:(step%B_train+1)*batch_size]] for b in range(0, B)]

    fd_train = {**{X[b]:batch_x[b] for b in range(0,B)}, 
                **{y[b]:batch_y[b] for b in range(0,B)}, 
                **{reg_lambda: reg_lambda_val, learning_rate:_learning_rate}}
   
    sess.run(train_op, feed_dict=fd_train)
    
    if step % display_step == 0:
        train_cost_acc = sess.run([cost, accuracy], feed_dict=fd_train)

        fd_test = {**{X[b]:X_test[step%B_test*batch_size:(step%B_test+1)*batch_size] for b in range(0,B)}, 
                   **{y[b]:y_test[step%B_test*batch_size:(step%B_test+1)*batch_size] for b in range(0,B)}, 
                   **{reg_lambda: reg_lambda_val}}

        val_cost_acc = sess.run([cost, accuracy], feed_dict=fd_test)
     
        print('Step ' + str(step))
        print('Train [cost] [accuracy]: %s' % train_cost_acc)
        print('Test [cost] [accuracy]: %s' % val_cost_acc)
    
    # lr-schedule    
    if step == 60000:
        _learning_rate = 10e-5
    if step == 70000:
        _learning_rate = 10e-6
    if step == 80000:
        _learning_rate = 10e-7

    step = step + 1 

print("Optimization finished.")
   
# Save model
# saver = tf.train.Saver()
# saver.save(sess, 'model.ckpt')

# Compute and display training stats
preds_train = np.zeros((B, N_train, TL))
preds_test = np.zeros((B, N_test, TL))
acc_train = np.zeros((B))
acc_test = np.zeros((B))
weights_bs = np.zeros((B, P))
cost_train = np.zeros((B))
normgrad_train = np.zeros((B))

batch_size = 100
Bs = int(N_train/batch_size)

preds_train = np.concatenate([sess.run(yhat, feed_dict={X[b]:X_train[bs*batch_size:(bs+1)*batch_size] for b in range(0,B)}) 
                              for bs in tqdm.tqdm(range(Bs), position=0, leave=True)], axis=1)

acc_train = np.mean(np.vstack([sess.run(accuracy, feed_dict={**{X[b]: X_train[bs*batch_size:(bs+1)*batch_size] for b in range(0,B)},
                                                             **{y[b]: y_train[bs*batch_size:(bs+1)*batch_size] for b in range(0,B)}})
                               for bs in tqdm.tqdm(range(Bs), position=0, leave=True)]),axis=0)

Bs = int(N_test/batch_size)

preds_test = np.concatenate([sess.run(yhat, feed_dict={X[b]:X_test[bs*batch_size:(bs+1)*batch_size] for b in range(0,B)}) 
                             for bs in tqdm.tqdm(range(Bs), position=0, leave=True)], axis=1)

acc_test = np.mean(np.vstack([sess.run(accuracy, feed_dict={**{X[b]: X_test[bs*batch_size:(bs+1)*batch_size] for b in range(0,B)},
                                                            **{y[b]: y_test[bs*batch_size:(bs+1)*batch_size] for b in range(0,B)}})
                              for bs in tqdm.tqdm(range(Bs), position=0, leave=True)]),axis=0)

    
Bs = int(N_train/batch_size)
cost_train = 0
for bs in tqdm.tqdm(range(Bs), position=0, leave=True):
    cost_train = cost_train + np.array(sess.run(cost, feed_dict={**{X[b]: X_train[indices[b][bs*batch_size:(bs+1)*batch_size]] for b in range(0,B)},
                                                                 **{y[b]: y_train[indices[b][bs*batch_size:(bs+1)*batch_size]] for b in range(0,B)},
                                                                 **{reg_lambda: reg_lambda_val}}))
cost_train = np.squeeze(cost_train / Bs)

dCdw_op = [tf.concat([tf.reshape(tf.gradients(cost, __params), [-1]) for __params in _params], axis=0) for _params in params]
dCdw = 0
for bs in tqdm.tqdm(range(Bs), position=0, leave=True):
    dCdw = dCdw + np.array(sess.run(dCdw_op, feed_dict={**{X[b]: X_train[indices[b][bs*batch_size:(bs+1)*batch_size]] for b in range(0,B)},
                                                        **{y[b]: y_train[indices[b][bs*batch_size:(bs+1)*batch_size]] for b in range(0,B)},
                                                        **{reg_lambda: reg_lambda_val}}))

normgrad_train = np.linalg.norm(np.squeeze(dCdw / Bs), axis=1)

print('Tot. training cost: %s' % cost_train)
print('Norm of gradient: %s' % normgrad_train)

print('Training set accuracy: %s%%' % (100*acc_train))
print('Test set accuracy: %s%%' % (100*acc_test))

# Finally, compute mean and standard deviation of predictions
mean_preds_train = np.mean(preds_train, axis=0)
mean_preds_test = np.mean(preds_test, axis=0)
std_preds_train = np.std(preds_train, axis=0)
std_preds_test = np.std(preds_test, axis=0)



