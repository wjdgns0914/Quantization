# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf
import random
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import importlib
import tensorflow.python.platform
import os
import numpy as np
# from progress.bar import Bar
from datetime import datetime
from tensorflow.python.platform import gfile
from funcData import *
from evaluate import evaluate
from tqdm import tqdm
from tensorflow.python.framework import ops
from models.MNIST0_nodrop import *
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("Datasets/MNIST", one_hot=False)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 64


##LR을 decay시켜주는 함수
def _learning_rate_decay_fn(learning_rate, global_step):
    print("learning_rate_decay_fn is executed!")
    return tf.train.exponential_decay(
        learning_rate,
        global_step,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True)


learning_rate_decay_fn = _learning_rate_decay_fn
# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])  # img 28x28x1 (black/white)
Y = tf.placeholder(tf.int32, [None, ])
logits= model(X_img, is_training=True)
# W1 = tf.get_variable("W1", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
# L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# W2 = tf.get_variable("W2", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
# L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#
# W3 = tf.get_variable("W3", shape=[3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
# L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
# L3 = tf.nn.relu(L3)
# L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
#
# W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625], initializer=tf.contrib.layers.xavier_initializer())
# L4 = tf.nn.relu(tf.matmul(L3_flat, W4))
#
# W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
# logits = tf.matmul(L4, W5)

global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64,
                              initializer=tf.constant_initializer(0),
                              trainable=False)
# define cost/loss & optimizer

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, Y, 1), tf.float32))
# optimizer = tf.contrib.layers.optimize_loss(cost, global_step, learning_rate, 'Adam',
#                                             gradient_noise_scale=None, gradient_multipliers=None,
#                                             clip_gradients=None,  # moving_average_decay=0.9,
#                                             update_ops=None, variables=None, name=None)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
# print("Definite Moving Average...")
MOVING_AVERAGE_DECAY = 0.997
ema = tf.train.ExponentialMovingAverage(
    MOVING_AVERAGE_DECAY, global_step, name='average')
ema_op = ema.apply([cost, accuracy] + tf.trainable_variables())
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

loss_avg = ema.average(cost)
tf.summary.scalar('loss/training', loss_avg)
accuracy_avg = ema.average(accuracy)
tf.summary.scalar('accuracy/training', accuracy_avg)

check_loss = tf. \
    check_numerics(cost, 'model diverged: loss->nan')
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, check_loss)
updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies([optimizer]):
    train_op = tf.group(*updates_collection)
list_W = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='L')
print(list_W)
path="./results/2017-12-5-18-14-34_testvanila/"
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver(var_list=list_W)
saver.restore(sess,tf.train.latest_checkpoint(path))
# train my model
"""
 [<tf.Variable 'L1_Convolution/weight:0' shape=(3, 3, 1, 32) dtype=float32_ref>, 
 <tf.Variable 'L4_Convolution/weight:0' shape=(3, 3, 32, 64) dtype=float32_ref>, 
 <tf.Variable 'L7_Convolution/weight:0' shape=(3, 3, 64, 128) dtype=float32_ref>, 
 <tf.Variable 'L10_FullyConnected/weight:0' shape=(2048, 625) dtype=float32_ref>, 
 <tf.Variable 'L12_FullyConnected/weight:0' shape=(625, 10) dtype=float32_ref>] 
"""
print('Learning started. It takes sometime.')
print(mnist.test.images.shape)
for epoch in range(training_epochs):
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    count_num = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, train_op], feed_dict=feed_dict)
        avg_cost += c / total_batch
        unique_elements, elements_counts = np.unique(batch_ys, return_counts=True)
        num_set = dict(zip(unique_elements, elements_counts))
        for ii in range(10):
            if num_set.__contains__(ii):
                count_num[ii] = count_num[ii] + num_set[ii]
    print(["%d : " % i + str(count_num[i]) for i in range(10)], " Totral num: ", count_num.sum())
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

print('Learning Finished!')

# Test model and check accuracy

# if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py


# Get one and predict
# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Accuracy: 0.1015
['0 : 5442', '1 : 6177', '2 : 5469', '3 : 5635', '4 : 5303', '5 : 4984', '6 : 5414', '7 : 5711', '8 : 5387', '9 : 5454']  Totral num:  54976
Epoch: 0001 cost = 0.140155893
Accuracy: 0.9862
['0 : 5440', '1 : 6179', '2 : 5468', '3 : 5639', '4 : 5304', '5 : 4985', '6 : 5415', '7 : 5715', '8 : 5384', '9 : 5447']  Totral num:  54976
Epoch: 0002 cost = 0.041377177
Accuracy: 0.9891
['0 : 5438', '1 : 6169', '2 : 5468', '3 : 5633', '4 : 5310', '5 : 4984', '6 : 5418', '7 : 5707', '8 : 5392', '9 : 5457']  Totral num:  54976
Epoch: 0003 cost = 0.030085703
Accuracy: 0.9896
['0 : 5447', '1 : 6183', '2 : 5462', '3 : 5636', '4 : 5302', '5 : 4985', '6 : 5411', '7 : 5717', '8 : 5384', '9 : 5449']  Totral num:  54976
Epoch: 0004 cost = 0.021978081
Accuracy: 0.9912
['0 : 5444', '1 : 6165', '2 : 5466', '3 : 5638', '4 : 5307', '5 : 4988', '6 : 5417', '7 : 5710', '8 : 5386', '9 : 5455']  Totral num:  54976
Epoch: 0005 cost = 0.016972911
Accuracy: 0.9926
['0 : 5446', '1 : 6184', '2 : 5475', '3 : 5636', '4 : 5301', '5 : 4986', '6 : 5412', '7 : 5712', '8 : 5388', '9 : 5436']  Totral num:  54976
Epoch: 0006 cost = 0.015309662
Accuracy: 0.9886
['0 : 5435', '1 : 6180', '2 : 5468', '3 : 5627', '4 : 5306', '5 : 4978', '6 : 5412', '7 : 5718', '8 : 5383', '9 : 5469']  Totral num:  54976
Epoch: 0007 cost = 0.013248279
Accuracy: 0.9918
['0 : 5445', '1 : 6174', '2 : 5461', '3 : 5635', '4 : 5307', '5 : 4989', '6 : 5415', '7 : 5717', '8 : 5392', '9 : 5441']  Totral num:  54976
Epoch: 0008 cost = 0.009061189
Accuracy: 0.9926
['0 : 5431', '1 : 6174', '2 : 5468', '3 : 5643', '4 : 5303', '5 : 4994', '6 : 5415', '7 : 5708', '8 : 5385', '9 : 5455']  Totral num:  54976
Epoch: 0009 cost = 0.008864510
Accuracy: 0.9929
'''