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
from data import *
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
X = tf.placeholder(tf.float32, [batch_size, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.int32, [batch_size,])
logits=model(X_img)
# W1 = tf.get_variable("W1", shape=[3, 3, 1, 32],initializer=tf.contrib.layers.xavier_initializer())
# L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#
# W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],initializer=tf.contrib.layers.xavier_initializer())
# L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
# L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
#
# W3 = tf.get_variable("W3", shape=[3, 3,64,128],initializer=tf.contrib.layers.xavier_initializer())
# L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
# L3 = tf.nn.relu(L3)
# L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
#
# W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],initializer=tf.contrib.layers.xavier_initializer())
# L4 = tf.nn.relu(tf.matmul(L3_flat, W4))
#
# W5 = tf.get_variable("W5", shape=[625, 10],initializer=tf.contrib.layers.xavier_initializer())
# logits = tf.matmul(L4, W5)


global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64,
                              initializer=tf.constant_initializer(0),
                              trainable=False)
# define cost/loss & optimizer

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, Y, 1), tf.float32))
optimizer = tf.contrib.layers.optimize_loss(cost, global_step, learning_rate, 'Adam',
                                      gradient_noise_scale=None, gradient_multipliers=None,
                                      clip_gradients=None,  # moving_average_decay=0.9,
                                      update_ops=None, variables=None, name=None)
# initialize
print("Definite Moving Average...")
MOVING_AVERAGE_DECAY = 0.997
ema = tf.train.ExponentialMovingAverage(
    MOVING_AVERAGE_DECAY, global_step, name='average')
ema_op = ema.apply([cost, accuracy] + tf.trainable_variables())
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

loss_avg = ema.average(cost)
tf.summary.scalar('loss/training', loss_avg)
accuracy_avg = ema.average(accuracy)
tf.summary.scalar('accuracy/training', accuracy_avg)

check_loss = tf.\
    check_numerics(cost, 'model diverged: loss->nan')
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, check_loss)
updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies([optimizer]):
    train_op = tf.group(*updates_collection)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    count_num=np.array([0,0,0,0,0,0,0,0,0,0])
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
Learning stared. It takes sometime.
Epoch: 0001 cost = 0.385748474
Epoch: 0002 cost = 0.092017397
Epoch: 0003 cost = 0.065854684
Epoch: 0004 cost = 0.055604566
Epoch: 0005 cost = 0.045996377
Epoch: 0006 cost = 0.040913645
Epoch: 0007 cost = 0.036924479
Epoch: 0008 cost = 0.032808939
Epoch: 0009 cost = 0.031791007
Epoch: 0010 cost = 0.030224456
Epoch: 0011 cost = 0.026849916
Epoch: 0012 cost = 0.026826763
Epoch: 0013 cost = 0.027188021
Epoch: 0014 cost = 0.023604777
Epoch: 0015 cost = 0.024607201
Learning Finished!
Accuracy: 0.9938
'''