
"""
target_level=input('input the target_level')
target_level=float(target_level)
assert (target_level > 1) and (int(target_level) == target_level), \
    'target_level is not valid, please input the integer bigger than 1 '
print('Success:' ,int(target_level))
"""
from scipy.optimize import minimize
import numpy as np
import time
import tensorflow as tf
import customfunc
"""
def rosen(x):
    #The Rosenbrock function
    return 100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0
x0 = np.array([1.3, 0.7])
res = minimize(rosen, x0, method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
print(type(rosen(x0)))

def cost_func(x):
    step_size=x[-2]
    xmin=x[-1]
    xmax = step_size + xmin
    x_bin = tf.round((tf.clip_by_value(x, xmin, xmax) - xmin) / step_size)
    results = step_size * x_bin + xmin
    return results-x
def my_func(x=None):
    if x is None:
        x=np.random.standard_normal([2,])
    result=minimize(cost_func, x, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    result=np.float32(result.x)
    return result

print(np.sinh(x0))
def my_func2(x):
    return np.sinh(x)
x1=tf.Variable(x0)
y=tf.py_func(my_func,[x1],tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3):
        x1array = sess.run(x1)
        result=my_func()
        new_x=np.random.standard_normal(size=[2,])
"""
"""
FLAGS = tf.app.flags.FLAGS
size=[50,50]
a=0.01
b=0.002
fluc_Reset = tf.reshape(tf.distributions.Normal(loc=a, scale=b).sample(1),size)
fluc_Reset2= get_distrib([0.,0.5,0.,1.1],size)
tf.summary.histogram("1",fluc_Reset)
tf.summary.histogram("2",fluc_Reset2)
summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("./result2", graph=sess.graph)
    print(fluc_Reset)
    print(fluc_Reset2)
    for i in range(10):
    a=sess.run(fluc_Reset2)
    print("a===================\n",a)
    c=sess.run(fluc_Reset)
    print("c===================\n",c)
    summary_writer.add_summary(sess.run(summary_op), global_step=i)
  summary_writer.close()
"""
a=5
b=3
customfunc.magic_print('We start training..num of trainable paramaters: %d' %a,b)